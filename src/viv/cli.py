from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from viv.config import (
    enabled_generation_configs,
    get_generation_config,
    load_prompts,
    load_yaml,
    merged_request,
)


LATENT_NAMES = ("initial", "pre_boundary", "final")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="viv")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_p = sub.add_parser("compile-plan", help="Build deterministic generation jobs.jsonl")
    compile_p.add_argument("--config", default="configs/experiment.yaml")
    compile_p.add_argument("--prompts", default="configs/prompts.pilot.jsonl")
    compile_p.add_argument("--run-id")
    compile_p.add_argument("--artifact-root")
    compile_p.add_argument("--output")
    compile_p.add_argument("--prompt-limit", type=int)
    compile_p.add_argument("--include-disabled", action="store_true")
    compile_p.add_argument("--check", action="store_true")
    compile_p.set_defaults(func=cmd_compile_plan)

    generate_p = sub.add_parser("generate", help="Run jobs against a vLLM-Omni video server")
    generate_p.add_argument("--jobs", required=True)
    generate_p.add_argument("--server-url", default="http://127.0.0.1:8091")
    generate_p.add_argument("--config-id")
    generate_p.add_argument("--shard-index", type=int, default=0)
    generate_p.add_argument("--shard-count", type=int, default=1)
    generate_p.add_argument("--force", action="store_true")
    generate_p.add_argument("--no-require-latents", action="store_true")
    generate_p.add_argument("--timeout", type=int, default=7200)
    generate_p.set_defaults(func=cmd_generate)

    inspect_p = sub.add_parser("inspect-run", help="Inspect a run directory")
    inspect_p.add_argument("--run-dir")
    inspect_p.add_argument("--run-id")
    inspect_p.add_argument("--artifact-root", default="/workspace/runs")
    inspect_p.add_argument("--jobs")
    inspect_p.add_argument("--no-require-latents", action="store_true")
    inspect_p.set_defaults(func=cmd_inspect_run)

    sync_p = sub.add_parser("sync-artifacts", help="Print or run an aws s3 sync command")
    sync_p.add_argument("--run-id", required=True)
    sync_p.add_argument("--datacenter", required=True)
    sync_p.add_argument("--network-volume-id", required=True)
    sync_p.add_argument("--artifact-root", default="/workspace/runs")
    sync_p.add_argument("--remote-prefix", default="runs")
    sync_p.add_argument("--execute", action="store_true")
    sync_p.set_defaults(func=cmd_sync_artifacts)

    serve_p = sub.add_parser("serve", help="Exec vLLM-Omni with env vars for one generation config")
    serve_p.add_argument("--config", default="configs/experiment.yaml")
    serve_p.add_argument("--config-id", required=True)
    serve_p.add_argument("--run-id", required=True)
    serve_p.add_argument("--artifact-root")
    serve_p.add_argument("--port", type=int, default=8091)
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"viv: error: {exc}", file=sys.stderr)
        return 2


def cmd_compile_plan(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    prompts_path = Path(args.prompts)
    config = load_yaml(config_path)
    prompts = load_prompts(prompts_path)
    if args.prompt_limit is not None:
        prompts = prompts[: args.prompt_limit]

    generation_configs = enabled_generation_configs(config, include_disabled=args.include_disabled)
    defaults = dict(config.get("request_defaults") or {})
    artifact_root = Path(args.artifact_root or config.get("artifact_root") or "/workspace/runs")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("pilot-%Y%m%dT%H%M%SZ")
    run_dir = artifact_root / run_id
    output = Path(args.output) if args.output else run_dir / "manifest" / "jobs.jsonl"

    jobs = []
    for generation_config in generation_configs:
        config_id = str(generation_config["id"])
        for prompt in prompts:
            request = merged_request(defaults, generation_config, prompt)
            job_key = f"{run_id}:{config_id}:{prompt.prompt_id}"
            job_id = hashlib.sha256(job_key.encode("utf-8")).hexdigest()[:24]
            jobs.append(
                {
                    "job_id": job_id,
                    "run_id": run_id,
                    "config_id": config_id,
                    "prompt_id": prompt.prompt_id,
                    "prompt": prompt.prompt,
                    "seed": int(request["seed"]),
                    "request": request,
                    "generation_config": generation_config,
                    "artifact_relpath": f"artifacts/{config_id}/{prompt.prompt_id}",
                }
            )

    print(
        f"config={config_path} prompts={len(prompts)} configs={len(generation_configs)} jobs={len(jobs)}"
    )
    if args.check:
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for job in jobs:
            fh.write(json.dumps(job, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {output}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    jobs_path = Path(args.jobs)
    jobs = load_jobs(jobs_path)
    run_dir = jobs_path.parent.parent
    log_dir = run_dir / "logs" / (args.config_id or "all")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard-{args.shard_index:04d}-of-{args.shard_count:04d}.jsonl"
    selected = select_jobs(jobs, args.config_id, args.shard_index, args.shard_count)
    if not selected:
        print("no jobs selected")
        return 0

    print(f"selected {len(selected)} jobs from {jobs_path}")
    failures = 0
    with log_path.open("a", encoding="utf-8") as log_fh:
        for job in selected:
            artifact_dir = run_dir / job["artifact_relpath"]
            if is_complete(artifact_dir, require_latents=not args.no_require_latents) and not args.force:
                write_log(log_fh, job, "skipped", {"reason": "complete"})
                print(f"skip complete {job['config_id']}/{job['prompt_id']}")
                continue
            try:
                run_job(job, run_dir, args.server_url, args.timeout, require_latents=not args.no_require_latents)
                write_log(log_fh, job, "completed", {})
                print(f"completed {job['config_id']}/{job['prompt_id']}")
            except Exception as exc:
                failures += 1
                write_log(log_fh, job, "failed", {"error": str(exc)})
                print(f"failed {job['config_id']}/{job['prompt_id']}: {exc}", file=sys.stderr)
    print(f"log {log_path}")
    return 1 if failures else 0


def cmd_inspect_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.artifact_root) / args.run_id
    jobs_path = Path(args.jobs) if args.jobs else run_dir / "manifest" / "jobs.jsonl"
    jobs = load_jobs(jobs_path)
    counts = {"complete": 0, "missing": 0}
    missing: list[str] = []
    for job in jobs:
        artifact_dir = run_dir / job["artifact_relpath"]
        if is_complete(artifact_dir, require_latents=not args.no_require_latents):
            counts["complete"] += 1
        else:
            counts["missing"] += 1
            missing.append(f"{job['config_id']}/{job['prompt_id']}")
    print(json.dumps({"run_dir": str(run_dir), "jobs": len(jobs), **counts}, indent=2))
    if missing:
        print("missing:")
        for item in missing[:200]:
            print(f"  {item}")
        if len(missing) > 200:
            print(f"  ... {len(missing) - 200} more")
        return 1
    return 0


def cmd_sync_artifacts(args: argparse.Namespace) -> int:
    local_dir = Path(args.artifact_root) / args.run_id
    endpoint = f"https://s3api-{args.datacenter.lower()}.runpod.io/"
    remote = f"s3://{args.network_volume_id}/{args.remote_prefix.strip('/')}/{args.run_id}"
    command = [
        "aws",
        "s3",
        "sync",
        "--region",
        args.datacenter,
        "--endpoint-url",
        endpoint,
        str(local_dir),
        remote,
    ]
    print(" ".join(command))
    if args.execute:
        return subprocess.call(command)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    config = load_yaml(Path(args.config))
    generation_config = get_generation_config(config, args.config_id)
    artifact_root = Path(args.artifact_root or config.get("artifact_root") or "/workspace/runs")
    server = generation_config.get("server") or {}
    env = os.environ.copy()
    env["VIV_CAPTURE_LATENTS"] = "1"
    env["VIV_CONFIG_ID"] = args.config_id
    env["VIV_LATENT_ROOT"] = str(artifact_root / args.run_id / "artifacts")
    env.setdefault("VIV_LATENT_DTYPE", str(config.get("latent_dtype", "float16")))
    if config.get("hf_home"):
        env.setdefault("HF_HOME", str(config["hf_home"]))

    model = str(config.get("model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"))
    command = ["vllm", "serve", model, "--omni", "--port", str(args.port)]
    if server.get("tensor_parallel_size"):
        command += ["--tensor-parallel-size", str(server["tensor_parallel_size"])]
    if server.get("boundary_ratio") is not None:
        command += ["--boundary-ratio", str(server["boundary_ratio"])]
    if server.get("flow_shift") is not None:
        command += ["--flow-shift", str(server["flow_shift"])]
    cache_backend = server.get("cache_backend")
    if cache_backend and cache_backend != "none":
        command += ["--cache-backend", str(cache_backend)]
    if server.get("enable_cache_dit_summary"):
        command.append("--enable-cache-dit-summary")

    print("exec:", " ".join(command), flush=True)
    os.execvpe(command[0], command, env)
    return 0


def load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                jobs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not jobs:
        raise ValueError(f"{path}: no jobs found")
    return jobs


def select_jobs(
    jobs: list[dict[str, Any]], config_id: str | None, shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    filtered = [job for job in jobs if not config_id or job["config_id"] == config_id]
    selected = []
    for index, job in enumerate(filtered):
        if index % shard_count == shard_index:
            selected.append(job)
    return selected


def run_job(job: dict[str, Any], run_dir: Path, server_url: str, timeout: int, require_latents: bool) -> None:
    artifact_dir = run_dir / job["artifact_relpath"]
    latents_dir = artifact_dir / "latents"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    payload = stringify_request(job["request"])
    started = time.time()
    response = requests.post(
        server_url.rstrip("/") + "/v1/videos/sync",
        files=[(key, (None, value)) for key, value in payload.items()],
        timeout=timeout,
    )
    response.raise_for_status()
    elapsed = time.time() - started

    video_tmp = artifact_dir / "video.mp4.tmp"
    video_path = artifact_dir / "video.mp4"
    video_tmp.write_bytes(response.content)
    video_tmp.replace(video_path)
    move_spooled_latents(run_dir, job, latents_dir)
    if require_latents:
        missing = [name for name in LATENT_NAMES if not (latents_dir / f"{name}.safetensors").exists()]
        if missing:
            raise RuntimeError(f"missing latent checkpoint(s): {', '.join(missing)}")

    checksums = write_checksums(artifact_dir)
    metadata = {
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "job": job,
        "server_url": server_url,
        "response_headers": dict(response.headers),
        "runtime": runtime_fingerprint(),
        "checksums": checksums,
    }
    write_json(artifact_dir / "metadata.json", metadata)


def stringify_request(request: dict[str, Any]) -> dict[str, str]:
    payload = {}
    for key, value in request.items():
        if value is None:
            continue
        if isinstance(value, bool):
            payload[key] = "true" if value else "false"
        else:
            payload[key] = str(value)
    return payload


def move_spooled_latents(run_dir: Path, job: dict[str, Any], final_latents_dir: Path) -> None:
    spool = run_dir / "artifacts" / job["config_id"] / str(job["seed"]) / "latents"
    if not spool.exists() or spool.resolve() == final_latents_dir.resolve():
        return
    for name in LATENT_NAMES:
        src = spool / f"{name}.safetensors"
        if src.exists():
            shutil.move(str(src), final_latents_dir / src.name)
    spool_meta = spool / "metadata.json"
    if spool_meta.exists():
        shutil.move(str(spool_meta), final_latents_dir / "latent_capture_metadata.json")


def is_complete(artifact_dir: Path, require_latents: bool) -> bool:
    if not (artifact_dir / "video.mp4").exists():
        return False
    if not (artifact_dir / "metadata.json").exists():
        return False
    if require_latents:
        for name in LATENT_NAMES:
            if not (artifact_dir / "latents" / f"{name}.safetensors").exists():
                return False
    return True


def write_checksums(artifact_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(p for p in artifact_dir.rglob("*") if p.is_file() and p.name != "checksums.json"):
        rel = path.relative_to(artifact_dir).as_posix()
        checksums[rel] = sha256_file(path)
    write_json(artifact_dir / "checksums.json", checksums)
    return checksums


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_log(fh: Any, job: dict[str, Any], status: str, extra: dict[str, Any]) -> None:
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "job_id": job["job_id"],
        "config_id": job["config_id"],
        "prompt_id": job["prompt_id"],
        **extra,
    }
    fh.write(json.dumps(row, sort_keys=True) + "\n")
    fh.flush()


def runtime_fingerprint() -> dict[str, Any]:
    return {
        "hostname": run_capture(["hostname"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "python": sys.version,
        "torch": run_capture(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())",
            ]
        ),
        "vllm": run_capture(["vllm", "--version"]),
    }


def run_capture(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
