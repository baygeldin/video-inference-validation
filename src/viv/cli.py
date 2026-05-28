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

from viv.config import (
    enabled_generation_configs,
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

    generate_p = sub.add_parser("generate", help="Run jobs with vLLM-Omni offline inference")
    generate_p.add_argument("--jobs", required=True)
    generate_p.add_argument("--config-id")
    generate_p.add_argument("--shard-index", type=int, default=0)
    generate_p.add_argument("--shard-count", type=int, default=1)
    generate_p.add_argument("--force", action="store_true")
    generate_p.add_argument("--no-require-latents", action="store_true")
    generate_p.set_defaults(func=cmd_generate)

    inspect_p = sub.add_parser("inspect-run", help="Inspect a run directory")
    inspect_p.add_argument("--run-dir")
    inspect_p.add_argument("--run-id")
    inspect_p.add_argument("--artifact-root", default="/workspace/runs")
    inspect_p.add_argument("--jobs")
    inspect_p.add_argument("--no-require-latents", action="store_true")
    inspect_p.set_defaults(func=cmd_inspect_run)

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
                    "model": str(config.get("model", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")),
                    "hf_home": config.get("hf_home"),
                    "latent_dtype": str(config.get("latent_dtype", "float16")),
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
    generators: dict[str, OfflineVideoGenerator] = {}
    with log_path.open("a", encoding="utf-8") as log_fh:
        for job in selected:
            artifact_dir = run_dir / job["artifact_relpath"]
            if is_complete(artifact_dir, require_latents=not args.no_require_latents) and not args.force:
                write_log(log_fh, job, "skipped", {"reason": "complete"})
                print(f"skip complete {job['config_id']}/{job['prompt_id']}")
                continue
            try:
                generator = generators.get(job["config_id"])
                if generator is None:
                    generator = OfflineVideoGenerator(job, run_dir)
                    generators[job["config_id"]] = generator
                run_job(job, run_dir, generator, require_latents=not args.no_require_latents)
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


class OfflineVideoGenerator:
    def __init__(self, first_job: dict[str, Any], run_dir: Path) -> None:
        generation_config = first_job["generation_config"]
        engine = generation_config.get("engine") or {}
        request = first_job["request"]
        model = str(first_job.get("model") or "Wan-AI/Wan2.2-T2V-A14B-Diffusers")

        os.environ["VIV_CAPTURE_LATENTS"] = "1"
        os.environ["VIV_CONFIG_ID"] = first_job["config_id"]
        os.environ["VIV_LATENT_ROOT"] = str(run_dir / "artifacts")
        os.environ.setdefault("VIV_LATENT_DTYPE", str(first_job.get("latent_dtype") or "float16"))
        if first_job.get("hf_home"):
            os.environ.setdefault("HF_HOME", str(first_job["hf_home"]))

        from viv.vllm_omni_monkey_patch import install as install_viv_patch

        install_viv_patch()

        from vllm_omni.diffusion.data import DiffusionParallelConfig
        from vllm_omni.entrypoints.omni import Omni

        cache_backend = engine.get("cache_backend")
        if cache_backend == "none":
            cache_backend = None

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=int(engine.get("tensor_parallel_size", 1) or 1),
            cfg_parallel_size=int(engine.get("cfg_parallel_size", 1) or 1),
            ulysses_degree=int(engine.get("ulysses_degree", 1) or 1),
            ring_degree=int(engine.get("ring_degree", 1) or 1),
            vae_patch_parallel_size=int(engine.get("vae_patch_parallel_size", 1) or 1),
            enable_expert_parallel=bool(engine.get("enable_expert_parallel", False)),
        )
        omni_kwargs: dict[str, Any] = {
            "model": model,
            "parallel_config": parallel_config,
            "cache_backend": cache_backend,
            "cache_config": _cache_dit_config() if cache_backend == "cache_dit" else None,
            "enable_cache_dit_summary": bool(engine.get("enable_cache_dit_summary", False)),
        }
        if engine.get("boundary_ratio") is not None:
            omni_kwargs["boundary_ratio"] = float(engine["boundary_ratio"])
        elif request.get("boundary_ratio") is not None:
            omni_kwargs["boundary_ratio"] = float(request["boundary_ratio"])
        if engine.get("flow_shift") is not None:
            omni_kwargs["flow_shift"] = float(engine["flow_shift"])
        elif request.get("flow_shift") is not None:
            omni_kwargs["flow_shift"] = float(request["flow_shift"])
        if engine.get("quantization") is not None:
            omni_kwargs["quantization"] = engine["quantization"]

        self.model = model
        self.engine = engine
        self.omni = Omni(**omni_kwargs)

    def generate(self, job: dict[str, Any], video_path: Path) -> None:
        import torch
        from diffusers.utils import export_to_video
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        request = job["request"]
        os.environ["VIV_CONFIG_ID"] = job["config_id"]

        prompt: dict[str, Any] = {"prompt": request["prompt"]}
        if request.get("negative_prompt"):
            prompt["negative_prompt"] = request["negative_prompt"]

        generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(int(request["seed"]))
        sampling_kwargs = {
            "height": int(request["height"]),
            "width": int(request["width"]),
            "generator": generator,
            "guidance_scale": float(request["guidance_scale"]),
            "num_inference_steps": int(request["num_inference_steps"]),
            "num_frames": int(request["num_frames"]),
        }
        if request.get("guidance_scale_2") is not None:
            sampling_kwargs["guidance_scale_2"] = float(request["guidance_scale_2"])

        output = self.omni.generate(prompt, OmniDiffusionSamplingParams(**sampling_kwargs))
        frames = extract_video_frames(output)
        video_tmp = video_path.with_suffix(video_path.suffix + ".tmp")
        export_to_video(frames, str(video_tmp), fps=int(request.get("fps", 16)))
        video_tmp.replace(video_path)


def run_job(job: dict[str, Any], run_dir: Path, generator: OfflineVideoGenerator, require_latents: bool) -> None:
    artifact_dir = run_dir / job["artifact_relpath"]
    latents_dir = artifact_dir / "latents"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    video_path = artifact_dir / "video.mp4"
    generator.generate(job, video_path)
    elapsed = time.time() - started

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
        "inference_mode": "offline",
        "runtime": runtime_fingerprint(),
        "checksums": checksums,
    }
    write_json(artifact_dir / "metadata.json", metadata)


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


def extract_video_frames(output: Any) -> Any:
    import numpy as np
    import torch
    from vllm_omni.outputs import OmniRequestOutput

    frames = output[0] if isinstance(output, list) and output else output
    if isinstance(frames, OmniRequestOutput):
        if frames.is_pipeline_output and frames.request_output is not None:
            frames = frames.request_output
        if isinstance(frames, OmniRequestOutput):
            if not frames.images:
                raise ValueError("No video frames found in OmniRequestOutput")
            frames = frames.images[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]
    if isinstance(frames, dict):
        frames = frames.get("frames") or frames.get("video")
    if isinstance(frames, list) and len(frames) == 1:
        first = frames[0]
        if isinstance(first, tuple) and len(first) == 2:
            frames = first[0]
        elif isinstance(first, dict):
            frames = first.get("frames") or first.get("video")
        elif isinstance(first, list):
            frames = first
    if frames is None:
        raise ValueError("No video frames found in output")
    if isinstance(frames, torch.Tensor):
        video = frames.detach().cpu()
        if video.dim() == 5:
            video = video[0]
        if video.dim() == 4 and video.shape[0] in (3, 4):
            video = video.permute(1, 2, 3, 0)
        if video.is_floating_point():
            video = video.clamp(-1, 1) * 0.5 + 0.5
        return list(video.float().numpy())
    if isinstance(frames, np.ndarray):
        video_array = frames[0] if frames.ndim == 5 else frames
        if np.issubdtype(video_array.dtype, np.integer):
            video_array = video_array.astype(np.float32) / 255.0
        return list(video_array)
    return frames


def _cache_dit_config() -> dict[str, Any]:
    return {
        "Fn_compute_blocks": 1,
        "Bn_compute_blocks": 0,
        "max_warmup_steps": 4,
        "max_cached_steps": 20,
        "residual_diff_threshold": 0.24,
        "max_continuous_cached_steps": 3,
        "enable_taylorseer": False,
        "taylorseer_order": 1,
        "scm_steps_mask_policy": None,
        "scm_steps_policy": "dynamic",
    }


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
