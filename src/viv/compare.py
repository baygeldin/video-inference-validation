from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


COMPARISON_FILENAME = "comparison.json"
FINAL_LATENT_SUFFIX = ".final_noise_latent.safetensors"
PREDICTION_LATENT_PATTERN = re.compile(r"\.denoising_step_(?P<step>\d+)\.safetensors$")
VIDEO_SUFFIX = ".mp4"
_SSIM_SCORE_PATTERN = re.compile(r"\bAll:(?P<score>[-+0-9.eE]+)")


@dataclass(frozen=True)
class GenerationArtifacts:
    gpu_model: str | None
    config_name: str
    examples: dict[str, ExampleArtifacts]


@dataclass(frozen=True)
class ExampleArtifacts:
    final_latent_path: Path
    video_path: Path
    prediction_latent_paths: dict[int, Path]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int


def compare_generations(
    output_dir: Path,
    baseline_dir: Path,
    generation_dirs: list[Path],
) -> Path:
    if not generation_dirs:
        raise ValueError("at least one generation path is required")

    baseline = _read_generation_artifacts(baseline_dir)
    generations = [_read_generation_artifacts(path) for path in generation_dirs]
    common_prompt_ids = sorted(
        set.intersection(
            set(baseline.examples),
            *(set(generation.examples) for generation in generations),
        )
    )

    output = {
        "baseline": {
            "gpu_model": baseline.gpu_model,
            "config_name": baseline.config_name,
        },
        "generations": [
            {
                "gpu_model": generation.gpu_model,
                "config_name": generation.config_name,
                "examples": [
                    {
                        "prompt_id": prompt_id,
                        "video_file": _video_file_metrics(
                            baseline.examples[prompt_id].video_path,
                            generation.examples[prompt_id].video_path,
                        ),
                        "final_latent": _latent_metrics(
                            baseline.examples[prompt_id].final_latent_path,
                            generation.examples[prompt_id].final_latent_path,
                        ),
                        "predictions": _prediction_metrics(
                            baseline.examples[prompt_id].prediction_latent_paths,
                            generation.examples[prompt_id].prediction_latent_paths,
                        ),
                    }
                    for prompt_id in common_prompt_ids
                ],
            }
            for generation in generations
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / COMPARISON_FILENAME
    tmp_path = output_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, allow_nan=False)
        fh.write("\n")
    tmp_path.replace(output_path)
    return output_path


def _read_generation_artifacts(generation_dir: Path) -> GenerationArtifacts:
    if not generation_dir.is_dir():
        raise ValueError(f"{generation_dir} is not a directory")

    metadata_paths = sorted(generation_dir.glob("*.json"))
    if not metadata_paths:
        raise ValueError(f"{generation_dir} does not contain sidecar JSON files")

    examples: dict[str, ExampleArtifacts] = {}
    config_names: set[str] = set()
    gpu_models: set[str | None] = set()
    for metadata_path in metadata_paths:
        metadata = _read_metadata(metadata_path)
        prompt_id = _metadata_string(metadata_path, metadata, "prompt_id")
        config_names.add(_metadata_string(metadata_path, metadata, "config_name"))
        environment = metadata.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError(f"{metadata_path} environment must be an object")
        gpu_model = environment.get("gpu_model")
        if gpu_model is not None and not isinstance(gpu_model, str):
            raise ValueError(
                f"{metadata_path} environment.gpu_model must be a string"
            )
        gpu_models.add(gpu_model)

        final_latent_path = generation_dir / f"{prompt_id}{FINAL_LATENT_SUFFIX}"
        video_path = generation_dir / f"{prompt_id}{VIDEO_SUFFIX}"
        if final_latent_path.exists() and video_path.exists():
            examples[prompt_id] = ExampleArtifacts(
                final_latent_path=final_latent_path,
                video_path=video_path,
                prediction_latent_paths=_prediction_latent_paths(
                    generation_dir,
                    prompt_id,
                ),
            )

    if len(config_names) != 1:
        raise ValueError(
            f"{generation_dir} contains multiple config_name values: "
            f"{sorted(config_names)}"
        )
    if len(gpu_models) != 1:
        raise ValueError(
            f"{generation_dir} contains multiple gpu_model values: "
            f"{sorted(gpu_models, key=lambda value: value or '')}"
        )
    if not examples:
        raise ValueError(
            f"{generation_dir} does not contain comparable final latent and video files"
        )

    return GenerationArtifacts(
        gpu_model=next(iter(gpu_models)),
        config_name=next(iter(config_names)),
        examples=examples,
    )


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    return metadata


def _metadata_string(
    metadata_path: Path,
    metadata: dict[str, Any],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{metadata_path} {key} must be a string")
    return value


def _latent_metrics(
    baseline_path: Path,
    generation_path: Path,
) -> dict[str, float]:
    import torch

    baseline = _load_latent_tensor(baseline_path)
    generation = _load_latent_tensor(generation_path)
    if baseline.shape != generation.shape:
        raise ValueError(
            "latent shapes differ for "
            f"{baseline_path.name} and {generation_path.name}: "
            f"{tuple(baseline.shape)} != {tuple(generation.shape)}"
        )

    diff = generation - baseline
    rmse = torch.sqrt(torch.mean(torch.square(diff))).item()
    baseline_norm = torch.linalg.vector_norm(baseline).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    if baseline_norm == 0.0:
        relative_l2_error = 0.0 if diff_norm == 0.0 else math.inf
    else:
        relative_l2_error = diff_norm / baseline_norm
    if not math.isfinite(relative_l2_error):
        raise ValueError(f"relative L2 is not finite for {generation_path.name}")
    return {
        "rmse": rmse,
        "relative_l2_error": relative_l2_error,
    }


def _prediction_metrics(
    baseline_paths: dict[int, Path],
    generation_paths: dict[int, Path],
) -> dict[str, Any]:
    common_steps = sorted(set(baseline_paths) & set(generation_paths))
    if not common_steps:
        raise ValueError("no matching prediction latent steps to compare")

    step_metrics = []
    for step in common_steps:
        metrics = _latent_metrics(baseline_paths[step], generation_paths[step])
        step_metrics.append({"step_idx": step, **metrics})
    rmses = [metrics["rmse"] for metrics in step_metrics]
    relative_l2_errors = [
        metrics["relative_l2_error"] for metrics in step_metrics
    ]
    return {
        "mean_rmse": sum(rmses) / len(rmses),
        "min_rmse": min(rmses),
        "max_rmse": max(rmses),
        "mean_relative_l2_error": (
            sum(relative_l2_errors) / len(relative_l2_errors)
        ),
        "min_relative_l2_error": min(relative_l2_errors),
        "max_relative_l2_error": max(relative_l2_errors),
        "steps": step_metrics,
    }


def _prediction_latent_paths(generation_dir: Path, prompt_id: str) -> dict[int, Path]:
    paths = {}
    for path in generation_dir.glob(f"{prompt_id}.denoising_step_*.safetensors"):
        match = PREDICTION_LATENT_PATTERN.search(path.name)
        if match is None:
            continue
        paths[int(match.group("step"))] = path
    return paths


def _load_latent_tensor(latent_path: Path) -> "torch.Tensor":
    from safetensors.torch import load_file

    tensors = load_file(latent_path, device="cpu")
    if "latents" in tensors:
        tensor = tensors["latents"]
    elif len(tensors) == 1:
        tensor = next(iter(tensors.values()))
    else:
        raise ValueError(
            f"{latent_path} must contain a 'latents' tensor or exactly one tensor"
        )
    if not tensor.is_floating_point():
        tensor = tensor.float()
    return tensor


def _video_file_metrics(
    baseline_path: Path,
    generation_path: Path,
) -> dict[str, float]:
    baseline_info = _video_info(baseline_path)
    generation_info = _video_info(generation_path)
    if (baseline_info.width, baseline_info.height) != (
        generation_info.width,
        generation_info.height,
    ):
        raise ValueError(
            "video dimensions differ for "
            f"{baseline_path.name} and {generation_path.name}: "
            f"{baseline_info.width}x{baseline_info.height} != "
            f"{generation_info.width}x{generation_info.height}"
        )
    if baseline_info.frame_count != generation_info.frame_count:
        raise ValueError(
            "video frame counts differ for "
            f"{baseline_path.name} and {generation_path.name}: "
            f"{baseline_info.frame_count} != {generation_info.frame_count}"
        )

    scores = _ffmpeg_ssim_scores(generation_path, baseline_path)
    if not scores:
        raise ValueError(f"{baseline_path} and {generation_path} contain no frames")
    if len(scores) != baseline_info.frame_count:
        raise ValueError(
            f"ffmpeg reported {len(scores)} SSIM scores for {generation_path.name}, "
            f"expected {baseline_info.frame_count}"
        )
    return {
        "mean_ssim": sum(scores) / len(scores),
        "max_ssim": max(scores),
        "min_ssim": min(scores),
    }


def _video_info(video_path: Path) -> VideoInfo:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe is required to compare video files")

    result = _run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,width,height",
            "-of",
            "json",
            str(video_path),
        ],
        context=f"probing {video_path}",
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"{video_path} does not contain a video stream")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValueError(f"{video_path} ffprobe stream output is invalid")
    return VideoInfo(
        width=_json_int(video_path, stream, "width"),
        height=_json_int(video_path, stream, "height"),
        frame_count=_json_int(video_path, stream, "nb_read_frames"),
    )


def _ffmpeg_ssim_scores(generation_path: Path, baseline_path: Path) -> list[float]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("ffmpeg is required to compare video files")

    with tempfile.TemporaryDirectory(prefix="viv-ssim-") as tmp_dir:
        stats_path = Path(tmp_dir) / "ssim.log"
        _run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-i",
                str(generation_path),
                "-i",
                str(baseline_path),
                "-lavfi",
                f"ssim=stats_file={stats_path}:eof_action=endall:repeatlast=0",
                "-f",
                "null",
                "-",
            ],
            context=f"computing SSIM for {generation_path} against {baseline_path}",
        )
        return _read_ssim_scores(stats_path)


def _read_ssim_scores(stats_path: Path) -> list[float]:
    scores = []
    with stats_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            match = _SSIM_SCORE_PATTERN.search(line)
            if match is None:
                continue
            scores.append(float(match.group("score")))
    return scores


def _json_int(path: Path, payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{path} ffprobe output is missing integer field {key}")


def _run_command(
    command: list[str],
    *,
    context: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"{context} failed: {message}")
    return result
