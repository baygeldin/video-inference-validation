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
    final_latent_path: Path | None
    video_path: Path | None
    prediction_latent_paths: dict[int, Path]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int


@dataclass(frozen=True)
class ComparisonOptions:
    final_latents: bool = True
    video_files: bool = True
    predictions: bool = True

    def any_enabled(self) -> bool:
        return self.final_latents or self.video_files or self.predictions


def compare_generations(
    output_path: Path,
    baseline_dir: Path,
    generation_dirs: list[Path],
    *,
    final_latents: bool = True,
    video_files: bool = True,
    predictions: bool = True,
) -> Path:
    if not generation_dirs:
        raise ValueError("at least one generation path is required")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"{output_path} is a directory; expected a JSON file path")

    options = ComparisonOptions(
        final_latents=final_latents,
        video_files=video_files,
        predictions=predictions,
    )
    if not options.any_enabled():
        raise ValueError("at least one comparison type must be enabled")

    baseline = _read_generation_artifacts(baseline_dir, options)
    generations = [
        _read_generation_artifacts(path, options) for path in generation_dirs
    ]
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
                    _example_comparison(
                        prompt_id,
                        baseline.examples[prompt_id],
                        generation.examples[prompt_id],
                        options,
                    )
                    for prompt_id in common_prompt_ids
                ],
            }
            for generation in generations
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, allow_nan=False)
        fh.write("\n")
    tmp_path.replace(output_path)
    return output_path


def _read_generation_artifacts(
    generation_dir: Path,
    options: ComparisonOptions,
) -> GenerationArtifacts:
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
        prediction_latent_paths = _prediction_latent_paths(
            generation_dir,
            prompt_id,
        )
        if _has_requested_artifacts(
            final_latent_path,
            video_path,
            prediction_latent_paths,
            options,
        ):
            examples[prompt_id] = ExampleArtifacts(
                final_latent_path=(
                    final_latent_path if final_latent_path.exists() else None
                ),
                video_path=video_path if video_path.exists() else None,
                prediction_latent_paths=prediction_latent_paths,
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
            f"{generation_dir} does not contain comparable artifacts for the "
            "selected comparison types"
        )

    return GenerationArtifacts(
        gpu_model=next(iter(gpu_models)),
        config_name=next(iter(config_names)),
        examples=examples,
    )


def _has_requested_artifacts(
    final_latent_path: Path,
    video_path: Path,
    prediction_latent_paths: dict[int, Path],
    options: ComparisonOptions,
) -> bool:
    if options.final_latents and not final_latent_path.exists():
        return False
    if options.video_files and not video_path.exists():
        return False
    if options.predictions and not prediction_latent_paths:
        return False
    return True


def _example_comparison(
    prompt_id: str,
    baseline: ExampleArtifacts,
    generation: ExampleArtifacts,
    options: ComparisonOptions,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"prompt_id": prompt_id}
    if options.video_files:
        if baseline.video_path is None or generation.video_path is None:
            raise ValueError(f"missing video file for prompt {prompt_id}")
        metrics["video_file"] = _video_file_metrics(
            baseline.video_path,
            generation.video_path,
        )
    if options.final_latents:
        if (
            baseline.final_latent_path is None
            or generation.final_latent_path is None
        ):
            raise ValueError(f"missing final latent for prompt {prompt_id}")
        metrics["final_latent"] = _latent_metrics(
            baseline.final_latent_path,
            generation.final_latent_path,
        )
    if options.predictions:
        metrics["predictions"] = _prediction_metrics(
            baseline.prediction_latent_paths,
            generation.prediction_latent_paths,
        )
    return metrics


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
) -> dict[str, Any]:
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
    abs_diff = torch.abs(diff)
    rmse = torch.sqrt(torch.mean(torch.square(diff))).item()
    baseline_norm = torch.linalg.vector_norm(baseline).item()
    generation_norm = torch.linalg.vector_norm(generation).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    baseline_l1_norm = torch.sum(torch.abs(baseline)).item()
    diff_l1_norm = torch.sum(abs_diff).item()
    baseline_std = torch.std(baseline).item()
    if baseline_norm == 0.0:
        relative_l2_error = 0.0 if diff_norm == 0.0 else math.inf
    else:
        relative_l2_error = diff_norm / baseline_norm
    if not math.isfinite(relative_l2_error):
        raise ValueError(f"relative L2 is not finite for {generation_path.name}")
    if baseline_l1_norm == 0.0:
        relative_l1_error = 0.0 if diff_l1_norm == 0.0 else math.inf
    else:
        relative_l1_error = diff_l1_norm / baseline_l1_norm
    if not math.isfinite(relative_l1_error):
        raise ValueError(f"relative L1 is not finite for {generation_path.name}")
    if baseline_norm == 0.0 or generation_norm == 0.0:
        cosine_similarity = 1.0 if diff_norm == 0.0 else math.nan
    else:
        cosine_similarity = (
            torch.sum(baseline * generation).item()
            / (baseline_norm * generation_norm)
        )
        cosine_similarity = max(-1.0, min(1.0, cosine_similarity))
    if not math.isfinite(cosine_similarity):
        raise ValueError(
            f"cosine similarity is not finite for {generation_path.name}"
        )
    nrmse = _normalized_rmse(rmse, baseline_std, generation_path)
    sqnr_db = _sqnr_db(baseline, diff)
    pearson_correlation = _pearson_correlation(baseline, generation)
    per_channel = _per_channel_metrics(baseline, generation, diff)
    return {
        "rmse": rmse,
        "relative_l1_error": relative_l1_error,
        "relative_l2_error": relative_l2_error,
        "cosine_similarity": cosine_similarity,
        "sqnr_db": sqnr_db,
        "pearson_correlation": pearson_correlation,
        "nrmse": nrmse,
        **per_channel,
    }


def _normalized_rmse(
    rmse: float,
    baseline_std: float,
    generation_path: Path,
) -> float:
    if baseline_std == 0.0:
        nrmse = 0.0 if rmse == 0.0 else math.inf
    else:
        nrmse = rmse / baseline_std
    if not math.isfinite(nrmse):
        raise ValueError(f"NRMSE is not finite for {generation_path.name}")
    return nrmse


def _sqnr_db(baseline: "torch.Tensor", diff: "torch.Tensor") -> float | None:
    import torch

    signal_power = torch.sum(torch.square(baseline)).item()
    noise_power = torch.sum(torch.square(diff)).item()
    if signal_power <= 0.0 or noise_power <= 0.0:
        return None
    return 10.0 * math.log10(signal_power / noise_power)


def _pearson_correlation(
    baseline: "torch.Tensor",
    generation: "torch.Tensor",
) -> float:
    import torch

    baseline_centered = baseline - torch.mean(baseline)
    generation_centered = generation - torch.mean(generation)
    baseline_norm = torch.linalg.vector_norm(baseline_centered).item()
    generation_norm = torch.linalg.vector_norm(generation_centered).item()
    if baseline_norm == 0.0 or generation_norm == 0.0:
        raw_diff_norm = torch.linalg.vector_norm(generation - baseline).item()
        if raw_diff_norm == 0.0:
            return 1.0
        raise ValueError("Pearson correlation is undefined for a constant tensor")
    correlation = (
        torch.sum(baseline_centered * generation_centered).item()
        / (baseline_norm * generation_norm)
    )
    correlation = max(-1.0, min(1.0, correlation))
    if not math.isfinite(correlation):
        raise ValueError("Pearson correlation is not finite")
    return correlation


def _per_channel_metrics(
    baseline: "torch.Tensor",
    generation: "torch.Tensor",
    diff: "torch.Tensor",
) -> dict[str, Any]:
    import torch

    if baseline.ndim < 2:
        return {}

    baseline_by_channel = baseline.transpose(0, 1).reshape(baseline.shape[1], -1)
    generation_by_channel = generation.transpose(0, 1).reshape(
        generation.shape[1], -1
    )
    diff_by_channel = diff.transpose(0, 1).reshape(diff.shape[1], -1)

    baseline_norms = torch.linalg.vector_norm(baseline_by_channel, dim=1)
    generation_norms = torch.linalg.vector_norm(generation_by_channel, dim=1)
    diff_norms = torch.linalg.vector_norm(diff_by_channel, dim=1)
    relative_l2_errors = _safe_divide_zero_equal(diff_norms, baseline_norms)

    dot_products = torch.sum(baseline_by_channel * generation_by_channel, dim=1)
    cosine_similarities = []
    for idx in range(baseline_by_channel.shape[0]):
        baseline_norm = baseline_norms[idx].item()
        generation_norm = generation_norms[idx].item()
        diff_norm = diff_norms[idx].item()
        if baseline_norm == 0.0 or generation_norm == 0.0:
            value = 1.0 if diff_norm == 0.0 else math.nan
        else:
            value = dot_products[idx].item() / (baseline_norm * generation_norm)
            value = max(-1.0, min(1.0, value))
        if not math.isfinite(value):
            raise ValueError("per-channel cosine similarity is not finite")
        cosine_similarities.append(value)

    rmses = torch.sqrt(torch.mean(torch.square(diff_by_channel), dim=1))
    baseline_stds = torch.std(baseline_by_channel, dim=1)
    nrmse_values = _safe_divide_zero_equal(rmses, baseline_stds)

    sqnr_values = [
        _sqnr_db(baseline_by_channel[idx], diff_by_channel[idx])
        for idx in range(baseline_by_channel.shape[0])
    ]
    pearson_values = [
        _pearson_correlation(
            baseline_by_channel[idx],
            generation_by_channel[idx],
        )
        for idx in range(baseline_by_channel.shape[0])
    ]

    relative_l2 = _tensor_to_float_list(relative_l2_errors)
    nrmse = _tensor_to_float_list(nrmse_values)
    return {
        "per_channel_relative_l2_error": relative_l2,
        "mean_per_channel_relative_l2_error": sum(relative_l2) / len(relative_l2),
        "rms_per_channel_relative_l2_error": _rms(relative_l2),
        "max_per_channel_relative_l2_error": max(relative_l2),
        "per_channel_cosine_similarity": cosine_similarities,
        "mean_per_channel_cosine_similarity": (
            sum(cosine_similarities) / len(cosine_similarities)
        ),
        "min_per_channel_cosine_similarity": min(cosine_similarities),
        "per_channel_sqnr_db": sqnr_values,
        "mean_per_channel_sqnr_db": _mean_optional(sqnr_values),
        "min_per_channel_sqnr_db": _min_optional(sqnr_values),
        "per_channel_pearson_correlation": pearson_values,
        "mean_per_channel_pearson_correlation": (
            sum(pearson_values) / len(pearson_values)
        ),
        "min_per_channel_pearson_correlation": min(pearson_values),
        "per_channel_nrmse": nrmse,
        "mean_per_channel_nrmse": sum(nrmse) / len(nrmse),
        "rms_per_channel_nrmse": _rms(nrmse),
        "max_per_channel_nrmse": max(nrmse),
    }


def _safe_divide_zero_equal(
    numerator: "torch.Tensor",
    denominator: "torch.Tensor",
) -> "torch.Tensor":
    import torch

    return torch.where(
        denominator == 0.0,
        torch.where(numerator == 0.0, torch.zeros_like(numerator), torch.inf),
        numerator / denominator,
    )


def _tensor_to_float_list(values: "torch.Tensor") -> list[float]:
    output = [float(value) for value in values.tolist()]
    if not all(math.isfinite(value) for value in output):
        raise ValueError("per-channel metric contains non-finite values")
    return output


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mean_optional(values: list[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def _min_optional(values: list[float | None]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return min(finite_values)


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
    relative_l1_errors = [
        metrics["relative_l1_error"] for metrics in step_metrics
    ]
    relative_l2_errors = [
        metrics["relative_l2_error"] for metrics in step_metrics
    ]
    cosine_similarities = [
        metrics["cosine_similarity"] for metrics in step_metrics
    ]
    sqnr_dbs = [metrics["sqnr_db"] for metrics in step_metrics]
    pearson_correlations = [
        metrics["pearson_correlation"] for metrics in step_metrics
    ]
    nrmse_values = [metrics["nrmse"] for metrics in step_metrics]
    mean_per_channel_relative_l2_errors = _metric_values(
        step_metrics, "mean_per_channel_relative_l2_error"
    )
    rms_per_channel_relative_l2_errors = _metric_values(
        step_metrics, "rms_per_channel_relative_l2_error"
    )
    mean_per_channel_cosine_similarities = _metric_values(
        step_metrics, "mean_per_channel_cosine_similarity"
    )
    mean_per_channel_sqnr_dbs = _optional_metric_values(
        step_metrics, "mean_per_channel_sqnr_db"
    )
    mean_per_channel_pearson_correlations = _metric_values(
        step_metrics, "mean_per_channel_pearson_correlation"
    )
    mean_per_channel_nrmse_values = _metric_values(
        step_metrics, "mean_per_channel_nrmse"
    )
    return {
        "mean_rmse": sum(rmses) / len(rmses),
        "min_rmse": min(rmses),
        "max_rmse": max(rmses),
        "mean_relative_l1_error": (
            sum(relative_l1_errors) / len(relative_l1_errors)
        ),
        "min_relative_l1_error": min(relative_l1_errors),
        "max_relative_l1_error": max(relative_l1_errors),
        "mean_relative_l2_error": (
            sum(relative_l2_errors) / len(relative_l2_errors)
        ),
        "min_relative_l2_error": min(relative_l2_errors),
        "max_relative_l2_error": max(relative_l2_errors),
        "mean_cosine_similarity": (
            sum(cosine_similarities) / len(cosine_similarities)
        ),
        "min_cosine_similarity": min(cosine_similarities),
        "max_cosine_similarity": max(cosine_similarities),
        "mean_sqnr_db": _mean_optional(sqnr_dbs),
        "min_sqnr_db": _min_optional(sqnr_dbs),
        "mean_pearson_correlation": (
            sum(pearson_correlations) / len(pearson_correlations)
        ),
        "min_pearson_correlation": min(pearson_correlations),
        "max_pearson_correlation": max(pearson_correlations),
        "mean_nrmse": sum(nrmse_values) / len(nrmse_values),
        "min_nrmse": min(nrmse_values),
        "max_nrmse": max(nrmse_values),
        "mean_per_channel_relative_l2_error": (
            sum(mean_per_channel_relative_l2_errors)
            / len(mean_per_channel_relative_l2_errors)
        ),
        "mean_rms_per_channel_relative_l2_error": (
            sum(rms_per_channel_relative_l2_errors)
            / len(rms_per_channel_relative_l2_errors)
        ),
        "mean_per_channel_cosine_similarity": (
            sum(mean_per_channel_cosine_similarities)
            / len(mean_per_channel_cosine_similarities)
        ),
        "mean_per_channel_sqnr_db": _mean_optional(
            mean_per_channel_sqnr_dbs
        ),
        "mean_per_channel_pearson_correlation": (
            sum(mean_per_channel_pearson_correlations)
            / len(mean_per_channel_pearson_correlations)
        ),
        "mean_per_channel_nrmse": (
            sum(mean_per_channel_nrmse_values)
            / len(mean_per_channel_nrmse_values)
        ),
        "steps": step_metrics,
    }


def _metric_values(
    step_metrics: list[dict[str, Any]],
    key: str,
) -> list[float]:
    values = [metrics[key] for metrics in step_metrics if key in metrics]
    return [float(value) for value in values]


def _optional_metric_values(
    step_metrics: list[dict[str, Any]],
    key: str,
) -> list[float | None]:
    return [metrics[key] for metrics in step_metrics if key in metrics]


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
    return tensor.float()


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
