from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


COMPARISON_FILENAME = "comparison.json"
FINAL_LATENT_SUFFIX = ".final_noise_latent.safetensors"


@dataclass(frozen=True)
class GenerationArtifacts:
    gpu_model: str | None
    config_name: str
    examples: dict[str, Path]


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
                        "final_latent": _final_latent_metrics(
                            baseline.examples[prompt_id],
                            generation.examples[prompt_id],
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

    examples: dict[str, Path] = {}
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

        latent_path = generation_dir / f"{prompt_id}{FINAL_LATENT_SUFFIX}"
        if latent_path.exists():
            examples[prompt_id] = latent_path

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
        raise ValueError(f"{generation_dir} does not contain final latent tensors")

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


def _final_latent_metrics(
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
        relative_l2 = 0.0 if diff_norm == 0.0 else math.inf
    else:
        relative_l2 = diff_norm / baseline_norm
    if not math.isfinite(relative_l2):
        raise ValueError(f"relative L2 is not finite for {generation_path.name}")
    return {
        "rmse": rmse,
        "relative_l2": relative_l2,
    }


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
