from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from viv.models import EnvironmentMetadata, GenerationResult, InferenceConfig, Prompt


def read_sidecar_generation_id(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    generation_id = metadata.get("generation_id")
    if generation_id is None:
        return None
    if not isinstance(generation_id, str):
        raise ValueError(f"{metadata_path} generation_id must be a string")
    return generation_id


def write_sidecar_metadata(
    metadata_path: Path,
    config_name: str,
    prompt: Prompt,
    config: InferenceConfig,
    result: GenerationResult,
    environment: EnvironmentMetadata,
) -> None:
    metadata = {
        "config_name": config_name,
        "generation_id": result.generation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": result.duration_seconds,
        "prompt_id": prompt.id,
        "prompt_text": prompt.prompt,
        "seed": result.seed,
        "initial_noise_latent_sha256": result.initial_noise_latent_sha256,
        "final_noise_latent_sha256": result.final_noise_latent_sha256,
        "sigma_schedule": result.sigma_schedule,
        "initial_noise_latent_reused": result.initial_noise_latent_reused,
        "final_noise_latent_reused": result.final_noise_latent_reused,
        "prediction_latents_reused": result.prediction_latents_reused,
        "reused_latents_from": result.reused_latents_from,
        "height": config.height,
        "width": config.width,
        "fps": config.fps,
        "num_frames": config.num_frames,
        "num_inference_steps": config.num_inference_steps,
        "boundary_ratio": config.boundary_ratio,
        "flow_shift": config.flow_shift,
        "guidance_scale": config.guidance_scale,
        "guidance_scale_2": config.guidance_scale_2,
        "model_name": config.model_name,
        "attention_backend": config.attention_backend,
        "model_revision": config.model_revision,
        "quality": config.export_quality,
        "tensor_parallelism": config.tensor_parallelism,
        "cache_backend": config.cache_backend,
        "environment": dict(environment),
    }
    tmp_path = metadata_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    tmp_path.replace(metadata_path)
