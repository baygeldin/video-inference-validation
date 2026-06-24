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
        "generation_id": result.generation_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": result.duration_seconds,
        "config_name": config_name,
        "prompt_id": prompt.id,
        "prompt_text": prompt.prompt,
        "seed": result.seed,
        "reuse": {
            "reused_from": result.reused_from,
            "prompt_embeds_reused": result.prompt_embeds_reused,
            "initial_noise_latent_reused": result.initial_noise_latent_reused,
            "prediction_latents_reused": result.prediction_latents_reused,
            "final_noise_latent_reused": result.final_noise_latent_reused,
            "skipped_reused_computation": result.skipped_reused_computation,
        },
        "parameters": {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "attention_backend": config.attention_backend,
            "cache_backend": config.cache_backend,
            "tensor_parallelism": config.tensor_parallelism,
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
            "quality": config.export_quality,
            "boundary_ratio": config.boundary_ratio,
            "flow_shift": config.flow_shift,
            "guidance_scale": config.guidance_scale,
            "guidance_scale_2": config.guidance_scale_2,
            "sigma_schedule": result.sigma_schedule,
        },
        "environment": {
            "gpu_model": environment.get("gpu_model"),
            "cuda_version": environment.get("cuda_version"),
            "nvidia_driver_version": environment.get("nvidia_driver_version"),
            "vllm_version": environment.get("vllm_version"),
            "vllm_omni_version": environment.get("vllm_omni_version"),
            "vllm_omni_commit": environment.get("vllm_omni_commit"),
            "pytorch_version": environment.get("pytorch_version"),
            "ffmpeg_version": environment.get("ffmpeg_version"),
            "python_version": environment.get("python_version"),
            "image_name": environment.get("image_name"),
        },
        "checksums": {
            "prompt_embeds_sha256": result.prompt_embeds_sha256,
            "negative_prompt_embeds_sha256": result.negative_prompt_embeds_sha256,
            "initial_noise_latent_sha256": result.initial_noise_latent_sha256,
            "final_noise_latent_sha256": result.final_noise_latent_sha256,
        },
    }
    tmp_path = metadata_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    tmp_path.replace(metadata_path)
