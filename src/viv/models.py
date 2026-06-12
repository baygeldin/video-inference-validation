from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


def _new_generation_id() -> str:
    return secrets.token_urlsafe(6)


@dataclass(frozen=True)
class Prompt:
    id: str
    prompt: str
    seed: int


@dataclass(frozen=True)
class InferenceConfig:
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    boundary_ratio: float
    flow_shift: float
    guidance_scale: float
    guidance_scale_2: float
    model_name: str
    model_revision: str
    attention_backend: str
    export_quality: float
    tensor_parallelism: int
    cache_backend: str | None
    random_seed: bool


@dataclass(frozen=True)
class LatentReuseConfig:
    source_dir: Path
    reuse_initial_latent: bool = False
    reuse_predictions: int | None = None
    reuse_final_latent: bool = False


@dataclass(frozen=True)
class GenerationResult:
    seed: int
    duration_seconds: float
    initial_noise_latent_sha256: str | None
    final_noise_latent_sha256: str | None
    sigma_schedule: list[float] | None = None
    initial_noise_latent_reused: bool = False
    final_noise_latent_reused: bool = False
    prediction_latents_reused: int = 0
    generation_id: str = field(default_factory=_new_generation_id)
    reused_latents_from: str | None = None


EnvironmentMetadata = Mapping[str, str | None]
