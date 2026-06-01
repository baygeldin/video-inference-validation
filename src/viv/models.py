from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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
class GenerationResult:
    seed: int
    duration_seconds: float


EnvironmentMetadata = Mapping[str, str | None]
