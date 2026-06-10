from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from viv.frames import wan_video_frames
from viv.latent_capture import (
    LATENT_PREFIX_EXTRA_ARG,
    final_noise_latent_path,
    initial_noise_latent_path,
    install_wan_latent_capture,
    latent_sha256,
    safetensors_sha256_metadata,
    save_initial_noise_latents,
)
from viv.models import GenerationResult, InferenceConfig, Prompt

WAN_LATENT_CHANNELS = 16
WAN_TEMPORAL_SCALE_FACTOR = 4
WAN_SPATIAL_SCALE_FACTOR = 8


def resolve_model_path(model: str, revision: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path)

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, revision=revision, allow_patterns=["*"])


class OfflineVideoGenerator:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        os.environ["DIFFUSION_ATTENTION_BACKEND"] = config.attention_backend
        model_path = resolve_model_path(config.model_name, config.model_revision)

        install_wan_latent_capture()

        from vllm_omni.diffusion.data import DiffusionParallelConfig
        from vllm_omni.entrypoints.omni import Omni

        parallel_config = DiffusionParallelConfig(
            tensor_parallel_size=self.config.tensor_parallelism
        )
        self.omni = Omni(
            model=model_path,
            revision=config.model_revision,
            attention_backend=config.attention_backend,
            parallel_config=parallel_config,
            cache_backend=config.cache_backend,
        )

    def generate(self, prompt: Prompt, video_path: Path) -> GenerationResult:
        from diffusers.utils import export_to_video
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        started_at = time.perf_counter()
        request: dict[str, object] = {"prompt": prompt.prompt}
        seed = secrets.randbits(63) if self.config.random_seed else prompt.seed
        if self.config.random_seed:
            print(f"using random seed {seed} for {prompt.id}", flush=True)

        latents = _initial_noise_latents(self.config, seed)
        initial_latent_sha256 = latent_sha256(latents)
        initial_latent_path = initial_noise_latent_path(video_path)
        save_initial_noise_latents(
            initial_latent_path, latents, seed, initial_latent_sha256
        )
        final_latent_path = final_noise_latent_path(video_path)

        sampling_params = OmniDiffusionSamplingParams(
            height=self.config.height,
            width=self.config.width,
            seed=seed,
            generator_device="cpu",
            latents=latents,
            boundary_ratio=self.config.boundary_ratio,
            extra_args={
                "flow_shift": self.config.flow_shift,
                LATENT_PREFIX_EXTRA_ARG: str(video_path.with_suffix("").resolve()),
            },
            guidance_scale=self.config.guidance_scale,
            guidance_scale_2=self.config.guidance_scale_2,
            num_inference_steps=self.config.num_inference_steps,
            num_frames=self.config.num_frames,
        )

        output = self.omni.generate(request, sampling_params)
        tmp_path = video_path.with_suffix(".tmp.mp4")
        export_to_video(
            wan_video_frames(output),
            str(tmp_path),
            fps=self.config.fps,
            quality=self.config.export_quality,
        )
        tmp_path.replace(video_path)
        final_latent_sha256 = safetensors_sha256_metadata(final_latent_path)
        return GenerationResult(
            seed=seed,
            duration_seconds=time.perf_counter() - started_at,
            initial_noise_latent_sha256=initial_latent_sha256,
            final_noise_latent_sha256=final_latent_sha256,
        )


def _initial_noise_latents(config: InferenceConfig, seed: int) -> Any:
    import torch
    from diffusers.utils.torch_utils import randn_tensor

    num_frames = config.num_frames
    if num_frames % WAN_TEMPORAL_SCALE_FACTOR != 1:
        num_frames = (
            num_frames // WAN_TEMPORAL_SCALE_FACTOR * WAN_TEMPORAL_SCALE_FACTOR + 1
        )

    shape = (
        1,
        WAN_LATENT_CHANNELS,
        (num_frames - 1) // WAN_TEMPORAL_SCALE_FACTOR + 1,
        config.height // WAN_SPATIAL_SCALE_FACTOR,
        config.width // WAN_SPATIAL_SCALE_FACTOR,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return randn_tensor(shape, generator=generator, device="cpu", dtype=torch.float32)
