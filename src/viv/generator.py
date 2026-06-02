from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from viv.frames import wan_video_frames
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
        latent_sha256 = _latent_sha256(latents)
        latent_path = _initial_noise_latent_path(video_path)
        _save_initial_noise_latents(latent_path, latents, seed, latent_sha256)

        sampling_params = OmniDiffusionSamplingParams(
            height=self.config.height,
            width=self.config.width,
            seed=seed,
            generator_device="cpu",
            latents=latents,
            boundary_ratio=self.config.boundary_ratio,
            extra_args={"flow_shift": self.config.flow_shift},
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
        return GenerationResult(
            seed=seed,
            duration_seconds=time.perf_counter() - started_at,
            initial_noise_latent_sha256=latent_sha256,
        )


def _initial_noise_latent_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.initial_noise_latent.safetensors")


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


def _latent_sha256(tensor: Any) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")

    metadata = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(metadata_bytes)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _save_initial_noise_latents(
    latent_path: Path, latents: Any, seed: int, latent_sha256: str
) -> None:
    from safetensors.torch import save_file

    tmp_path = latent_path.with_name(f"{latent_path.stem}.tmp{latent_path.suffix}")
    save_file(
        {"latents": latents.detach().cpu().contiguous()},
        str(tmp_path),
        metadata={
            "seed": str(seed),
            "sha256": latent_sha256,
        },
    )
    tmp_path.replace(latent_path)
