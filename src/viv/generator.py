from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from viv.frames import wan_video_frames
from viv.latent_capture import (
    LATENT_PREFIX_EXTRA_ARG,
    REUSE_FINAL_LATENT_EXTRA_ARG,
    REUSE_LATENT_PREFIX_EXTRA_ARG,
    REUSE_PREDICTIONS_EXTRA_ARG,
    SIGMA_SCHEDULE_PATH_EXTRA_ARG,
    final_noise_latent_path,
    initial_noise_latent_path,
    install_wan_latent_capture,
    load_latents,
    read_sigma_schedule,
    safetensors_sha256_metadata,
    save_initial_noise_latents,
)
from viv.metadata import read_sidecar_generation_id
from viv.models import GenerationResult, InferenceConfig, LatentReuseConfig, Prompt

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
    def __init__(
        self,
        config: InferenceConfig,
        save_latents: bool = False,
        latent_reuse: LatentReuseConfig | None = None,
    ) -> None:
        self.config = config
        self.save_latents = save_latents
        self.latent_reuse = latent_reuse
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

        sigma_schedule_path = _sigma_schedule_path(video_path)
        sigma_schedule_path.unlink(missing_ok=True)

        reuse_prefix = _latent_reuse_prefix(self.latent_reuse, video_path)
        reuse_initial_latent = self.latent_reuse is not None and (
            self.latent_reuse.reuse_initial_latent
            or self.latent_reuse.reuse_predictions is not None
        )
        reuse_final_latent = (
            self.latent_reuse is not None and self.latent_reuse.reuse_final_latent
        )
        initial_latent_reused = reuse_initial_latent and not reuse_final_latent
        reused_prediction_latents = (
            self.latent_reuse.reuse_predictions
            if not reuse_final_latent
            and self.latent_reuse is not None
            and self.latent_reuse.reuse_predictions is not None
            else 0
        )
        if reuse_final_latent:
            latents = None
        elif reuse_initial_latent and reuse_prefix is not None:
            latents = load_latents(initial_noise_latent_path(reuse_prefix))
        else:
            latents = _initial_noise_latents(self.config, seed)
        reused_latents_from = _reused_latents_generation_id(
            reuse_prefix,
            reuse_initial_latent=reuse_initial_latent,
            reuse_final_latent=reuse_final_latent,
        )

        initial_latent_sha256 = None
        final_latent_path = None
        if latents is not None:
            if reuse_initial_latent and reuse_prefix is not None:
                initial_latent_path = initial_noise_latent_path(reuse_prefix)
                initial_latent_sha256 = safetensors_sha256_metadata(
                    initial_latent_path
                )
            if self.save_latents:
                initial_latent_sha256 = save_initial_noise_latents(
                    initial_noise_latent_path(video_path),
                    latents,
                    seed,
                ) or initial_latent_sha256
                final_latent_path = final_noise_latent_path(video_path)

        extra_args: dict[str, object] = {
            "flow_shift": self.config.flow_shift,
            SIGMA_SCHEDULE_PATH_EXTRA_ARG: str(sigma_schedule_path.resolve()),
        }
        if self.save_latents:
            extra_args[LATENT_PREFIX_EXTRA_ARG] = str(
                video_path.with_suffix("").resolve()
            )
        if reuse_prefix is not None:
            extra_args[REUSE_LATENT_PREFIX_EXTRA_ARG] = str(reuse_prefix.resolve())
        if (
            self.latent_reuse is not None
            and self.latent_reuse.reuse_predictions is not None
        ):
            extra_args[REUSE_PREDICTIONS_EXTRA_ARG] = (
                self.latent_reuse.reuse_predictions
            )
        if reuse_final_latent:
            extra_args[REUSE_FINAL_LATENT_EXTRA_ARG] = True

        sampling_params = OmniDiffusionSamplingParams(
            height=self.config.height,
            width=self.config.width,
            seed=seed,
            generator_device="cpu",
            latents=latents,
            boundary_ratio=self.config.boundary_ratio,
            extra_args=extra_args,
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
        try:
            sigma_schedule = read_sigma_schedule(sigma_schedule_path)
        finally:
            sigma_schedule_path.unlink(missing_ok=True)
        if reuse_final_latent:
            if reuse_prefix is None:
                raise ValueError("missing latent reuse source prefix")
            final_latent_sha256 = safetensors_sha256_metadata(
                final_noise_latent_path(reuse_prefix)
            )
        elif final_latent_path is not None:
            final_latent_sha256 = safetensors_sha256_metadata(final_latent_path)
        else:
            final_latent_sha256 = None
        return GenerationResult(
            seed=seed,
            duration_seconds=time.perf_counter() - started_at,
            initial_noise_latent_sha256=initial_latent_sha256,
            final_noise_latent_sha256=final_latent_sha256,
            sigma_schedule=sigma_schedule,
            initial_noise_latent_reused=initial_latent_reused,
            final_noise_latent_reused=reuse_final_latent,
            prediction_latents_reused=reused_prediction_latents,
            reused_latents_from=reused_latents_from,
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


def _latent_reuse_prefix(
    latent_reuse: LatentReuseConfig | None, video_path: Path
) -> Path | None:
    if latent_reuse is None:
        return None
    return latent_reuse.source_dir / video_path.with_suffix("").name


def _reused_latents_generation_id(
    reuse_prefix: Path | None,
    *,
    reuse_initial_latent: bool,
    reuse_final_latent: bool,
) -> str | None:
    if reuse_prefix is None or not (reuse_initial_latent or reuse_final_latent):
        return None
    return read_sidecar_generation_id(reuse_prefix.with_suffix(".json"))


def _sigma_schedule_path(video_path: Path) -> Path:
    return video_path.with_name(f".{video_path.stem}.sigma_schedule.json")
