from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any

from viv.frames import wan_video_frames
from viv.latent_capture import (
    LATENT_PREFIX_EXTRA_ARG,
    PROMPT_EMBEDS_PREFIX_EXTRA_ARG,
    REUSE_PROMPT_EMBEDS_EXTRA_ARG,
    REUSE_FINAL_LATENT_EXTRA_ARG,
    REUSE_LATENT_PREFIX_EXTRA_ARG,
    REUSE_PREDICTIONS_EXTRA_ARG,
    SAVE_FINAL_LATENTS_EXTRA_ARG,
    SAVE_PREDICTION_LATENTS_EXTRA_ARG,
    SAVE_PROMPT_EMBEDS_EXTRA_ARG,
    SKIP_REUSED_COMPUTATION_EXTRA_ARG,
    SIGMA_SCHEDULE_PATH_EXTRA_ARG,
    final_noise_latent_path,
    initial_noise_latent_path,
    install_wan_latent_capture,
    load_latents,
    prompt_embeddings_path,
    prompt_embeddings_sha256_metadata,
    read_sigma_schedule,
    saved_denoising_prediction_count,
    safetensors_sha256_metadata,
    save_initial_noise_latents,
)
from viv.metadata import read_sidecar_generation_id
from viv.models import (
    GenerationResult,
    InferenceConfig,
    LatentReuseConfig,
    Prompt,
)

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
        save_initial_latents: bool = False,
        save_final_latents: bool = False,
        save_prediction_latents: bool = False,
        save_prompt_embeds: bool = False,
        latent_reuse: LatentReuseConfig | None = None,
    ) -> None:
        self.config = config
        self.save_initial_latents = save_initial_latents
        self.save_final_latents = save_final_latents
        self.save_prediction_latents = save_prediction_latents
        self.save_prompt_embeds = save_prompt_embeds
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
            or self.latent_reuse.reuse_all_predictions
        )
        reuse_final_latent = (
            self.latent_reuse is not None and self.latent_reuse.reuse_final_latent
        )
        reuse_prompt_embeds = (
            self.latent_reuse is not None
            and self.latent_reuse.reuse_prompt_embeds
        )
        skip_reused_computation = (
            self.latent_reuse is not None
            and self.latent_reuse.skip_reused_computation
        )
        initial_latent_reused = reuse_initial_latent
        reuse_predictions = (
            _resolve_reuse_prediction_count(self.latent_reuse, reuse_prefix)
            if self.latent_reuse is not None
            and (
                self.latent_reuse.reuse_predictions is not None
                or self.latent_reuse.reuse_all_predictions
            )
            else None
        )
        reused_prediction_latents = reuse_predictions or 0
        generated_initial_latents = None
        if not (reuse_initial_latent and skip_reused_computation):
            generated_initial_latents = _initial_noise_latents(self.config, seed)
        if reuse_initial_latent:
            if reuse_prefix is None:
                raise ValueError("missing latent reuse source prefix")
            latents = load_latents(initial_noise_latent_path(reuse_prefix))
        else:
            latents = generated_initial_latents
        reused_from = _reused_generation_id(
            reuse_prefix,
            reuse_initial_latent=reuse_initial_latent,
            reuse_final_latent=reuse_final_latent,
            reuse_predictions=reuse_predictions,
            reuse_prompt_embeds=reuse_prompt_embeds,
        )

        initial_latent_sha256 = None
        final_latent_path = None
        prompt_embeds_path = prompt_embeddings_path(video_path)
        prompt_embeds_sha256 = None
        negative_prompt_embeds_sha256 = None
        if latents is not None:
            if reuse_initial_latent and reuse_prefix is not None:
                initial_latent_path = initial_noise_latent_path(reuse_prefix)
                initial_latent_sha256 = safetensors_sha256_metadata(
                    initial_latent_path
                )
            if (
                self.save_initial_latents
                and not (reuse_initial_latent and skip_reused_computation)
            ):
                initial_latent_sha256 = save_initial_noise_latents(
                    initial_noise_latent_path(video_path),
                    generated_initial_latents,
                    seed,
                ) or initial_latent_sha256
            if (
                self.save_final_latents
                and not (reuse_final_latent and skip_reused_computation)
            ):
                final_latent_path = final_noise_latent_path(video_path)

        extra_args: dict[str, object] = {
            "flow_shift": self.config.flow_shift,
            SIGMA_SCHEDULE_PATH_EXTRA_ARG: str(sigma_schedule_path.resolve()),
        }
        if final_latent_path is not None or self.save_prediction_latents:
            extra_args[LATENT_PREFIX_EXTRA_ARG] = str(
                video_path.with_suffix("").resolve()
            )
        if final_latent_path is not None:
            extra_args[SAVE_FINAL_LATENTS_EXTRA_ARG] = True
        if self.save_prediction_latents:
            extra_args[SAVE_PREDICTION_LATENTS_EXTRA_ARG] = True
        if (
            self.save_prompt_embeds
            and not (reuse_prompt_embeds and skip_reused_computation)
        ):
            extra_args[PROMPT_EMBEDS_PREFIX_EXTRA_ARG] = str(
                video_path.with_suffix("").resolve()
            )
            extra_args[SAVE_PROMPT_EMBEDS_EXTRA_ARG] = True
        if reuse_prefix is not None:
            extra_args[REUSE_LATENT_PREFIX_EXTRA_ARG] = str(reuse_prefix.resolve())
        if reuse_prompt_embeds:
            extra_args[REUSE_PROMPT_EMBEDS_EXTRA_ARG] = True
        if (
            self.latent_reuse is not None
            and reuse_predictions is not None
        ):
            extra_args[REUSE_PREDICTIONS_EXTRA_ARG] = reuse_predictions
            if self.latent_reuse.skip_reused_computation:
                extra_args[SKIP_REUSED_COMPUTATION_EXTRA_ARG] = True
        if reuse_final_latent:
            extra_args[REUSE_FINAL_LATENT_EXTRA_ARG] = True
        if skip_reused_computation:
            extra_args[SKIP_REUSED_COMPUTATION_EXTRA_ARG] = True

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
        if final_latent_path is not None:
            if not final_latent_path.exists():
                raise FileNotFoundError(
                    "worker-side latent capture did not write expected final "
                    f"latent: {final_latent_path}"
                )
            final_latent_sha256 = safetensors_sha256_metadata(final_latent_path)
        elif reuse_final_latent:
            if reuse_prefix is None:
                raise ValueError("missing latent reuse source prefix")
            final_latent_sha256 = safetensors_sha256_metadata(
                final_noise_latent_path(reuse_prefix)
            )
        else:
            final_latent_sha256 = None
        prompt_embeds_saved = self.save_prompt_embeds and not (
            reuse_prompt_embeds and skip_reused_computation
        )
        if prompt_embeds_saved:
            if not prompt_embeds_path.exists():
                raise FileNotFoundError(
                    "worker-side prompt embedding capture did not write expected "
                    f"file: {prompt_embeds_path}"
                )
            (
                prompt_embeds_sha256,
                negative_prompt_embeds_sha256,
            ) = prompt_embeddings_sha256_metadata(prompt_embeds_path)
        elif reuse_prompt_embeds:
            if reuse_prefix is None:
                raise ValueError("missing prompt embedding reuse source prefix")
            (
                prompt_embeds_sha256,
                negative_prompt_embeds_sha256,
            ) = prompt_embeddings_sha256_metadata(prompt_embeddings_path(reuse_prefix))
        return GenerationResult(
            seed=seed,
            duration_seconds=time.perf_counter() - started_at,
            initial_noise_latent_sha256=initial_latent_sha256,
            final_noise_latent_sha256=final_latent_sha256,
            prompt_embeds_sha256=prompt_embeds_sha256,
            negative_prompt_embeds_sha256=negative_prompt_embeds_sha256,
            sigma_schedule=sigma_schedule,
            initial_noise_latent_reused=initial_latent_reused,
            final_noise_latent_reused=reuse_final_latent,
            prediction_latents_reused=reused_prediction_latents,
            prompt_embeds_reused=reuse_prompt_embeds,
            skipped_reused_computation=skip_reused_computation,
            reused_from=reused_from,
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


def _resolve_reuse_prediction_count(
    latent_reuse: LatentReuseConfig,
    reuse_prefix: Path | None,
) -> int:
    if not latent_reuse.reuse_all_predictions:
        if latent_reuse.reuse_predictions is None:
            raise ValueError("missing prediction latent reuse count")
        return latent_reuse.reuse_predictions
    if reuse_prefix is None:
        raise ValueError("missing latent reuse source prefix")
    return saved_denoising_prediction_count(reuse_prefix)


def _latent_reuse_prefix(
    latent_reuse: LatentReuseConfig | None, video_path: Path
) -> Path | None:
    if latent_reuse is None:
        return None
    return latent_reuse.source_dir / video_path.with_suffix("").name


def _reused_generation_id(
    reuse_prefix: Path | None,
    *,
    reuse_initial_latent: bool,
    reuse_final_latent: bool,
    reuse_predictions: int | None,
    reuse_prompt_embeds: bool,
) -> str | None:
    if reuse_prefix is None or not (
        reuse_initial_latent
        or reuse_final_latent
        or reuse_predictions is not None
        or reuse_prompt_embeds
    ):
        return None
    return read_sidecar_generation_id(reuse_prefix.with_suffix(".json"))


def _sigma_schedule_path(video_path: Path) -> Path:
    return video_path.with_name(f".{video_path.stem}.sigma_schedule.json")
