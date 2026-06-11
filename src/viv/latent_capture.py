from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


LATENT_PREFIX_EXTRA_ARG = "viv_latent_prefix"
REUSE_LATENT_PREFIX_EXTRA_ARG = "viv_reuse_latent_prefix"
REUSE_DENOISING_SIGMA_THRESHOLD_EXTRA_ARG = "viv_reuse_denoising_sigma_threshold"
REUSE_FINAL_LATENT_EXTRA_ARG = "viv_reuse_final_latent"


def install_wan_latent_capture() -> None:
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

    if getattr(Wan22Pipeline, "_viv_latent_capture_patched", False):
        return

    original_forward = Wan22Pipeline.forward
    original_diffuse = Wan22Pipeline.diffuse
    original_scheduler_step = Wan22Pipeline.scheduler_step_maybe_with_cfg

    def forward(self: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
        previous_prefix = getattr(self, "_viv_latent_prefix", None)
        previous_seed = getattr(self, "_viv_seed", None)
        previous_reuse_prefix = getattr(self, "_viv_reuse_latent_prefix", None)
        previous_reuse_sigma_threshold = getattr(
            self, "_viv_reuse_denoising_sigma_threshold", None
        )
        self._viv_latent_prefix = _prefix_from_request(req)
        self._viv_seed = _seed_from_request(req)
        self._viv_reuse_latent_prefix = _reuse_prefix_from_request(req)
        self._viv_reuse_denoising_sigma_threshold = (
            _reuse_denoising_sigma_threshold_from_request(req)
        )
        try:
            if _reuse_final_latent_from_request(req):
                return _decode_reused_final_latent(
                    self, req, _output_type_from_call(req, args, kwargs)
                )
            return original_forward(self, req, *args, **kwargs)
        finally:
            self._viv_latent_prefix = previous_prefix
            self._viv_seed = previous_seed
            self._viv_reuse_latent_prefix = previous_reuse_prefix
            self._viv_reuse_denoising_sigma_threshold = (
                previous_reuse_sigma_threshold
            )

    def diffuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_step_idx = getattr(self, "_viv_step_idx", None)
        reuse_sigma_threshold = getattr(
            self, "_viv_reuse_denoising_sigma_threshold", None
        )
        self._viv_step_idx = 0
        try:
            if reuse_sigma_threshold is None:
                latents = original_diffuse(self, *args, **kwargs)
            else:
                latents = _diffuse_with_reused_denoising(
                    self,
                    original_diffuse,
                    reuse_sigma_threshold,
                    *args,
                    **kwargs,
                )
            final_latent_path = _final_noise_latent_path_from_prefix(
                getattr(self, "_viv_latent_prefix", None)
            )
            if final_latent_path is not None:
                _save_latents(
                    final_latent_path,
                    "final_noise",
                    latents,
                    {
                        "seed": getattr(self, "_viv_seed", None),
                        "sha256": latent_sha256(latents),
                    },
                )
            return latents
        finally:
            if reuse_sigma_threshold is not None and hasattr(self, "_sync_pp_send"):
                self._sync_pp_send()
            self._viv_step_idx = previous_step_idx

    def scheduler_step_maybe_with_cfg(
        self: Any,
        noise_pred: Any,
        t: Any,
        latents: Any,
        do_true_cfg: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        updated_latents = original_scheduler_step(
            self, noise_pred, t, latents, do_true_cfg, *args, **kwargs
        )
        step_idx = getattr(self, "_viv_step_idx", 0)
        _save_latents(
            _denoising_step_latent_path_from_prefix(
                getattr(self, "_viv_latent_prefix", None), step_idx
            ),
            "denoising_step",
            updated_latents,
            {
                "seed": getattr(self, "_viv_seed", None),
                "step_idx": step_idx,
                "timestep": _scalar_float(t),
                "sigma": _sigma_for_step(self.scheduler, step_idx, t),
            },
        )
        self._viv_step_idx = step_idx + 1
        return updated_latents

    Wan22Pipeline.forward = forward
    Wan22Pipeline.diffuse = diffuse
    Wan22Pipeline.scheduler_step_maybe_with_cfg = scheduler_step_maybe_with_cfg
    Wan22Pipeline._viv_latent_capture_patched = True


def initial_noise_latent_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.initial_noise_latent.safetensors")


def final_noise_latent_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.final_noise_latent.safetensors")


def denoising_step_latent_path(video_path: Path, step_idx: int) -> Path:
    return video_path.with_name(f"{video_path.stem}.denoising_step_{step_idx}.safetensors")


def denoising_step_latent_paths(video_path: Path, num_inference_steps: int) -> list[Path]:
    return [
        denoising_step_latent_path(video_path, step_idx)
        for step_idx in range(num_inference_steps)
    ]


def load_latents(
    latent_path: Path, device: Any | None = None, dtype: Any | None = None
) -> Any:
    from safetensors.torch import load_file

    tensors = load_file(str(latent_path), device="cpu")
    try:
        latents = tensors["latents"]
    except KeyError as exc:
        raise ValueError(f"{latent_path} does not contain a latents tensor") from exc
    if device is not None or dtype is not None:
        latents = latents.to(device=device, dtype=dtype)
    return latents


def latent_sha256(tensor: Any) -> str:
    import torch

    tensor = _resolve_latents(tensor)
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


def save_initial_noise_latents(
    latent_path: Path, latents: Any, seed: int, latent_sha256: str
) -> None:
    _save_latents(
        latent_path,
        "initial_noise",
        latents,
        {
            "seed": seed,
            "sha256": latent_sha256,
        },
    )


def safetensors_sha256_metadata(latent_path: Path) -> str:
    from safetensors import safe_open

    with safe_open(str(latent_path), framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata() or {}
    sha256 = metadata.get("sha256")
    if not sha256:
        raise ValueError(f"{latent_path} does not contain sha256 metadata")
    return sha256


def _save_latents(
    latent_path: Path | None, kind: str, latents: Any, metadata: dict[str, Any]
) -> None:
    if latent_path is None or latents is None or not _is_rank_zero():
        return

    import torch
    from safetensors.torch import save_file

    latents = _resolve_latents(latents)
    if not isinstance(latents, torch.Tensor):
        return

    latent_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = latent_path.with_name(f"{latent_path.stem}.tmp{latent_path.suffix}")
    stored_metadata = {
        "kind": kind,
        **{
            key: str(value)
            for key, value in metadata.items()
            if value is not None
        },
    }
    save_file(
        {"latents": latents.detach().to("cpu").contiguous()},
        str(tmp_path),
        metadata=stored_metadata,
    )
    tmp_path.replace(latent_path)


def _prefix_from_request(req: Any) -> Path | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    prefix = extra_args.get(LATENT_PREFIX_EXTRA_ARG)
    if prefix is None:
        return None
    return Path(prefix)


def _reuse_prefix_from_request(req: Any) -> Path | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    prefix = extra_args.get(REUSE_LATENT_PREFIX_EXTRA_ARG)
    if prefix is None:
        return None
    return Path(prefix)


def _reuse_denoising_sigma_threshold_from_request(req: Any) -> float | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    threshold = extra_args.get(REUSE_DENOISING_SIGMA_THRESHOLD_EXTRA_ARG)
    if threshold is None:
        return None
    return float(threshold)


def _reuse_final_latent_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(REUSE_FINAL_LATENT_EXTRA_ARG))


def _seed_from_request(req: Any) -> int | None:
    seed = getattr(getattr(req, "sampling_params", None), "seed", None)
    return int(seed) if seed is not None else None


def _final_noise_latent_path_from_prefix(prefix: Path | None) -> Path | None:
    if prefix is None:
        return None
    return prefix.with_name(f"{prefix.name}.final_noise_latent.safetensors")


def _denoising_step_latent_path_from_prefix(
    prefix: Path | None, step_idx: int
) -> Path | None:
    if prefix is None:
        return None
    return prefix.with_name(f"{prefix.name}.denoising_step_{step_idx}.safetensors")


def _diffuse_with_reused_denoising(
    self: Any,
    original_diffuse: Any,
    sigma_threshold: float,
    *args: Any,
    **kwargs: Any,
) -> Any:
    from vllm_omni.diffusion.forward_context import (
        set_forward_context_denoise_step_idx,
    )

    bound = inspect.signature(original_diffuse).bind(self, *args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments

    latents = values["latents"]
    timesteps = values["timesteps"]
    prompt_embeds = values["prompt_embeds"]
    negative_prompt_embeds = values["negative_prompt_embeds"]
    guidance_low = values["guidance_low"]
    guidance_high = values["guidance_high"]
    boundary_timestep = values["boundary_timestep"]
    dtype = values["dtype"]
    attention_kwargs = values["attention_kwargs"] or {}
    latent_condition = values["latent_condition"]
    first_frame_mask = values["first_frame_mask"]

    reuse_prefix = getattr(self, "_viv_reuse_latent_prefix", None)
    if reuse_prefix is None:
        raise ValueError("missing latent reuse source prefix")

    resumed_scheduler = False
    with self.progress_bar(total=len(timesteps)) as pbar:
        for step_idx, t in enumerate(timesteps):
            self._current_timestep = t
            set_forward_context_denoise_step_idx(step_idx)

            sigma = _sigma_for_step(self.scheduler, step_idx, t)
            if sigma >= sigma_threshold:
                latent_path = _denoising_step_latent_path_from_prefix(
                    reuse_prefix, step_idx
                )
                latents = load_latents(
                    latent_path, device=latents.device, dtype=latents.dtype
                )
                _save_latents(
                    _denoising_step_latent_path_from_prefix(
                        getattr(self, "_viv_latent_prefix", None), step_idx
                    ),
                    "denoising_step",
                    latents,
                    {
                        "seed": getattr(self, "_viv_seed", None),
                        "step_idx": step_idx,
                        "timestep": _scalar_float(t),
                        "sigma": sigma,
                        "reused_from": str(latent_path),
                    },
                )
                self._viv_step_idx = step_idx + 1
                pbar.update()
                continue

            if not resumed_scheduler and step_idx > 0:
                _reset_scheduler_to_step(self.scheduler, step_idx)
                resumed_scheduler = True

            if boundary_timestep is not None and t < boundary_timestep:
                current_guidance_scale = guidance_high
                if self.transformer_2 is not None:
                    current_model = self.transformer_2
                elif self.transformer is not None:
                    current_model = self.transformer
                else:
                    raise RuntimeError("No transformer available for low-noise stage")
            else:
                current_guidance_scale = guidance_low
                if self.transformer is not None:
                    current_model = self.transformer
                elif self.transformer_2 is not None:
                    current_model = self.transformer_2
                else:
                    raise RuntimeError("No transformer available for high-noise stage")

            if self.expand_timesteps and latent_condition is not None:
                latent_model_input = (
                    (1 - first_frame_mask) * latent_condition
                    + first_frame_mask * latents
                )
                latent_model_input = latent_model_input.to(dtype)

                patch_size = self.transformer_config.patch_size
                patch_height = latents.shape[3] // patch_size[1]
                patch_width = latents.shape[4] // patch_size[2]

                patch_mask = first_frame_mask[
                    :, :, :, :: patch_size[1], :: patch_size[2]
                ]
                patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]
                temp_ts = (patch_mask[0][0] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

            do_true_cfg = (
                current_guidance_scale > 1.0 and negative_prompt_embeds is not None
            )
            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
                "current_model": current_model,
            }
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=current_guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )
            pbar.update()

    return latents


def _decode_reused_final_latent(
    self: Any, req: Any, output_type: str | None
) -> Any:
    from vllm_omni.diffusion.data import DiffusionOutput

    reuse_prefix = _reuse_prefix_from_request(req)
    latent_path = _final_noise_latent_path_from_prefix(reuse_prefix)
    if latent_path is None:
        raise ValueError("missing latent reuse source prefix")

    latents = load_latents(latent_path, device=self.device, dtype=self.vae.dtype)
    if output_type == "latent":
        output = latents
    else:
        import torch

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        output = self.vae.decode(latents, return_dict=False)[0]

    return DiffusionOutput(
        output=output,
        custom_output={"viv_reused_final_latent": str(latent_path)},
        stage_durations=self.stage_durations
        if hasattr(self, "stage_durations")
        else {},
    )


def _output_type_from_call(
    req: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> str | None:
    if "output_type" in kwargs:
        return kwargs["output_type"]
    sampling_params_output_type = getattr(
        getattr(req, "sampling_params", None), "output_type", None
    )
    if sampling_params_output_type is not None:
        return sampling_params_output_type
    # Wan22Pipeline.forward defaults output_type after seven optional positional args.
    if len(args) >= 8:
        return args[7]
    return "np"


def _sigma_for_step(scheduler: Any, step_idx: int, timestep: Any) -> float:
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is not None and len(sigmas) > step_idx:
        return _scalar_float(sigmas[step_idx])
    num_train_timesteps = getattr(
        getattr(scheduler, "config", None),
        "num_train_timesteps",
        getattr(scheduler, "num_train_timesteps", 1000),
    )
    return _scalar_float(timestep) / float(num_train_timesteps)


def _reset_scheduler_to_step(scheduler: Any, step_idx: int) -> None:
    if hasattr(scheduler, "model_outputs"):
        scheduler.model_outputs = [None] * len(scheduler.model_outputs)
    if hasattr(scheduler, "timestep_list"):
        scheduler.timestep_list = [None] * len(scheduler.timestep_list)
    if hasattr(scheduler, "lower_order_nums"):
        scheduler.lower_order_nums = 0
    if hasattr(scheduler, "last_sample"):
        scheduler.last_sample = None
    if hasattr(scheduler, "this_order"):
        scheduler.this_order = 1
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(step_idx)
    elif hasattr(scheduler, "_step_index"):
        scheduler._step_index = step_idx


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_latents(latents: Any) -> Any:
    if latents.__class__.__name__ == "AsyncLatents" and hasattr(latents, "_resolve"):
        return latents._resolve()
    return latents


def _scalar_float(value: Any) -> float:
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _is_rank_zero() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
    if rank not in {None, "", "0"}:
        return False
    try:
        import torch.distributed as dist

        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
    except Exception:
        return True
