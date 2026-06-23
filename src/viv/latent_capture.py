from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any


LATENT_PREFIX_EXTRA_ARG = "viv_latent_prefix"
PROMPT_EMBEDS_PREFIX_EXTRA_ARG = "viv_prompt_embeds_prefix"
REUSE_LATENT_PREFIX_EXTRA_ARG = "viv_reuse_latent_prefix"
REUSE_PREDICTIONS_EXTRA_ARG = "viv_reuse_predictions"
SAVE_FINAL_LATENTS_EXTRA_ARG = "viv_save_final_latents"
SAVE_PREDICTION_LATENTS_EXTRA_ARG = "viv_save_prediction_latents"
SKIP_REUSED_STEPS_EXTRA_ARG = "viv_skip_reused_steps"
REUSE_FINAL_LATENT_EXTRA_ARG = "viv_reuse_final_latent"
SAVE_PROMPT_EMBEDS_EXTRA_ARG = "viv_save_prompt_embeds"
REUSE_PROMPT_EMBEDS_EXTRA_ARG = "viv_reuse_prompt_embeds"
SIGMA_SCHEDULE_PATH_EXTRA_ARG = "viv_sigma_schedule_path"


def install_wan_latent_capture() -> None:
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

    if getattr(Wan22Pipeline, "_viv_latent_capture_patched", False):
        return

    original_forward = Wan22Pipeline.forward
    original_diffuse = Wan22Pipeline.diffuse
    original_scheduler_step = Wan22Pipeline.scheduler_step_maybe_with_cfg

    def forward(self: Any, req: Any, *args: Any, **kwargs: Any) -> Any:
        previous_prefix = getattr(self, "_viv_latent_prefix", None)
        previous_prompt_embeds_prefix = getattr(
            self, "_viv_prompt_embeds_prefix", None
        )
        previous_seed = getattr(self, "_viv_seed", None)
        previous_reuse_prefix = getattr(self, "_viv_reuse_latent_prefix", None)
        previous_reuse_predictions = getattr(self, "_viv_reuse_predictions", None)
        previous_save_final_latents = getattr(self, "_viv_save_final_latents", None)
        previous_save_prediction_latents = getattr(
            self, "_viv_save_prediction_latents", None
        )
        previous_skip_reused_steps = getattr(self, "_viv_skip_reused_steps", None)
        previous_suppress_prediction_latent_save = getattr(
            self, "_viv_suppress_prediction_latent_save", None
        )
        previous_save_prompt_embeds = getattr(
            self, "_viv_save_prompt_embeds", None
        )
        previous_sigma_schedule_path = getattr(
            self, "_viv_sigma_schedule_path", None
        )
        previous_sigma_schedule = getattr(self, "_viv_sigma_schedule", None)
        self._viv_latent_prefix = _prefix_from_request(req)
        self._viv_prompt_embeds_prefix = _prompt_embeds_prefix_from_request(req)
        self._viv_seed = _seed_from_request(req)
        self._viv_reuse_latent_prefix = _reuse_prefix_from_request(req)
        self._viv_reuse_predictions = _reuse_predictions_from_request(req)
        self._viv_save_final_latents = _save_final_latents_from_request(req)
        self._viv_save_prediction_latents = (
            _save_prediction_latents_from_request(req)
        )
        self._viv_skip_reused_steps = _skip_reused_steps_from_request(req)
        self._viv_suppress_prediction_latent_save = False
        self._viv_save_prompt_embeds = _save_prompt_embeds_from_request(req)
        self._viv_sigma_schedule_path = _sigma_schedule_path_from_request(req)
        self._viv_sigma_schedule = None
        try:
            if _reuse_final_latent_from_request(req):
                _write_sigma_schedule(self._viv_sigma_schedule_path, [])
                return _decode_reused_final_latent(
                    self, req, _output_type_from_call(req, args, kwargs)
                )
            if _reuse_prompt_embeds_from_request(req):
                reuse_prefix = getattr(self, "_viv_reuse_latent_prefix", None)
                if reuse_prefix is None:
                    raise ValueError("missing prompt embedding reuse source prefix")
                prompt_embeds, negative_prompt_embeds = load_prompt_embeddings(
                    prompt_embeddings_path(reuse_prefix)
                )
                return _call_forward_with_prompt_embeddings(
                    original_forward,
                    self,
                    req,
                    args,
                    kwargs,
                    prompt_embeds,
                    negative_prompt_embeds,
                )
            return original_forward(self, req, *args, **kwargs)
        finally:
            self._viv_latent_prefix = previous_prefix
            self._viv_prompt_embeds_prefix = previous_prompt_embeds_prefix
            self._viv_seed = previous_seed
            self._viv_reuse_latent_prefix = previous_reuse_prefix
            self._viv_reuse_predictions = previous_reuse_predictions
            self._viv_save_final_latents = previous_save_final_latents
            self._viv_save_prediction_latents = previous_save_prediction_latents
            self._viv_skip_reused_steps = previous_skip_reused_steps
            self._viv_suppress_prediction_latent_save = (
                previous_suppress_prediction_latent_save
            )
            self._viv_save_prompt_embeds = previous_save_prompt_embeds
            self._viv_sigma_schedule_path = previous_sigma_schedule_path
            self._viv_sigma_schedule = previous_sigma_schedule

    def diffuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_step_idx = getattr(self, "_viv_step_idx", None)
        previous_sigma_schedule = getattr(self, "_viv_sigma_schedule", None)
        reuse_predictions = getattr(self, "_viv_reuse_predictions", None)
        self._viv_step_idx = 0
        self._viv_sigma_schedule = []
        try:
            if getattr(self, "_viv_save_prompt_embeds", False):
                values = _diffuse_arguments(original_diffuse, self, args, kwargs)
                _save_prompt_embeddings(
                    prompt_embeddings_path_from_prefix(
                        getattr(self, "_viv_prompt_embeds_prefix", None)
                    ),
                    values["prompt_embeds"],
                    values["negative_prompt_embeds"],
                    {
                        "seed": getattr(self, "_viv_seed", None),
                    },
                )
            if reuse_predictions is None:
                latents = original_diffuse(self, *args, **kwargs)
            else:
                latents = _diffuse_with_reused_denoising(
                    self,
                    original_diffuse,
                    reuse_predictions,
                    *args,
                    **kwargs,
                )
            final_latent_path = _final_noise_latent_path_from_prefix(
                getattr(self, "_viv_latent_prefix", None)
            )
            if getattr(self, "_viv_save_final_latents", False):
                _save_latents(
                    final_latent_path,
                    "final_noise",
                    latents,
                    {
                        "seed": getattr(self, "_viv_seed", None),
                    },
                )
            _write_sigma_schedule(
                getattr(self, "_viv_sigma_schedule_path", None),
                getattr(self, "_viv_sigma_schedule", None),
            )
            return latents
        finally:
            if reuse_predictions is not None and hasattr(self, "_sync_pp_send"):
                self._sync_pp_send()
            self._viv_step_idx = previous_step_idx
            self._viv_sigma_schedule = previous_sigma_schedule

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
        sigma = _sigma_for_step(self.scheduler, step_idx, t)
        sigma_schedule = getattr(self, "_viv_sigma_schedule", None)
        if sigma_schedule is not None:
            sigma_schedule.append(sigma)
        noise_pred_to_save = getattr(self, "_viv_noise_pred_to_save", noise_pred)
        if (
            getattr(self, "_viv_save_prediction_latents", False)
            and not getattr(self, "_viv_suppress_prediction_latent_save", False)
        ):
            _save_noise_pred(
                _denoising_step_latent_path_from_prefix(
                    getattr(self, "_viv_latent_prefix", None), step_idx
                ),
                "denoising_step",
                noise_pred_to_save,
                {
                    "seed": getattr(self, "_viv_seed", None),
                    "step_idx": step_idx,
                    "timestep": _scalar_float(t),
                    "sigma": sigma,
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


def prompt_embeddings_path(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}.prompt_embeds.safetensors")


def denoising_step_latent_paths(video_path: Path, num_inference_steps: int) -> list[Path]:
    return [
        denoising_step_latent_path(video_path, step_idx)
        for step_idx in range(num_inference_steps)
    ]


def saved_denoising_prediction_count(video_path: Path) -> int:
    step_idx = 0
    while True:
        latent_path = denoising_step_latent_path(video_path, step_idx)
        if not latent_path.exists():
            return step_idx
        step_idx += 1


def load_latents(
    latent_path: Path, device: Any | None = None, dtype: Any | None = None
) -> Any:
    return _load_tensor(latent_path, "latents", device=device, dtype=dtype)


def _load_noise_pred(
    latent_path: Path, device: Any | None = None, dtype: Any | None = None
) -> Any:
    return _load_tensor(latent_path, "noise_pred", device=device, dtype=dtype)


def load_prompt_embeddings(
    embeddings_path: Path, device: Any | None = None, dtype: Any | None = None
) -> tuple[Any, Any | None]:
    tensors, _ = _load_prompt_embeddings_with_metadata(
        embeddings_path, device=device, dtype=dtype
    )
    return tensors


def prompt_embeddings_sha256_metadata(
    embeddings_path: Path,
) -> tuple[str, str | None]:
    metadata = _safetensors_metadata(embeddings_path)
    prompt_sha256 = metadata.get("prompt_embeds_sha256")
    if not prompt_sha256:
        raise ValueError(
            f"{embeddings_path} does not contain prompt_embeds_sha256 metadata"
        )
    return prompt_sha256, metadata.get("negative_prompt_embeds_sha256")


def _load_prompt_embeddings_with_metadata(
    embeddings_path: Path,
    device: Any | None = None,
    dtype: Any | None = None,
) -> tuple[tuple[Any, Any | None], dict[str, str]]:
    from safetensors import safe_open

    with safe_open(str(embeddings_path), framework="pt", device="cpu") as tensors:
        keys = tensors.keys()
        if "prompt_embeds" not in keys:
            raise ValueError(f"{embeddings_path} does not contain a prompt_embeds tensor")
        prompt_embeds = tensors.get_tensor("prompt_embeds")
        negative_prompt_embeds = (
            tensors.get_tensor("negative_prompt_embeds")
            if "negative_prompt_embeds" in keys
            else None
        )
        metadata = dict(tensors.metadata() or {})
    if device is not None or dtype is not None:
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device, dtype=dtype
            )
    return (prompt_embeds, negative_prompt_embeds), metadata


def _load_tensor(
    tensor_path: Path,
    key: str,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any:
    tensor, _ = _load_tensor_with_metadata(
        tensor_path, key, device=device, dtype=dtype
    )
    return tensor


def _load_tensor_with_metadata(
    tensor_path: Path,
    key: str,
    device: Any | None = None,
    dtype: Any | None = None,
) -> tuple[Any, dict[str, str]]:
    from safetensors import safe_open

    with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
        if key not in tensors.keys():
            raise ValueError(f"{tensor_path} does not contain a {key} tensor")
        tensor = tensors.get_tensor(key)
        metadata = dict(tensors.metadata() or {})
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor, metadata


def latent_sha256(tensor: Any) -> str:
    import torch

    tensor = _resolve_latents(tensor)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")

    tensor = tensor.detach().cpu().contiguous()
    metadata = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    raw_bytes = tensor.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(metadata_bytes)
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def save_initial_noise_latents(
    latent_path: Path, latents: Any, seed: int
) -> str | None:
    return _save_latents(
        latent_path,
        "initial_noise",
        latents,
        {
            "seed": seed,
        },
    )


def safetensors_sha256_metadata(latent_path: Path) -> str:
    metadata = _safetensors_metadata(latent_path)
    sha256 = metadata.get("sha256")
    if not sha256:
        raise ValueError(f"{latent_path} does not contain sha256 metadata")
    return sha256


def read_sigma_schedule(schedule_path: Path) -> list[float] | None:
    if not schedule_path.exists():
        return None
    with schedule_path.open("r", encoding="utf-8") as fh:
        raw_schedule = json.load(fh)
    if not isinstance(raw_schedule, list):
        raise ValueError(f"{schedule_path} sigma schedule must be a JSON array")
    return [float(sigma) for sigma in raw_schedule]


def _safetensors_metadata(latent_path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(latent_path), framework="pt", device="cpu") as tensors:
        metadata = tensors.metadata() or {}
    return dict(metadata)


def _save_latents(
    latent_path: Path | None,
    kind: str,
    latents: Any,
    metadata: dict[str, Any],
) -> str | None:
    return _save_tensor(latent_path, "latents", kind, latents, metadata)


def _save_noise_pred(
    latent_path: Path | None,
    kind: str,
    noise_pred: Any,
    metadata: dict[str, Any],
) -> str | None:
    return _save_tensor(latent_path, "noise_pred", kind, noise_pred, metadata)


def _save_prompt_embeddings(
    embeddings_path: Path | None,
    prompt_embeds: Any,
    negative_prompt_embeds: Any | None,
    metadata: dict[str, Any],
) -> tuple[str, str | None] | None:
    if embeddings_path is None or prompt_embeds is None or not _is_rank_zero():
        return None

    import torch
    from safetensors.torch import save_file

    prompt_embeds = _resolve_latents(prompt_embeds)
    if not isinstance(prompt_embeds, torch.Tensor):
        return None

    tensors = {
        "prompt_embeds": prompt_embeds.detach().to("cpu").contiguous(),
    }
    negative_sha256 = None
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = _resolve_latents(negative_prompt_embeds)
        if not isinstance(negative_prompt_embeds, torch.Tensor):
            return None
        tensors["negative_prompt_embeds"] = (
            negative_prompt_embeds.detach().to("cpu").contiguous()
        )
        negative_sha256 = latent_sha256(tensors["negative_prompt_embeds"])

    prompt_sha256 = latent_sha256(tensors["prompt_embeds"])
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = embeddings_path.with_name(
        f"{embeddings_path.stem}.tmp{embeddings_path.suffix}"
    )
    stored_metadata = {
        "kind": "prompt_embeddings",
        **{
            metadata_key: str(value)
            for metadata_key, value in metadata.items()
            if value is not None
        },
        "prompt_embeds_sha256": prompt_sha256,
    }
    if negative_sha256 is not None:
        stored_metadata["negative_prompt_embeds_sha256"] = negative_sha256
    save_file(tensors, str(tmp_path), metadata=stored_metadata)
    tmp_path.replace(embeddings_path)
    return prompt_sha256, negative_sha256


def _save_tensor(
    tensor_path: Path | None,
    key: str,
    kind: str,
    tensor: Any,
    metadata: dict[str, Any],
) -> str | None:
    if tensor_path is None or tensor is None or not _is_rank_zero():
        return None

    import torch
    from safetensors.torch import save_file

    tensor = _resolve_latents(tensor)
    if not isinstance(tensor, torch.Tensor):
        return None

    tensor_to_save = tensor.detach().to("cpu").contiguous()
    sha256 = latent_sha256(tensor_to_save)

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tensor_path.with_name(f"{tensor_path.stem}.tmp{tensor_path.suffix}")
    stored_metadata = {
        "kind": kind,
        **{
            metadata_key: str(value)
            for metadata_key, value in metadata.items()
            if value is not None and metadata_key != "sha256"
        },
        "sha256": sha256,
    }
    save_file(
        {key: tensor_to_save},
        str(tmp_path),
        metadata=stored_metadata,
    )
    tmp_path.replace(tensor_path)
    return sha256


def _prefix_from_request(req: Any) -> Path | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    prefix = extra_args.get(LATENT_PREFIX_EXTRA_ARG)
    if prefix is None:
        return None
    return Path(prefix)


def _prompt_embeds_prefix_from_request(req: Any) -> Path | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    prefix = extra_args.get(PROMPT_EMBEDS_PREFIX_EXTRA_ARG)
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


def _reuse_predictions_from_request(req: Any) -> int | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    count = extra_args.get(REUSE_PREDICTIONS_EXTRA_ARG)
    if count is None:
        return None
    return int(count)


def _save_final_latents_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(SAVE_FINAL_LATENTS_EXTRA_ARG))


def _save_prediction_latents_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(SAVE_PREDICTION_LATENTS_EXTRA_ARG))


def _skip_reused_steps_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(SKIP_REUSED_STEPS_EXTRA_ARG))


def _save_prompt_embeds_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(SAVE_PROMPT_EMBEDS_EXTRA_ARG))


def _reuse_prompt_embeds_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(REUSE_PROMPT_EMBEDS_EXTRA_ARG))


def _reuse_final_latent_from_request(req: Any) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return _truthy(extra_args.get(REUSE_FINAL_LATENT_EXTRA_ARG))


def _sigma_schedule_path_from_request(req: Any) -> Path | None:
    sampling_params = getattr(req, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    path = extra_args.get(SIGMA_SCHEDULE_PATH_EXTRA_ARG)
    if path is None:
        return None
    return Path(path)


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


def prompt_embeddings_path_from_prefix(prefix: Path | None) -> Path | None:
    if prefix is None:
        return None
    return prefix.with_name(f"{prefix.name}.prompt_embeds.safetensors")


def _call_forward_with_prompt_embeddings(
    original_forward: Any,
    self: Any,
    req: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    prompt_embeds: Any,
    negative_prompt_embeds: Any | None,
) -> Any:
    bound = inspect.signature(original_forward).bind_partial(
        self, req, *args, **kwargs
    )
    bound.arguments["prompt"] = None
    bound.arguments["negative_prompt"] = None
    bound.arguments["prompt_embeds"] = prompt_embeds
    bound.arguments["negative_prompt_embeds"] = negative_prompt_embeds

    previous_prompts = getattr(req, "prompts", None)
    if previous_prompts is not None:
        req.prompts = [_prompt_without_text(prompt) for prompt in previous_prompts]
    try:
        return original_forward(*bound.args, **bound.kwargs)
    finally:
        if previous_prompts is not None:
            req.prompts = previous_prompts


def _prompt_without_text(prompt: Any) -> Any:
    if isinstance(prompt, str):
        return {}
    if isinstance(prompt, dict):
        cleaned = dict(prompt)
        cleaned.pop("prompt", None)
        cleaned.pop("negative_prompt", None)
        return cleaned
    return prompt


def _diffuse_arguments(
    original_diffuse: Any,
    self: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    bound = inspect.signature(original_diffuse).bind(self, *args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _diffuse_with_reused_denoising(
    self: Any,
    original_diffuse: Any,
    reuse_predictions: int,
    *args: Any,
    **kwargs: Any,
) -> Any:
    from vllm_omni.diffusion.forward_context import (
        set_forward_context_denoise_step_idx,
    )

    values = _diffuse_arguments(original_diffuse, self, args, kwargs)

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
    skip_reused_steps = bool(getattr(self, "_viv_skip_reused_steps", False))
    if reuse_prefix is None:
        raise ValueError("missing latent reuse source prefix")
    if reuse_predictions > len(timesteps):
        raise ValueError(
            f"--reuse-prediction-latents={reuse_predictions} exceeds this run's "
            f"{len(timesteps)} denoising steps"
        )

    timesteps = _apply_reused_prediction_schedule(
        self.scheduler,
        timesteps,
        reuse_prefix,
        reuse_predictions,
    )

    with self.progress_bar(total=len(timesteps)) as pbar:
        for step_idx, t in enumerate(timesteps):
            self._current_timestep = t
            set_forward_context_denoise_step_idx(step_idx)

            if step_idx < reuse_predictions and skip_reused_steps:
                latent_path = _denoising_step_latent_path_from_prefix(
                    reuse_prefix, step_idx
                )
                noise_pred = _load_noise_pred(
                    latent_path, device=latents.device, dtype=dtype
                )
                previous_suppress_prediction_latent_save = getattr(
                    self, "_viv_suppress_prediction_latent_save", False
                )
                self._viv_suppress_prediction_latent_save = True
                try:
                    latents = self.scheduler_step_maybe_with_cfg(
                        noise_pred, t, latents, do_true_cfg=False
                    )
                finally:
                    self._viv_suppress_prediction_latent_save = (
                        previous_suppress_prediction_latent_save
                    )
                pbar.update()
                continue

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

            step_noise_pred = noise_pred
            step_do_true_cfg = do_true_cfg
            if step_idx < reuse_predictions:
                had_previous_noise_pred_to_save = hasattr(
                    self, "_viv_noise_pred_to_save"
                )
                previous_noise_pred_to_save = getattr(
                    self, "_viv_noise_pred_to_save", None
                )
                latent_path = _denoising_step_latent_path_from_prefix(
                    reuse_prefix, step_idx
                )
                step_noise_pred = _load_noise_pred(
                    latent_path, device=latents.device, dtype=dtype
                )
                step_do_true_cfg = False
                self._viv_noise_pred_to_save = noise_pred

            try:
                latents = self.scheduler_step_maybe_with_cfg(
                    step_noise_pred, t, latents, step_do_true_cfg
                )
            finally:
                if step_idx < reuse_predictions:
                    if had_previous_noise_pred_to_save:
                        self._viv_noise_pred_to_save = previous_noise_pred_to_save
                    else:
                        del self._viv_noise_pred_to_save
            pbar.update()

    return latents


def _apply_reused_prediction_schedule(
    scheduler: Any,
    timesteps: Any,
    reuse_prefix: Path,
    reuse_predictions: int,
) -> Any:
    import torch

    source_sigmas, source_timesteps = _load_saved_denoising_schedule(reuse_prefix)
    target_steps = len(timesteps)
    remaining_steps = target_steps - reuse_predictions

    if reuse_predictions > len(source_sigmas):
        raise ValueError(
            f"--reuse-prediction-latents={reuse_predictions} requested, but only "
            f"{len(source_sigmas)} saved denoising predictions were found"
        )
    if remaining_steps > 0 and len(source_sigmas) <= reuse_predictions:
        raise ValueError(
            "cannot build the remaining denoising schedule: expected at least "
            f"{reuse_predictions + 1} saved prediction sigmas, found "
            f"{len(source_sigmas)}"
        )

    num_train_timesteps = _num_train_timesteps(scheduler)
    sigmas = list(source_sigmas[:reuse_predictions])
    adjusted_timesteps = [
        source_timesteps[idx]
        if source_timesteps[idx] is not None
        else source_sigmas[idx] * num_train_timesteps
        for idx in range(reuse_predictions)
    ]

    if remaining_steps > 0:
        tail_sigmas = _lambda_spaced_tail_sigmas(
            source_sigmas[reuse_predictions],
            source_sigmas[-1],
            remaining_steps,
        )
        sigmas.extend(tail_sigmas)
        adjusted_timesteps.extend(
            sigma * num_train_timesteps for sigma in tail_sigmas
        )

    if len(sigmas) != target_steps:
        raise AssertionError("internal error while building denoising schedule")

    final_sigma = _final_scheduler_sigma(scheduler)
    device = getattr(timesteps, "device", None)
    scheduler.sigmas = torch.tensor(
        [*sigmas, final_sigma], dtype=torch.float32, device="cpu"
    )
    scheduler.timesteps = torch.tensor(
        adjusted_timesteps, dtype=torch.float32, device=device
    )
    scheduler.num_inference_steps = len(adjusted_timesteps)
    _reset_scheduler_history(scheduler)
    return scheduler.timesteps


def _load_saved_denoising_schedule(
    reuse_prefix: Path,
) -> tuple[list[float], list[float | None]]:
    sigmas: list[float] = []
    timesteps: list[float | None] = []
    step_idx = 0
    while True:
        latent_path = _denoising_step_latent_path_from_prefix(reuse_prefix, step_idx)
        if latent_path is None or not latent_path.exists():
            break
        metadata = _safetensors_metadata(latent_path)
        sigmas.append(_required_metadata_float(latent_path, metadata, "sigma"))
        timesteps.append(_optional_metadata_float(metadata, "timestep"))
        step_idx += 1

    if not sigmas:
        raise ValueError(f"no saved denoising predictions found for {reuse_prefix}")
    return sigmas, timesteps


def _lambda_spaced_tail_sigmas(
    start_sigma: float,
    end_sigma: float,
    count: int,
) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [start_sigma]

    lambda_start = _sigma_to_lambda(start_sigma)
    lambda_end = _sigma_to_lambda(end_sigma)
    return [
        _lambda_to_sigma(
            lambda_start + (lambda_end - lambda_start) * idx / (count - 1)
        )
        for idx in range(count)
    ]


def _sigma_to_lambda(sigma: float) -> float:
    sigma = min(max(float(sigma), 1e-12), 1.0 - 1e-12)
    return math.log((1.0 - sigma) / sigma)


def _lambda_to_sigma(lambda_value: float) -> float:
    if lambda_value >= 0:
        exp_neg = math.exp(-lambda_value)
        return exp_neg / (1.0 + exp_neg)
    return 1.0 / (1.0 + math.exp(lambda_value))


def _required_metadata_float(
    latent_path: Path,
    metadata: dict[str, str],
    key: str,
) -> float:
    value = metadata.get(key)
    if value is None:
        raise ValueError(f"{latent_path} does not contain {key} metadata")
    return float(value)


def _optional_metadata_float(metadata: dict[str, str], key: str) -> float | None:
    value = metadata.get(key)
    return None if value is None else float(value)


def _num_train_timesteps(scheduler: Any) -> float:
    return float(
        getattr(
            getattr(scheduler, "config", None),
            "num_train_timesteps",
            getattr(scheduler, "num_train_timesteps", 1000),
        )
    )


def _final_scheduler_sigma(scheduler: Any) -> float:
    final_sigmas_type = getattr(
        getattr(scheduler, "config", None), "final_sigmas_type", "zero"
    )
    if final_sigmas_type == "sigma_min":
        return float(getattr(scheduler, "sigma_min", 0.0))
    return 0.0


def _reset_scheduler_history(scheduler: Any) -> None:
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
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None


def _write_sigma_schedule(
    schedule_path: Path | None, sigma_schedule: list[float] | None
) -> None:
    if schedule_path is None or sigma_schedule is None or not _is_rank_zero():
        return

    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = schedule_path.with_name(
        f"{schedule_path.stem}.tmp{schedule_path.suffix}"
    )
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump([float(sigma) for sigma in sigma_schedule], fh, indent=2)
        fh.write("\n")
    tmp_path.replace(schedule_path)


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
