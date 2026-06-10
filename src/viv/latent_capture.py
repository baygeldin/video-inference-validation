from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


LATENT_PREFIX_EXTRA_ARG = "viv_latent_prefix"


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
        self._viv_latent_prefix = _prefix_from_request(req)
        self._viv_seed = _seed_from_request(req)
        try:
            return original_forward(self, req, *args, **kwargs)
        finally:
            self._viv_latent_prefix = previous_prefix
            self._viv_seed = previous_seed

    def diffuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous_step_idx = getattr(self, "_viv_step_idx", None)
        self._viv_step_idx = 0
        try:
            latents = original_diffuse(self, *args, **kwargs)
            _save_latents(
                _final_noise_latent_path_from_prefix(
                    getattr(self, "_viv_latent_prefix", None)
                ),
                "final_noise",
                latents,
                {
                    "seed": getattr(self, "_viv_seed", None),
                    "sha256": latent_sha256(latents),
                },
            )
            return latents
        finally:
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
