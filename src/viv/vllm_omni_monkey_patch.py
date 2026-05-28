from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def install() -> None:
    if not _capture_enabled():
        return

    try:
        from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline
    except Exception:
        logger.exception("VIV latent capture monkey patch could not import Wan22Pipeline")
        return

    if getattr(Wan22Pipeline, "_viv_latent_capture_patched", False):
        return

    original_prepare_latents = Wan22Pipeline.prepare_latents
    original_diffuse = Wan22Pipeline.diffuse
    original_scheduler_step = Wan22Pipeline.scheduler_step_maybe_with_cfg

    def prepare_latents(self: Any, *args: Any, **kwargs: Any) -> Any:
        latents = original_prepare_latents(self, *args, **kwargs)
        seed = _seed_from_generator(kwargs.get("generator") if "generator" in kwargs else _positional_generator(args))
        self._viv_seed = seed
        self._viv_latent_dir = _latent_dir(seed)
        _save_latent(self._viv_latent_dir, "initial", latents, {"seed": seed})
        return latents

    def diffuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._viv_boundary_timestep = kwargs.get("boundary_timestep")
        self._viv_pre_boundary_saved = False
        self._viv_step_idx = 0
        try:
            result = original_diffuse(self, *args, **kwargs)
        finally:
            self._viv_boundary_timestep = None
        _save_latent(getattr(self, "_viv_latent_dir", None), "final", result, {"seed": getattr(self, "_viv_seed", None)})
        return result

    def scheduler_step_maybe_with_cfg(self: Any, noise_pred: Any, t: Any, latents: Any, do_true_cfg: bool) -> Any:
        boundary_timestep = getattr(self, "_viv_boundary_timestep", None)
        if (
            boundary_timestep is not None
            and _scalar_float(t) < _scalar_float(boundary_timestep)
            and not getattr(self, "_viv_pre_boundary_saved", False)
        ):
            _save_latent(
                getattr(self, "_viv_latent_dir", None),
                "pre_boundary",
                latents,
                {"step_idx": getattr(self, "_viv_step_idx", None), "timestep": _scalar_float(t)},
            )
            self._viv_pre_boundary_saved = True

        try:
            return original_scheduler_step(self, noise_pred, t, latents, do_true_cfg)
        finally:
            self._viv_step_idx = getattr(self, "_viv_step_idx", 0) + 1

    Wan22Pipeline.prepare_latents = prepare_latents
    Wan22Pipeline.diffuse = diffuse
    Wan22Pipeline.scheduler_step_maybe_with_cfg = scheduler_step_maybe_with_cfg
    Wan22Pipeline._viv_latent_capture_patched = True
    logger.info("Installed VIV latent capture monkey patch for vLLM-Omni Wan22Pipeline")


def _capture_enabled() -> bool:
    return os.environ.get("VIV_CAPTURE_LATENTS", "").lower() in {"1", "true", "yes", "on"}


def _latent_dir(seed: int | None) -> Path | None:
    root = os.environ.get("VIV_LATENT_ROOT")
    config_id = os.environ.get("VIV_CONFIG_ID", "unknown_config")
    if not root or seed is None:
        return None
    return Path(root) / config_id / str(seed) / "latents"


def _save_latent(latent_dir: Path | None, name: str, tensor: Any, metadata: dict[str, Any]) -> None:
    if latent_dir is None or tensor is None or not _is_rank_zero():
        return
    try:
        import torch
        from safetensors.torch import save_file

        if not isinstance(tensor, torch.Tensor):
            return

        latent_dir.mkdir(parents=True, exist_ok=True)
        path = latent_dir / f"{name}.safetensors"
        original_dtype = str(tensor.dtype)
        stored = tensor.detach().to("cpu")
        latent_dtype = os.environ.get("VIV_LATENT_DTYPE", "float16").lower()
        if latent_dtype in {"float16", "fp16"}:
            stored = stored.to(torch.float16)
        elif latent_dtype in {"bfloat16", "bf16"}:
            stored = stored.to(torch.bfloat16)
        elif latent_dtype not in {"preserve", "original"}:
            logger.warning("Unknown VIV_LATENT_DTYPE=%s; preserving latent dtype", latent_dtype)

        metadata_path = latent_dir / "metadata.json"
        existing: dict[str, Any] = {}
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing[name] = {
            **metadata,
            "shape": list(tensor.shape),
            "original_dtype": original_dtype,
            "stored_dtype": str(stored.dtype),
            "path": str(path),
        }
        save_file({name: stored}, path)
        metadata_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        logger.exception("Failed to capture VIV latent checkpoint %s", name)


def _seed_from_generator(generator: Any) -> int | None:
    if isinstance(generator, list):
        generator = generator[0] if generator else None
    if generator is None:
        return None
    try:
        return int(generator.initial_seed())
    except Exception:
        return None


def _positional_generator(args: tuple[Any, ...]) -> Any:
    return args[7] if len(args) > 7 else None


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

        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True
