from __future__ import annotations

from viv.models import InferenceConfig
from viv.paths import CONFIGS_PATH


def load_inference_config(config_name: str) -> InferenceConfig:
    import yaml

    name = config_name.strip()
    if not name:
        raise ValueError("you must provide config name")

    with CONFIGS_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{CONFIGS_PATH}: config file must contain a mapping")
    if name not in raw:
        available = ", ".join(sorted(str(key) for key in raw))
        raise ValueError(f"unknown config '{name}' (available: {available})")

    values = raw[name]
    if not isinstance(values, dict):
        raise ValueError(f"{CONFIGS_PATH}: config '{name}' must be a mapping")

    required = {
        "width",
        "height",
        "num_frames",
        "fps",
        "num_inference_steps",
        "boundary_ratio",
        "flow_shift",
        "guidance_scale",
        "guidance_scale_2",
        "model_name",
        "model_revision",
        "attention_backend",
        "export_quality",
    }
    missing = sorted(required - values.keys())
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{CONFIGS_PATH}: config '{name}' missing {joined}")

    return InferenceConfig(
        width=_positive_int(values["width"], name, "width"),
        height=_positive_int(values["height"], name, "height"),
        num_frames=_positive_int(values["num_frames"], name, "num_frames"),
        fps=_positive_int(values["fps"], name, "fps"),
        num_inference_steps=_positive_int(
            values["num_inference_steps"], name, "num_inference_steps"
        ),
        boundary_ratio=_ratio(values["boundary_ratio"], name, "boundary_ratio"),
        flow_shift=_positive_float(values["flow_shift"], name, "flow_shift"),
        guidance_scale=_float(values["guidance_scale"], name, "guidance_scale"),
        guidance_scale_2=_float(values["guidance_scale_2"], name, "guidance_scale_2"),
        model_name=_non_empty_str(values["model_name"], name, "model_name"),
        model_revision=_non_empty_str(values["model_revision"], name, "model_revision"),
        attention_backend=_attention_backend(
            values["attention_backend"], name, "attention_backend"
        ),
        export_quality=_quality(values["export_quality"], name, "export_quality"),
        tensor_parallelism=_positive_int(
            values.get("tensor_parallelism", 1), name, "tensor_parallelism"
        ),
        cache_backend=_cache_backend(
            values.get("cache_backend", None), name, "cache_backend"
        ),
        quantization=_quantization(
            values.get("quantization", None), name, "quantization"
        ),
        force_cutlass_fp8=_bool(
            values.get("force_cutlass_fp8", False), name, "force_cutlass_fp8"
        ),
        random_seed=_bool(values.get("random_seed", False), name, "random_seed"),
    )


def _positive_int(value: object, config_name: str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config '{config_name}' field '{field}' must be an int") from exc
    if parsed <= 0:
        raise ValueError(f"config '{config_name}' field '{field}' must be positive")
    return parsed


def _float(value: object, config_name: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config '{config_name}' field '{field}' must be a number") from exc


def _positive_float(value: object, config_name: str, field: str) -> float:
    parsed = _float(value, config_name, field)
    if parsed <= 0:
        raise ValueError(f"config '{config_name}' field '{field}' must be positive")
    return parsed


def _ratio(value: object, config_name: str, field: str) -> float:
    parsed = _float(value, config_name, field)
    if parsed < 0 or parsed > 1:
        raise ValueError(
            f"config '{config_name}' field '{field}' must be between 0 and 1"
        )
    return parsed


def _quality(value: object, config_name: str, field: str) -> float:
    parsed = _float(value, config_name, field)
    if parsed < 0 or parsed > 10:
        raise ValueError(
            f"config '{config_name}' field '{field}' must be between 0 and 10"
        )
    return parsed


def _bool(value: object, config_name: str, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"config '{config_name}' field '{field}' must be a boolean")


def _non_empty_str(value: object, config_name: str, field: str) -> str:
    parsed = str(value or "").strip()
    if not parsed:
        raise ValueError(
            f"config '{config_name}' field '{field}' must be a non-empty string"
        )
    return parsed


def _attention_backend(value: object, config_name: str, field: str) -> str:
    parsed = _non_empty_str(value, config_name, field).upper()
    allowed = {"FLASH_ATTN", "TORCH_SDPA", "SAGE_ATTN"}
    if parsed not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(
            f"config '{config_name}' field '{field}' must be one of: {joined}"
        )
    return parsed


def _cache_backend(value: object, config_name: str, field: str) -> str | None:
    if value is None:
        return None

    parsed = _non_empty_str(value, config_name, field).lower()
    allowed = {"none", "cache_dit", "tea_cache"}
    if parsed not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(
            f"config '{config_name}' field '{field}' must be one of: {joined}"
        )
    if parsed == "none":
        return None
    return parsed


def _quantization(value: object, config_name: str, field: str) -> str | None:
    if value is None:
        return None

    parsed = _non_empty_str(value, config_name, field).lower()
    allowed = {"none", "fp8"}
    if parsed not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(
            f"config '{config_name}' field '{field}' must be one of: {joined}"
        )
    if parsed == "none":
        return None
    return parsed
