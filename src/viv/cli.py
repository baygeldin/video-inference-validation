from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CONFIGS_PATH = PROJECT_ROOT / "configs.yml"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv",
        description="Generate Wan2.2 videos with vLLM-Omni.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        dest="prompts_arg",
        required=True,
        help="Prompt collection identifier, or path to the JSONL file",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="default",
        help="Inference config identifier",
    )
    parser.add_argument("output_dir", type=Path, help="Output folder path")
    args = parser.parse_args(argv)

    try:
        prompts_path = resolve_prompts_path(args.prompts_arg)
        config = load_inference_config(args.config)
        run(prompts_path, args.output_dir, args.config.strip(), config)
    except Exception as exc:
        print(f"viv: error: {exc}", file=sys.stderr)
        return 2
    return 0


def run(
    prompts_path: Path, output_dir: Path, config_name: str, config: InferenceConfig
) -> None:
    prompts = load_prompts(prompts_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "using "
        f"{config.model_name}@{config.model_revision}, "
        f"attention={config.attention_backend}, "
        f"export_quality={config.export_quality}, "
        f"tp={config.tensor_parallelism}, "
        f"cache_backend={config.cache_backend}",
        flush=True,
    )
    generator = OfflineVideoGenerator(config)
    environment = collect_environment_metadata()
    for prompt in prompts:
        video_path = output_dir / f"{prompt.id}.mp4"
        metadata_path = output_dir / f"{prompt.id}.json"
        print(f"generating {prompt.id} -> {video_path}", flush=True)
        result = generator.generate(prompt, video_path)
        write_sidecar_metadata(
            metadata_path,
            config_name,
            prompt,
            config,
            result,
            environment,
        )
        print(f"completed {prompt.id}", flush=True)


def write_sidecar_metadata(
    metadata_path: Path,
    config_name: str,
    prompt: Prompt,
    config: InferenceConfig,
    result: GenerationResult,
    environment: EnvironmentMetadata,
) -> None:
    metadata = {
        "config_name": config_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": result.duration_seconds,
        "prompt_id": prompt.id,
        "prompt_text": prompt.prompt,
        "seed": result.seed,
        "height": config.height,
        "width": config.width,
        "fps": config.fps,
        "num_frames": config.num_frames,
        "num_inference_steps": config.num_inference_steps,
        "guidance_scale": config.guidance_scale,
        "guidance_scale_2": config.guidance_scale_2,
        "model_name": config.model_name,
        "attention_backend": config.attention_backend,
        "model_revision": config.model_revision,
        "quality": config.export_quality,
        "tensor_parallelism": config.tensor_parallelism,
        "cache_backend": config.cache_backend,
        "environment": dict(environment),
    }
    tmp_path = metadata_path.with_suffix(".tmp.json")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    tmp_path.replace(metadata_path)


def collect_environment_metadata() -> dict[str, str | None]:
    nvidia_smi = _nvidia_smi_metadata()
    return {
        "gpu_model": _gpu_model()
        or nvidia_smi.get("gpu_model")
        or _gpu_model_from_proc(),
        "vllm_omni_version": _package_version("vllm-omni"),
        "pytorch_version": _pytorch_version(),
        "cuda_version": _cuda_version() or nvidia_smi.get("cuda_version"),
        "nvidia_driver_version": nvidia_smi.get("nvidia_driver_version")
        or _nvidia_driver_version_from_proc(),
        "ffmpeg_version": _ffmpeg_version(),
        "python_version": sys.version,
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _pytorch_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return str(torch.__version__)


def _cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda_version:
            return str(cuda_version)

    nvcc_output = _run_command(["nvcc", "--version"])
    if nvcc_output:
        for line in nvcc_output.splitlines():
            marker = "release "
            if marker in line:
                return line.split(marker, 1)[1].split(",", 1)[0].strip()
    return None


def _gpu_model() -> str | None:
    try:
        import torch
    except Exception:
        return None

    try:
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        return None
    return None


def _nvidia_smi_metadata() -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "gpu_model": None,
        "nvidia_driver_version": None,
        "cuda_version": None,
    }
    query_output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if query_output:
        first_line = query_output.splitlines()[0]
        parts = [part.strip() for part in first_line.split(",", maxsplit=1)]
        if parts:
            metadata["gpu_model"] = parts[0] or None
        if len(parts) > 1:
            metadata["nvidia_driver_version"] = parts[1] or None

    summary_output = _run_command(["nvidia-smi"])
    if summary_output and "CUDA Version:" in summary_output:
        metadata["cuda_version"] = (
            summary_output.split("CUDA Version:", 1)[1].split("|", 1)[0].strip()
            or None
        )
    return metadata


def _gpu_model_from_proc() -> str | None:
    for info_path in sorted(Path("/proc/driver/nvidia/gpus").glob("*/information")):
        try:
            for line in info_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("Model:"):
                    return line.split(":", 1)[1].strip() or None
        except OSError:
            continue
    return None


def _nvidia_driver_version_from_proc() -> str | None:
    version_path = Path("/proc/driver/nvidia/version")
    try:
        contents = version_path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"Kernel Module\s+(\S+)", contents)
    if match:
        return match.group(1)
    return contents.splitlines()[0].strip() if contents.splitlines() else None


def _ffmpeg_version() -> str | None:
    output = _run_command(["ffmpeg", "-version"])
    if not output:
        return None
    first_line = output.splitlines()[0].strip()
    prefix = "ffmpeg version "
    if first_line.startswith(prefix):
        return first_line[len(prefix) :].split(" ", 1)[0]
    return first_line or None


def _run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: prompt row must be an object")

            id_value = str(raw.get("id") or "").strip()
            prompt = str(raw.get("prompt") or "").strip()
            if not id_value:
                raise ValueError(f"{path}:{line_no}: missing id")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if "seed" not in raw:
                raise ValueError(f"{path}:{line_no}: missing seed")
            if id_value in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id: {id_value}")
            seen.add(id_value)

            prompts.append(Prompt(id=id_value, prompt=prompt, seed=int(raw["seed"])))

    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


def resolve_prompts_path(prompts_arg: str) -> Path:
    raw = prompts_arg.strip()
    if not raw:
        raise ValueError("you must provide prompt collection")

    named_prompt_path = PROMPTS_DIR / f"{raw}.jsonl"
    if named_prompt_path.is_file():
        return named_prompt_path

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


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
        import torch
        from diffusers.utils import export_to_video
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        started_at = time.perf_counter()
        request: dict[str, object] = {"prompt": prompt.prompt}
        seed = secrets.randbits(63) if self.config.random_seed else prompt.seed
        if self.config.random_seed:
            print(f"using random seed {seed} for {prompt.id}", flush=True)

        torch_generator = torch.Generator(
            device=current_omni_platform.device_type
        ).manual_seed(seed)
        sampling_params = OmniDiffusionSamplingParams(
            height=self.config.height,
            width=self.config.width,
            generator=torch_generator,
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
        )


def wan_video_frames(output: object) -> list[object]:
    from vllm_omni.outputs import OmniRequestOutput

    result = OmniRequestOutput.unwrap_result(output)
    return coerce_video_frames(result.images[0])


def coerce_video_frames(video: object) -> list[object]:
    import numpy as np

    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None and isinstance(video, Image.Image):
        return [video]

    if isinstance(video, (list, tuple)):
        if len(video) == 1 and _ndim(video[0]) >= 4:
            return coerce_video_frames(video[0])
        frames: list[object] = []
        for frame in video:
            if _ndim(frame) >= 4:
                frames.extend(coerce_video_frames(frame))
            else:
                frames.append(_coerce_single_frame(frame))
        return frames

    array = _as_numpy(video)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        return [_prepare_frame_array(array)]
    if array.ndim != 4:
        raise ValueError(f"unsupported video output shape: {array.shape}")

    if array.shape[-1] in (1, 3, 4):
        frames = array
    elif array.shape[0] in (1, 3, 4):
        frames = np.moveaxis(array, 0, -1)
    elif array.shape[1] in (1, 3, 4):
        frames = np.moveaxis(array, 1, -1)
    else:
        raise ValueError(f"unsupported video output shape: {array.shape}")

    return [_prepare_frame_array(frame) for frame in frames]


def _coerce_single_frame(frame: object) -> object:
    try:
        from PIL import Image
    except Exception:
        Image = None

    if Image is not None and isinstance(frame, Image.Image):
        return frame

    return _prepare_frame_array(_as_numpy(frame))


def _prepare_frame_array(frame: object) -> object:
    import numpy as np

    array = _as_numpy(frame)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (
        1,
        3,
        4,
    ):
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 2 and not (array.ndim == 3 and array.shape[-1] in (1, 3, 4)):
        raise ValueError(f"unsupported frame output shape: {array.shape}")

    array = array.astype(np.float32, copy=False)
    if array.size and array.min() < 0:
        array = (array + 1.0) / 2.0
    elif array.size and array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _as_numpy(value: object) -> object:
    import numpy as np

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _ndim(value: object) -> int:
    if hasattr(value, "ndim"):
        return int(value.ndim)
    if hasattr(value, "dim"):
        return int(value.dim())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
