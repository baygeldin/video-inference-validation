from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path


def collect_environment_metadata() -> dict[str, str | None]:
    nvidia_smi = _nvidia_smi_metadata()
    return {
        "gpu_model": _gpu_model()
        or nvidia_smi.get("gpu_model")
        or _gpu_model_from_proc(),
        "vllm_version": _package_version("vllm"),
        "vllm_omni_version": _package_version("vllm-omni"),
        "vllm_omni_commit": _package_direct_url_commit("vllm-omni"),
        "pytorch_version": _pytorch_version(),
        "cuda_version": _cuda_version() or nvidia_smi.get("cuda_version"),
        "nvidia_driver_version": nvidia_smi.get("nvidia_driver_version")
        or _nvidia_driver_version_from_proc(),
        "ffmpeg_version": _ffmpeg_version(),
        "python_version": sys.version,
        "image_name": _env_value("VIV_IMAGE_NAME"),
    }


def _package_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip() or None


def _package_direct_url_commit(package_name: str) -> str | None:
    try:
        distribution = importlib_metadata.distribution(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None

    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None

    try:
        direct_url_data = json.loads(direct_url)
    except json.JSONDecodeError:
        return None

    vcs_info = direct_url_data.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    return commit_id if isinstance(commit_id, str) and commit_id else None


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
