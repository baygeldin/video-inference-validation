#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "bootstrap failed: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
nvidia-smi >/dev/null || fail "nvidia-smi failed"

[[ -d /workspace ]] || fail "/workspace does not exist"
touch /workspace/.viv-write-test || fail "/workspace is not writable"
rm -f /workspace/.viv-write-test

mkdir -p /workspace/hf-cache /workspace/models /workspace/runs /workspace/logs

if [[ -z "${HF_TOKEN:-}" ]]; then
  warn "HF_TOKEN is not set; public models may still work, but downloads can be rate-limited"
fi

python3 --version
python3 -m pip show video-inference-validation >/dev/null || fail "video-inference-validation package is not installed"
command -v vllm >/dev/null || fail "vllm command not found"
command -v ffmpeg >/dev/null || fail "ffmpeg command not found"

echo "GPU:"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
echo
echo "workspace:"
df -h /workspace
echo
echo "viv:"
viv compile-plan --check --config /opt/video-inference-validation/configs/experiment.yaml --prompts /opt/video-inference-validation/configs/prompts.pilot.jsonl
