#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/opt/video-inference-validation/configs/experiment.yaml}"
PROMPTS="${PROMPTS:-/opt/video-inference-validation/configs/prompts.pilot.jsonl}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/runs}"
RUN_ID="${RUN_ID:-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_ID="${CONFIG_ID:-canonical_h100_bf16_tp1}"
PORT="${PORT:-8091}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
PROMPT_LIMIT="${PROMPT_LIMIT:-10}"

JOBS="$ARTIFACT_ROOT/$RUN_ID/manifest/jobs.jsonl"

viv compile-plan \
  --config "$CONFIG" \
  --prompts "$PROMPTS" \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-id "$RUN_ID" \
  --prompt-limit "$PROMPT_LIMIT"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

viv serve \
  --config "$CONFIG" \
  --config-id "$CONFIG_ID" \
  --run-id "$RUN_ID" \
  --artifact-root "$ARTIFACT_ROOT" \
  --port "$PORT" &
SERVER_PID=$!

echo "waiting for vLLM-Omni server on port $PORT"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null

viv generate \
  --jobs "$JOBS" \
  --config-id "$CONFIG_ID" \
  --server-url "http://127.0.0.1:$PORT" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

viv inspect-run --run-id "$RUN_ID" --artifact-root "$ARTIFACT_ROOT"
