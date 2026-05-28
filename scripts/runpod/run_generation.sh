#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/opt/video-inference-validation/configs/experiment.yaml}"
PROMPTS="${PROMPTS:-/opt/video-inference-validation/configs/prompts.pilot.jsonl}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/runs}"
RUN_ID="${RUN_ID:-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
CONFIG_ID="${CONFIG_ID:-canonical_h100_bf16_tp1}"
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

viv generate \
  --jobs "$JOBS" \
  --config-id "$CONFIG_ID" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT"

viv inspect-run --run-id "$RUN_ID" --artifact-root "$ARTIFACT_ROOT"
