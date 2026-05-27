#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ghcr.io/YOUR_ORG/video-inference-validation:latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.runpod}"

docker build -f "$DOCKERFILE" -t "$IMAGE_NAME" .

if [[ "${PUSH:-0}" == "1" ]]; then
  docker push "$IMAGE_NAME"
fi

echo "$IMAGE_NAME"
