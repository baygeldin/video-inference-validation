#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-baygeldin/video-inference-validation:latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.runpod}"
PLATFORM="${PLATFORM:-linux/amd64}"

docker build --platform "$PLATFORM" -f "$DOCKERFILE" -t "$IMAGE_NAME" .
docker push "$IMAGE_NAME"

printf 'Built and pushed %s\n' "$IMAGE_NAME"
