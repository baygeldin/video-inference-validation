#!/usr/bin/env bash
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-baygeldin/video-inference-validation}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.runpod}"
PLATFORM="${PLATFORM:-linux/amd64}"

IMAGE_TAG="${1:-}"

if [[ -z "$IMAGE_TAG" ]]; then
  IMAGE_TAG="$(git rev-parse --short HEAD 2>/dev/null || true)"
fi

IMAGE_NAME="${IMAGE_REPO}:${IMAGE_TAG}"

docker build --platform "$PLATFORM" -f "$DOCKERFILE" -t "$IMAGE_NAME" .
docker push "$IMAGE_NAME"

printf 'Built and pushed %s\n' "$IMAGE_NAME"
