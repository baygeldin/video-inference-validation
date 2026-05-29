#!/usr/bin/env bash
set -euo pipefail

IMAGE_REPO="${IMAGE_REPO:-baygeldin/video-inference-validation}"
PYPROJECT_FILE="${PYPROJECT_FILE:-pyproject.toml}"
VERSION="${VERSION:-$(awk -F '"' '/^version = "/ {print $2; exit}' "$PYPROJECT_FILE")}"
IMAGE_NAME="${IMAGE_NAME:-${IMAGE_REPO}:${VERSION}}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.runpod}"
PLATFORM="${PLATFORM:-linux/amd64}"

if [[ -z "$VERSION" ]]; then
  printf 'Could not determine version from %s\n' "$PYPROJECT_FILE" >&2
  exit 1
fi

docker build --platform "$PLATFORM" -f "$DOCKERFILE" -t "$IMAGE_NAME" .
docker push "$IMAGE_NAME"

printf 'Built and pushed %s\n' "$IMAGE_NAME"
