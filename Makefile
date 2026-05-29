.PHONY: all
all: install

IMAGE_NAME ?= ghcr.io/YOUR_ORG/video-inference-validation:latest
DOCKERFILE ?= Dockerfile.runpod

# Install all dependencies from the lockfile.
.PHONY: install
install:
	uv sync --locked

# Update dependencies and the lockfile.
.PHONY: update
update:
	uv sync

# Build the RunPod image.
.PHONY: build-image
build-image:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .
	@echo $(IMAGE_NAME)

# Push the RunPod image.
.PHONY: push-image
push-image: build-image
	docker push $(IMAGE_NAME)
