# Video Inference Validation

Tiny vLLM-Omni text-to-video runner for the first experiment pass.

## Prompt File

Use JSONL with one prompt per line:

```jsonl
{"id":"pilot-0001","seed":420001,"prompt":"A blue car drives past a white picket fence on a sunny day"}
{"id":"pilot-0002","seed":420002,"prompt":"A cat slinking to the left side of a cozy living room"}
```

## Run

```bash
viv configs/prompts.pilot.jsonl /workspace/outputs
```

The script runs offline inference through vLLM-Omni and writes:

```text
/workspace/outputs/<id>.mp4
```

The current hardcoded generation defaults are Wan2.2 T2V at `832x480`, `81` frames, `40` steps, `16` fps, and CFG guidance `4.0`.

## RunPod Image

```bash
IMAGE_NAME=ghcr.io/YOUR_ORG/video-inference-validation:latest \
PUSH=1 \
scripts/runpod/build_image.sh
```

Inside the pod:

```bash
cd /opt/video-inference-validation
viv configs/prompts.pilot.jsonl /workspace/outputs
```
