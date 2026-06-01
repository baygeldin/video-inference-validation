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
viv -p pilot /workspace/outputs
```

The script runs offline inference through vLLM-Omni and writes:

```text
/workspace/outputs/<id>.mp4
```

The current hardcoded generation defaults are Wan2.2 T2V at `832x480`, `81` frames, `40` steps, `16` fps, and CFG guidance `4.0`.

## RunPod Image
Build and push the image to Docker Hub from your host machine:
```bash
./build-image.sh
```

It will use the version from `pyproject.toml` to tag the image.

Then, inside the pod:
```bash
cd /opt/video-inference-validation
viv -p pilot /workspace/outputs
```
