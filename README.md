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

Use `--config`/`-c` to select an inference config from `configs.yml`:

```bash
viv -p pilot -c cfg_disabled /workspace/outputs
viv -p pilot --config random_seed /workspace/outputs
viv -p pilot --config steps_minus_1 /workspace/outputs
viv -p pilot --config steps_minus_10 /workspace/outputs
viv -p pilot --config tp4_parallelism /workspace/outputs
viv -p pilot --config cache_dit /workspace/outputs
viv -p pilot --config int4_quantization /workspace/outputs
```

The script runs offline inference through vLLM-Omni and writes:

```text
/workspace/outputs/<id>.mp4
/workspace/outputs/<id>.json
```

The JSON sidecar records the config name, UTC timestamp, generation duration,
prompt id/text, actual seed, video dimensions, fps, frame count, inference
steps, guidance scales, attention backend, model revision, MP4 export quality,
tensor parallel size, cache backend, model name, and runtime environment details: GPU
model, vLLM-Omni version, PyTorch version, CUDA version, NVIDIA driver version,
ffmpeg version, and Python version.

The default generation config is Wan2.2 T2V at `832x480`, `81` frames, `40` steps, `16` fps, CFG guidance `4.0`, HuggingFace revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7`, diffusion attention backend `FLASH_ATTN`, MP4 export quality `5.0`, `tensor_parallelism: 1`, and `cache_backend: null`.
Use the `tp4_parallelism` config for 4-way tensor parallelism. Use the `cache_dit` config (`cache_backend: cache_dit`) to enable Cache-DiT acceleration.
Use the `int4_quantization` config to run `Intel/Wan2.2-T2V-A14B-Diffusers-int4-AutoRound`.

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
