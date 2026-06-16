# Video Inference Validation

This repository contains a harness for running inference validation experiments for video models, specifically Wan2.2-T2V-A14B. More details: https://github.com/gonka-ai/gonka/discussions/1155#discussioncomment-16976435

## Structure
- `prompts/` contains the built-in prompt datasets
- `configs.yml` defines the available inference configurations

The bundled prompt sets use prompts from the [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2/prompts) dataset, which covers diverse video generation settings that are challenging for current models.

## Usage
- Build and publish the `Dockerfile.runpod` image with `build-image.sh`
- Choose the desired GPU configuration on RunPod.io and deploy the image to the pod
- SSH into the container and run inference with the `viv` CLI

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

Latent capture is disabled by default. Use `--save-latents` to also write the initial noise latent, final noise latent, and each denoising step model prediction:

```bash
viv -p pilot --save-latents /workspace/outputs
```

Saved tensors can be reused by pointing at the folder that contains files named
like `<id>.initial_noise_latent.safetensors`,
`<id>.denoising_step_0.safetensors`, and
`<id>.final_noise_latent.safetensors`:

```bash
viv -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-initial-latents /workspace/outputs

viv -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 /workspace/outputs

viv -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-final-latents /workspace/outputs
```

If `--reuse-prediction-latents` is specified, then `--reuse-initial-latents` is enabled by default (because it makes no sense to reuse predictions without sharing the initial noise latent). It reuses the first `COUNT` saved model predictions and their original sigmas, then compresses the remaining saved sigma trajectory into the new run's remaining steps with equal spacing in UniPC lambda space.

The script runs offline inference through vLLM-Omni and writes:
```text
/workspace/outputs/<id>.mp4
/workspace/outputs/<id>.json
```

The JSON sidecar records the generation parameters and runtime environment:
```json
{
  "config_name": "default",
  "generation_id": "PrDYmJfK",
  "timestamp": "2026-06-02T11:36:05.916536+00:00",
  "duration_seconds": 415.33602340100333,
  "prompt_id": "pilot-0001",
  "prompt_text": "A blue car drives past a white picket fence on a sunny day",
  "seed": 420001,
  "initial_noise_latent_sha256": null,
  "final_noise_latent_sha256": null,
  "sigma_schedule": [0.9999999403953552, ..., 0.11361567676067352],
  "initial_noise_latent_reused": false,
  "final_noise_latent_reused": false,
  "prediction_latents_reused": 0,
  "reused_latents_from": null,
  "height": 480,
  "width": 832,
  "fps": 16,
  "num_frames": 81,
  "num_inference_steps": 40,
  "guidance_scale": 4.0,
  "guidance_scale_2": 4.0,
  "model_name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "attention_backend": "FLASH_ATTN",
  "model_revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
  "quality": 5.0,
  "tensor_parallelism": 1,
  "cache_backend": null,
  "environment": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "vllm_version": "0.22.0",
    "vllm_omni_version": "0.22.0rc2.dev13+gbc794e625",
    "vllm_omni_commit": "bc794e625f14ce425575210199bbb53f71cb860c",
    "pytorch_version": "2.11.0+cu130",
    "cuda_version": "13.0",
    "nvidia_driver_version": "580.159.03",
    "ffmpeg_version": "6.1.1-3ubuntu5",
    "python_version": "3.13.11 (main, Jan 28 2026, 00:01:45) [Clang 21.1.4 ]"
  }
}
```

Notes about some of the fields:
- `generation_id` is a short random identifier for the run
- `reused_latents_from` records the source generation's `generation_id`
- `sigma_schedule` records the actual sigma value used for each denoising step


## Syncing data between pods
- Generate a key pair for syncing files between pods on your host machine in the root of the repository via `ssh-keygen -t ed25519 -f runpod_sync -N ""`.
- Add the `runpod_sync` private key as `SYNC_PRIVATE_KEY` secret on RunPod.

Every pod with that environment variable can pull `/workspace/` from any other pod running the same image. From the destination pod, run:

```bash
runpod-sync <source-ssh-host> <source-ssh-port>
```

This copies the source pod's `/workspace/` into the current pod's `/workspace/`. Top-level hidden files and folders in `/workspace`, such as `.cache`, are ignored. Use `--dry-run` to preview changes, and add `--delete` only when the destination should exactly match the source:

```bash
runpod-sync <source-ssh-host> <source-ssh-port> --dry-run
runpod-sync <source-ssh-host> <source-ssh-port> --delete
```
