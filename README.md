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
viv generate -p pilot /workspace/outputs
```

Use `--config`/`-c` to select an inference config from `configs.yml`:

```bash
viv generate -p pilot -c cfg_disabled /workspace/outputs
viv generate -p pilot --config random_seed /workspace/outputs
viv generate -p pilot --config steps_minus_1 /workspace/outputs
viv generate -p pilot --config steps_minus_10 /workspace/outputs
viv generate -p pilot --config tp4_parallelism /workspace/outputs
viv generate -p pilot --config cache_dit /workspace/outputs
viv generate -p pilot --config int4_quantization /workspace/outputs
```

Latent capture is disabled by default. Use `--save-latents` to also write the initial noise latent, final noise latent, and each denoising step model prediction:

```bash
viv generate -p pilot --save-latents /workspace/outputs
```

Prompt embedding capture is also disabled by default. Add `--save-prompt-embeds` to write the encoded positive and negative prompt embedding tensors:

```bash
viv generate -p pilot --save-latents --save-prompt-embeds /workspace/outputs
```

Saved tensors can be reused by pointing at the folder that contains files named like `<id>.initial_noise_latent.safetensors`, `<id>.denoising_step_0.safetensors`, and `<id>.final_noise_latent.safetensors`:

```bash
viv generate -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-initial-latents /workspace/outputs

viv generate -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 /workspace/outputs

viv generate -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-prediction-latents /workspace/outputs

viv generate -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-final-latents /workspace/outputs
```

Saved prompt embeddings can be reused independently with `--reuse-prompt-embeds`, or combined with latent reuse to fully capture the inference state:

```bash
viv generate -p pilot --reuse-latents-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 \
  --reuse-prompt-embeds /workspace/outputs
```

If `--reuse-prediction-latents` is specified, then `--reuse-initial-latents` is enabled by default (because it makes no sense to reuse predictions without sharing the initial noise latent). With a `COUNT`, it uses the first `COUNT` saved model predictions and their original sigmas for denoising, then compresses the remaining saved sigma trajectory into the new run's remaining steps with equal spacing in UniPC lambda space. Without a `COUNT`, it reuses all saved prediction steps from the source generation.

By default, reused prediction steps skip model inference and load the saved predictions directly. Add `--save-original-predictions` to still run model inference for reused steps and save the fresh prediction tensors for comparison; those fresh tensors are discarded after saving and the previously saved predictions are used to update the latent state:

```bash
viv generate -p pilot \
  --save-latents \
  --save-original-predictions \
  --reuse-latents-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 \
  /workspace/outputs
```

## Output

The script runs offline inference through vLLM-Omni and writes:
```text
/workspace/outputs/<id>.mp4
/workspace/outputs/<id>.json
```

When enabled, tensor capture also writes files such as:
```text
/workspace/outputs/<id>.initial_noise_latent.safetensors
/workspace/outputs/<id>.denoising_step_N.safetensors
/workspace/outputs/<id>.final_noise_latent.safetensors
/workspace/outputs/<id>.prompt_embeds.safetensors
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
  "prompt_embeds_sha256": null,
  "negative_prompt_embeds_sha256": null,
  "sigma_schedule": [0.9999999403953552, ..., 0.11361567676067352],
  "initial_noise_latent_reused": false,
  "final_noise_latent_reused": false,
  "prediction_latents_reused": 0,
  "prompt_embeds_reused": false,
  "original_prediction_latents_saved": false,
  "reused_latents_from": null,
  "reused_prompt_embeds_from": null,
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
    "vllm_version": "0.23.0",
    "vllm_omni_version": "0.23.0rc1",
    "vllm_omni_commit": "7b837944a16ff440df0b19e71c6eca310d8dfc36",
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
- `original_prediction_latents_saved` records whether reused prediction steps ran fresh model inference and saved those fresh predictions

## Analysis

Saved latent runs can be compared with `viv compare`. The command writes a JSON report to the output path and includes only prompt IDs that are present in the baseline and every compared generation:

```bash
viv compare \
  --output /workspace/comparison.json \
  --baseline /workspace/baseline-outputs \
  /workspace/experiment-a \
  /workspace/experiment-b
```

Comparison reports RMSE and relative L2 error for each final latent tensor, mean/min/max RMSE and relative L2 error across matching denoising prediction latents, per-step prediction drift metrics, and mean/min/max per-frame SSIM for each generated video file.

```json
{
  "baseline": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "config_name": "default"
  },
  "generations": [
    {
      "gpu_model": "NVIDIA A100-SXM4-80GB",
      "config_name": "tp2_parallelism",
      "examples": [
        {
          "prompt_id": "action-binding-001",
          "video_file": {
            "mean_ssim": 0.95,
            "max_ssim": 0.98,
            "min_ssim": 0.91
          },
          "final_latent": {
            "rmse": 0.001,
            "relative_l2_error": 0.0001
          },
          "predictions": {
            "mean_rmse": 0.001,
            "min_rmse": 0.0,
            "max_rmse": 0.01,
            "mean_relative_l2_error": 0.0001,
            "min_relative_l2_error": 0.0,
            "max_relative_l2_error": 0.001,
            "steps": [
              {
                "step_idx": 0,
                "rmse": 0.0,
                "relative_l2_error": 0.0
              }
            ]
          }
        }
      ]
    }
  ]
}
```


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
