# Video Inference Validation

This repository contains a harness for running inference validation experiments for video models, specifically Wan2.2-T2V-A14B. More details: https://github.com/gonka-ai/gonka/discussions/1155#discussioncomment-16976435

## Structure
- `configs.yml` defines all available inference configurations
- `prompts/` contains the built-in prompt datasets
- `notes/` contains experiment reports

The bundled prompt sets use prompts from the [T2V-CompBench](https://github.com/KaiyueSun98/T2V-CompBench/tree/V2/prompts) dataset, which covers diverse video generation settings that are challenging for current models.

## Usage
- Build and publish the `Dockerfile.runpod` image with `build-image.sh`
- Choose the desired GPU configuration on RunPod.io and deploy the image to the pod
- SSH into the container and run inference with the `viv` CLI

## CLI
Use `--prompts`/`-p` to select the prompt collection from `prompts/`:
```bash
viv generate -p pilot /workspace/outputs
```

Use `--config`/`-c` to select an inference config from `configs.yml`:

```bash
viv generate -p pilot -c cfg_disabled /workspace/outputs
viv generate -p pilot -c random_seed /workspace/outputs
viv generate -p pilot -c steps_minus_1 /workspace/outputs
viv generate -p pilot -c steps_minus_10 /workspace/outputs
viv generate -p pilot -c tp2_parallelism /workspace/outputs
viv generate -p pilot -c tp4_parallelism /workspace/outputs
viv generate -p pilot -c cache_dit /workspace/outputs
viv generate -p pilot -c int4_quantization /workspace/outputs
```

Tensor capture is disabled by default. Use the explicit save flags to write the artifacts you need:

```bash
viv generate -p pilot \
  --save-prompt-embeds \
  --save-initial-latents \
  --save-prediction-latents \
  --save-final-latents \
  /workspace/outputs
```

Or use `--save-all` to write all inference artifacts (initial noise latent, final noise latent, each denoising step model prediction, and prompt embeddings):

```bash
viv generate -p pilot --save-all /workspace/outputs
```

Saved tensors can be reused by pointing at the folder that contains artifacts from a previous generation:

```bash
viv generate -p pilot \
  --reuse-prompt-embeds \
  --reuse-initial-latents \
  --reuse-prediction-latents \
  --reuse-final-latents \
  --reuse-from /workspace/previous-outputs \
  /workspace/outputs
```

If `--reuse-prediction-latents` is specified, then `--reuse-initial-latents` is enabled by default (because it makes no sense to reuse predictions without sharing the initial noise latent). With a `COUNT`, it uses the first `COUNT` saved model predictions and their original sigmas for denoising, then compresses the remaining saved sigma trajectory into the new run's remaining steps with equal spacing in UniPC lambda space. Without a `COUNT`, it reuses all saved prediction steps from the source generation.

By default, reused artifacts still run their corresponding computation so save flags capture fresh tensors for later comparison. Fresh initial latents, prompt embeddings, prediction tensors, and final latents are discarded after saving and calculating their checksums, and their reused counterparts are selected for further generation instead. Add `--skip-reused-computation` skip replaying computation for reused parts of the generation pipeline:

```bash
viv generate -p pilot \
  --save-prediction-latents \
  --reuse-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 \
  /workspace/outputs

viv generate -p pilot \
  --save-prediction-latents \
  --reuse-from /workspace/previous-outputs \
  --reuse-prediction-latents 10 \
  --skip-reused-computation \
  /workspace/outputs
```

When `--reuse-final-latents` is specified without `--skip-reused-computation`, generation still runs normally and writes any requested artifacts. After denoising and capture, the saved final latent from `--reuse-from` is used for video decoding. With `--skip-reused-computation`, denoising is skipped for final-latent reuse and the saved final latent is decoded directly.

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
```jsonc
{
  "generation_id": "PrDYmJfK", // Short random identifier for the run
  "timestamp": "2026-06-02T11:36:05.916536+00:00",
  "duration_seconds": 415.33602340100333,
  "config_name": "default",
  "prompt_id": "pilot-0001",
  "prompt_text": "A blue car drives past a white picket fence on a sunny day",
  "seed": 420001,
  "reuse": {
    "reused_from": null, // `generation_id` of the source generation
    "prompt_embeds_reused": false,
    "initial_noise_latent_reused": false,
    "prediction_latents_reused": 0,
    "final_noise_latent_reused": false,
    "skipped_reused_computation": false
  },
  "parameters": {
    "model_name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "model_revision": "5be7df9619b54f4e2667b2755bc6a756675b5cd7",
    "attention_backend": "FLASH_ATTN",
    "cache_backend": null,
    "tensor_parallelism": 1,
    "width": 832,
    "height": 480,
    "fps": 16,
    "num_frames": 81,
    "num_inference_steps": 40,
    "quality": 5.0,
    "boundary_ratio": 0.875,
    "flow_shift": 5.0,
    "guidance_scale": 4.0,
    "guidance_scale_2": 4.0,
    "sigma_schedule": [
      0.9999599456787109,
      0.9948573112487793,
      0.9895420670509338,
      0.9840006232261658,
      0.9782182574272156,
      0.9721789360046387,
      0.9658651351928711,
      0.9592576622962952,
      0.95233553647995,
      0.9450758099555969,
      0.9374530911445618,
      0.9294394850730896,
      0.9210041165351868,
      0.9121127724647522,
      0.9027275443077087,
      0.8928060531616211,
      0.8823009729385376,
      0.8711592555046082,
      0.8593212366104126,
      0.84671950340271,
      0.833277702331543,
      0.8189089894294739,
      0.8035139441490173,
      0.7869786620140076,
      0.7691715359687805,
      0.7499399185180664,
      0.7291058301925659,
      0.7064602375030518,
      0.6817561388015747,
      0.654699444770813,
      0.6249374151229858,
      0.592042863368988,
      0.5554937720298767,
      0.5146452784538269,
      0.468691349029541,
      0.41661104559898376,
      0.3570917844772339,
      0.28841710090637207,
      0.20829857885837555,
      0.11361567676067352
    ] // Actual sigma values used for each denoising step
  },
  "environment": {
    "gpu_model": "NVIDIA H100 80GB HBM3",
    "cuda_version": "13.0",
    "nvidia_driver_version": "580.159.03",
    "vllm_version": "0.23.0",
    "vllm_omni_version": "0.23.0rc1",
    "vllm_omni_commit": "7b837944a16ff440df0b19e71c6eca310d8dfc36",
    "pytorch_version": "2.11.0+cu130",
    "ffmpeg_version": "6.1.1-3ubuntu5",
    "python_version": "3.13.11 (main, Jan 28 2026, 00:01:45) [Clang 21.1.4 ]",
    "image_name": "viv:runpod-cu130"
  },
  "checksums": { // Checksums of this run's saved artifacts
    "prompt_embeds_sha256": "240c0423e4fadd092fb5ea2f9f7b447ccf68b8100e3d9bf85c137d324374bd1f",
    "negative_prompt_embeds_sha256": "16519678367ef0a3863dc0eff20945e106a8e812b425054e709f3062009d9a81",
    "initial_noise_latent_sha256": "980eaf0a67d5d9de2c386cebe874d0f8dbe05103d7b76fdf49e59d02185f7807",
    "final_noise_latent_sha256": "3d5af30e660bbdb1dfd4b04052b6f94ea7b21792f8ce89112ae95d4552584460"
  }
}
```

## Analysis

Saved latent runs can be compared with `viv compare`. The command writes a JSON report to the output path and includes only prompt IDs that are present in the baseline and every compared generation:

```bash
viv compare \
  --output /workspace/comparison.json \
  --baseline /workspace/baseline-outputs \
  --final-latents \
  --video-files \
  --predictions \
  /workspace/experiment-a \
  /workspace/experiment-b
```

Add one or more comparison flags to choose what the report contains:
- `--final-latents` reports RMSE, relative L2 error, and cosine similarity for each final latent tensor.
- `--video-files` reports mean/min/max per-frame SSIM for each generated video file.
- `--predictions` reports sigma, relative L2 error, and cosine similarity for each matching denoising prediction latent.

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
            "relative_l2_error": 0.0001,
            "cosine_similarity": 0.9999
          },
          "predictions": [
            {
              "step_idx": 0,
              "sigma": 0.9999,
              "relative_l2_error": 0.0,
              "cosine_similarity": 1.0
            }
          ]
        }
      ]
    }
  ]
}
```


## Syncing data between pods
To reuse inference artifacts generated on a different pod there is a helper utility to easily sync generation results between pods: To set it up, follow these steps:
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
