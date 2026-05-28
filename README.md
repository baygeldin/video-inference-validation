# Video Inference Validation Experiment

Dataset-generation harness for the Gonka video inference validation experiment:
https://github.com/gonka-ai/gonka/discussions/1155#discussioncomment-16976435

The harness is intentionally pragmatic. It builds deterministic job manifests, runs Wan2.2 text-to-video generations through vLLM-Omni, captures final videos plus selected latent checkpoints, and stores artifacts under a Runpod network volume mounted at `/workspace`.

## Local Checks

Install the local CLI:

```bash
make install
```

Validate the experiment config and pilot prompt file:

```bash
viv compile-plan --check \
  --config configs/experiment.yaml \
  --prompts configs/prompts.pilot.jsonl
```

Compile a small local manifest without running generation:

```bash
viv compile-plan \
  --config configs/experiment.yaml \
  --prompts configs/prompts.pilot.jsonl \
  --artifact-root /tmp/viv-runs \
  --run-id pilot-local \
  --prompt-limit 2
```

## Build The Runpod Image

Set the image name to a registry you can push to:

```bash
IMAGE_NAME=ghcr.io/YOUR_ORG/video-inference-validation:latest \
PUSH=1 \
scripts/runpod/build_image.sh
```

The Docker build installs vLLM-Omni from upstream and enables latent capture through `runtime_patches/sitecustomize.py`. No vLLM-Omni source files are edited in the image.

## Create A Runpod Pod

Create a Runpod network volume in the datacenter you plan to use. Attach it at `/workspace`.

Use `scripts/runpod/create_pod.json` as the API body template, or mirror the same settings in the Runpod console:

- custom image: the image you pushed above
- GPU Pod, not Serverless
- Secure Cloud preferred for core runs
- network volume mounted at `/workspace`
- H100/H200 for canonical H100 configs, A100 for A100 configs, `gpuCount: 4` for TP-4 configs

For API launch, replace the placeholders in `scripts/runpod/create_pod.json` first.

## Run The Pilot

Inside the Pod:

```bash
cd /opt/video-inference-validation
scripts/runpod/bootstrap_pod.sh
```

Run one config for up to 10 pilot prompts:

```bash
RUN_ID=pilot-001 \
CONFIG_ID=canonical_h100_bf16_tp1 \
PROMPT_LIMIT=10 \
scripts/runpod/run_generation.sh
```

For another config, run the same script with a different `CONFIG_ID`, using the same `RUN_ID` if you want all config outputs in one run directory.

## Inspect Artifacts

Artifacts are written under:

```text
/workspace/runs/<run_id>/
  manifest/jobs.jsonl
  artifacts/<config_id>/<prompt_id>/video.mp4
  artifacts/<config_id>/<prompt_id>/latents/initial.safetensors
  artifacts/<config_id>/<prompt_id>/latents/pre_boundary.safetensors
  artifacts/<config_id>/<prompt_id>/latents/final.safetensors
  artifacts/<config_id>/<prompt_id>/metadata.json
  artifacts/<config_id>/<prompt_id>/checksums.json
  logs/<config_id>/shard-*.jsonl
```

Inspect completion:

```bash
viv inspect-run --run-id pilot-001 --artifact-root /workspace/runs
```

## Export Or Sync

To print the Runpod S3-compatible sync command:

```bash
viv sync-artifacts \
  --run-id pilot-001 \
  --datacenter US-KS-2 \
  --network-volume-id YOUR_NETWORK_VOLUME_ID
```

Add `--execute` to run it. Configure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` with your Runpod S3 API credentials first.
