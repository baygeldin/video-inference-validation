from __future__ import annotations

from pathlib import Path

from viv.environment import collect_environment_metadata
from viv.generator import OfflineVideoGenerator
from viv.metadata import write_sidecar_metadata
from viv.models import InferenceConfig, LatentReuseConfig
from viv.prompts import load_prompts


def run(
    prompts_path: Path,
    output_dir: Path,
    config_name: str,
    config: InferenceConfig,
    save_latents: bool = False,
    latent_reuse: LatentReuseConfig | None = None,
) -> None:
    prompts = load_prompts(prompts_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "using "
        f"{config.model_name}@{config.model_revision}, "
        f"attention={config.attention_backend}, "
        f"export_quality={config.export_quality}, "
        f"tp={config.tensor_parallelism}, "
        f"cache_backend={config.cache_backend}",
        flush=True,
    )
    generator = OfflineVideoGenerator(
        config, save_latents=save_latents, latent_reuse=latent_reuse
    )
    environment = collect_environment_metadata()
    for prompt in prompts:
        video_path = output_dir / f"{prompt.id}.mp4"
        metadata_path = output_dir / f"{prompt.id}.json"
        print(f"generating {prompt.id} -> {video_path}", flush=True)
        result = generator.generate(prompt, video_path)
        write_sidecar_metadata(
            metadata_path,
            config_name,
            prompt,
            config,
            result,
            environment,
        )
        print(f"completed {prompt.id}", flush=True)
