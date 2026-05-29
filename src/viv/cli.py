from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
WIDTH = 832
HEIGHT = 480
NUM_FRAMES = 81
FPS = 16
NUM_INFERENCE_STEPS = 40
GUIDANCE_SCALE = 4.0
GUIDANCE_SCALE_2 = 4.0


@dataclass(frozen=True)
class Prompt:
    id: str
    prompt: str
    seed: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv",
        description="Generate Wan2.2 videos from a JSONL prompt file with vLLM-Omni.",
    )
    parser.add_argument("prompts", type=Path, help="Path to JSONL file with prompts")
    parser.add_argument("output_dir", type=Path, help="Output folder path")
    args = parser.parse_args(argv)

    try:
        run(args.prompts, args.output_dir)
    except Exception as exc:
        print(f"viv: error: {exc}", file=sys.stderr)
        return 2
    return 0


def run(prompts_path: Path, output_dir: Path) -> None:
    prompts = load_prompts(prompts_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = OfflineVideoGenerator()
    for prompt in prompts:
        video_path = output_dir / f"{prompt.id}.mp4"
        print(f"generating {prompt.id} -> {video_path}", flush=True)
        generator.generate(prompt, video_path)
        print(f"completed {prompt.id}", flush=True)


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_no}: prompt row must be an object")

            id_value = str(raw.get("id") or "").strip()
            prompt = str(raw.get("prompt") or "").strip()
            if not id_value:
                raise ValueError(f"{path}:{line_no}: missing id")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if "seed" not in raw:
                raise ValueError(f"{path}:{line_no}: missing seed")
            if id_value in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id: {id_value}")
            seen.add(id_value)

            prompts.append(Prompt(id=id_value, prompt=prompt, seed=int(raw["seed"])))

    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


class OfflineVideoGenerator:
    def __init__(self) -> None:
        from vllm_omni.diffusion.data import DiffusionParallelConfig
        from vllm_omni.entrypoints.omni import Omni

        parallel_config = DiffusionParallelConfig(tensor_parallel_size=1)
        self.omni = Omni(model=MODEL, parallel_config=parallel_config)

    def generate(self, prompt: Prompt, video_path: Path) -> None:
        import torch
        from diffusers.utils import export_to_video
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        from vllm_omni.platforms import current_omni_platform

        request: dict[str, object] = {"prompt": prompt.prompt}

        torch_generator = torch.Generator(
            device=current_omni_platform.device_type
        ).manual_seed(prompt.seed)
        sampling_params = OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            generator=torch_generator,
            guidance_scale=GUIDANCE_SCALE,
            guidance_scale_2=GUIDANCE_SCALE_2,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
        )

        output = self.omni.generate(request, sampling_params)
        tmp_path = video_path.with_suffix(".tmp.mp4")
        export_to_video(wan_video_frames(output), str(tmp_path), fps=FPS)
        tmp_path.replace(video_path)


def wan_video_frames(output: object) -> list[object]:
    from vllm_omni.outputs import OmniRequestOutput

    result = OmniRequestOutput.unwrap_result(output)
    return list(result.images[0])


if __name__ == "__main__":
    raise SystemExit(main())
