from __future__ import annotations

import argparse
import sys
from pathlib import Path

from viv.config import load_inference_config
from viv.models import LatentReuseConfig
from viv.prompts import resolve_prompts_path
from viv.runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv",
        description="Generate Wan2.2 videos with vLLM-Omni.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "-p",
        "--prompts",
        dest="prompts",
        required=True,
        help="Prompt collection identifier, or path to the JSONL file",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="default",
        help="Inference config identifier",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help=(
            "Save initial, final, and denoising step latent tensors "
            "(disabled by default)"
        ),
    )
    parser.add_argument(
        "--reuse-latents-from",
        dest="reuse_latents_from",
        type=Path,
        help="Folder containing saved latent tensors to reuse",
    )
    parser.add_argument(
        "--reuse-initial-latents",
        dest="reuse_initial_latent",
        action="store_true",
        help="Reuse the saved initial latent instead of generating initial noise",
    )
    parser.add_argument(
        "--reuse-prediction-latents",
        dest="reuse_predictions",
        metavar="COUNT",
        type=int,
        help="Reuse the first COUNT saved denoising predictions for each prompt",
    )
    parser.add_argument(
        "--reuse-final-latents",
        dest="reuse_final_latent",
        action="store_true",
        help="Decode the saved final latent directly, skipping prompt encoding and denoising",
    )
    parser.add_argument("output_dir", type=Path, help="Output folder path")
    args = parser.parse_args(argv)
    reuse_requested = (
        args.reuse_initial_latent
        or args.reuse_predictions is not None
        or args.reuse_final_latent
    )
    if reuse_requested and args.reuse_latents_from is None:
        parser.error("--reuse-latents-from is required when reusing latents")
    if args.reuse_final_latent and (
        args.reuse_initial_latent or args.reuse_predictions is not None
    ):
        parser.error("--reuse-final-latents cannot be combined with other reuse modes")
    if args.reuse_predictions is not None and args.reuse_predictions < 0:
        parser.error("--reuse-prediction-latents must be a non-negative integer")

    latent_reuse = None
    if args.reuse_latents_from is not None:
        latent_reuse = LatentReuseConfig(
            source_dir=args.reuse_latents_from,
            reuse_initial_latent=args.reuse_initial_latent,
            reuse_predictions=args.reuse_predictions,
            reuse_final_latent=args.reuse_final_latent,
        )

    try:
        prompts_path = resolve_prompts_path(args.prompts)
        config = load_inference_config(args.config)
        run(
            prompts_path,
            args.output_dir,
            args.config.strip(),
            config,
            save_latents=args.save_latents,
            latent_reuse=latent_reuse,
        )
    except Exception as exc:
        print(f"viv: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
