from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from viv.compare import compare_generations
from viv.config import load_inference_config
from viv.models import LatentReuseConfig
from viv.prompts import resolve_prompts_path
from viv.runner import run

_REUSE_ALL_PREDICTIONS_MARKER = "__viv_reuse_all_predictions__"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv",
        description="Generate Wan2.2 videos and compare saved generation artifacts.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate",
        description="Generate Wan2.2 videos with vLLM-Omni.",
        help="Generate Wan2.2 videos with vLLM-Omni.",
        allow_abbrev=False,
    )
    _add_generate_arguments(generate_parser)
    generate_parser.set_defaults(handler=_run_generate, parser=generate_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        description="Compare saved generation artifacts against a baseline.",
        help="Compare saved generation artifacts against a baseline.",
        allow_abbrev=False,
    )
    _add_compare_arguments(compare_parser)
    compare_parser.set_defaults(handler=_run_compare)

    args = parser.parse_args(_normalize_optional_reuse_prediction_count(argv))
    return args.handler(args)


def _add_generate_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--save-initial-latents",
        action="store_true",
        help="Save initial noise latent tensors (disabled by default)",
    )
    parser.add_argument(
        "--save-final-latents",
        action="store_true",
        help="Save final noise latent tensors (disabled by default)",
    )
    parser.add_argument(
        "--save-prediction-latents",
        action="store_true",
        help="Save denoising step prediction latent tensors (disabled by default)",
    )
    parser.add_argument(
        "--save-prompt-embeds",
        action="store_true",
        help="Save encoded prompt embedding tensors (disabled by default)",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help=(
            "Save initial, final, prediction, and prompt embedding tensors "
            "(disabled by default)"
        ),
    )
    parser.add_argument(
        "--reuse-from",
        dest="reuse_from",
        type=Path,
        help="Folder containing saved tensors to reuse",
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
        nargs="?",
        const=_REUSE_ALL_PREDICTIONS_MARKER,
        type=_parse_reuse_prediction_count,
        help=(
            "Reuse saved denoising predictions for each prompt. If COUNT is "
            "omitted, reuse all saved prediction steps from the source generation"
        ),
    )
    parser.add_argument(
        "--skip-reused-computation",
        action="store_true",
        help=(
            "When reusing tensors, load reused artifacts directly instead of "
            "rerunning the corresponding computation"
        ),
    )
    parser.add_argument(
        "--reuse-prompt-embeds",
        action="store_true",
        help="Reuse saved prompt embedding tensors instead of encoding the prompt",
    )
    parser.add_argument(
        "--reuse-final-latents",
        dest="reuse_final_latent",
        action="store_true",
        help=(
            "After denoising and artifact capture, decode the saved final "
            "latent from --reuse-from"
        ),
    )
    parser.add_argument("output_dir", type=Path, help="Output folder path")


def _run_generate(args: argparse.Namespace) -> int:
    parser = args.parser
    if args.save_all:
        args.save_initial_latents = True
        args.save_final_latents = True
        args.save_prediction_latents = True
        args.save_prompt_embeds = True

    reuse_all_predictions = (
        args.reuse_predictions == _REUSE_ALL_PREDICTIONS_MARKER
    )
    reuse_predictions_requested = (
        reuse_all_predictions or args.reuse_predictions is not None
    )
    reuse_requested = (
        args.reuse_initial_latent
        or reuse_predictions_requested
        or args.reuse_prompt_embeds
        or args.reuse_final_latent
    )
    if reuse_requested and args.reuse_from is None:
        parser.error("--reuse-from is required when reusing saved tensors")
    if isinstance(args.reuse_predictions, int) and args.reuse_predictions < 0:
        parser.error("--reuse-prediction-latents must be a non-negative integer")
    if args.skip_reused_computation and not reuse_requested:
        parser.error("--skip-reused-computation requires at least one reuse flag")

    latent_reuse = None
    if args.reuse_from is not None:
        latent_reuse = LatentReuseConfig(
            source_dir=args.reuse_from,
            reuse_initial_latent=args.reuse_initial_latent,
            reuse_predictions=(
                None if reuse_all_predictions else args.reuse_predictions
            ),
            reuse_all_predictions=reuse_all_predictions,
            skip_reused_computation=args.skip_reused_computation,
            reuse_final_latent=args.reuse_final_latent,
            reuse_prompt_embeds=args.reuse_prompt_embeds,
        )

    try:
        prompts_path = resolve_prompts_path(args.prompts)
        config = load_inference_config(args.config)
        run(
            prompts_path,
            args.output_dir,
            args.config.strip(),
            config,
            save_initial_latents=args.save_initial_latents,
            save_final_latents=args.save_final_latents,
            save_prediction_latents=args.save_prediction_latents,
            save_prompt_embeds=args.save_prompt_embeds,
            latent_reuse=latent_reuse,
        )
    except Exception as exc:
        print(f"viv generate: error: {exc}", file=sys.stderr)
        return 2
    return 0


def _normalize_optional_reuse_prediction_count(
    argv: list[str] | None,
) -> list[str] | None:
    if argv is None:
        argv = sys.argv[1:]

    normalized: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        normalized.append(token)
        if token == "--reuse-prediction-latents":
            next_idx = idx + 1
            if next_idx >= len(argv) or not _looks_like_int(argv[next_idx]):
                normalized.append(_REUSE_ALL_PREDICTIONS_MARKER)
        idx += 1
    return normalized


def _parse_reuse_prediction_count(value: str) -> int | str:
    if value == _REUSE_ALL_PREDICTIONS_MARKER:
        return _REUSE_ALL_PREDICTIONS_MARKER
    try:
        return int(value)
    except ValueError:
        _raise_argparse_error(
            f"--reuse-prediction-latents expected an integer count, got {value!r}"
        )


def _looks_like_int(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return True


def _raise_argparse_error(message: str) -> NoReturn:
    raise argparse.ArgumentTypeError(message)


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        required=True,
        help="JSON file path where comparison results should be written",
    )
    parser.add_argument(
        "--baseline",
        dest="baseline_dir",
        type=Path,
        required=True,
        help="Generation folder to use as the comparison baseline",
    )
    parser.add_argument(
        "generation_dirs",
        nargs="+",
        type=Path,
        help="Generation folders to compare with the baseline",
    )


def _run_compare(args: argparse.Namespace) -> int:
    try:
        output_path = compare_generations(
            args.output_path,
            args.baseline_dir,
            args.generation_dirs,
        )
    except Exception as exc:
        print(f"viv compare: error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
