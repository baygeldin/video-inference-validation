from __future__ import annotations

import argparse
import sys
from pathlib import Path

from viv.config import load_inference_config
from viv.prompts import resolve_prompts_path
from viv.runner import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="viv",
        description="Generate Wan2.2 videos with vLLM-Omni.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        dest="prompts_arg",
        required=True,
        help="Prompt collection identifier, or path to the JSONL file",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="default",
        help="Inference config identifier",
    )
    parser.add_argument("output_dir", type=Path, help="Output folder path")
    args = parser.parse_args(argv)

    try:
        prompts_path = resolve_prompts_path(args.prompts_arg)
        config = load_inference_config(args.config)
        run(prompts_path, args.output_dir, args.config.strip(), config)
    except Exception as exc:
        print(f"viv: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
