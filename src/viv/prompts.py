from __future__ import annotations

import json
from pathlib import Path

from viv.models import Prompt
from viv.paths import PROMPTS_DIR


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


def resolve_prompts_path(prompts_arg: str) -> Path:
    raw = prompts_arg.strip()
    if not raw:
        raise ValueError("you must provide prompt collection")

    named_prompt_path = PROMPTS_DIR / f"{raw}.jsonl"
    if named_prompt_path.is_file():
        return named_prompt_path

    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()
