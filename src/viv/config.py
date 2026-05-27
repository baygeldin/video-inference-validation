from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    prompt: str
    seed: int


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return data


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
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
            prompt_id = str(raw.get("prompt_id") or "").strip()
            prompt = str(raw.get("prompt") or "").strip()
            if not prompt_id:
                raise ValueError(f"{path}:{line_no}: missing prompt_id")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: missing prompt")
            if "seed" not in raw:
                raise ValueError(f"{path}:{line_no}: missing seed")
            prompts.append(Prompt(prompt_id=prompt_id, prompt=prompt, seed=int(raw["seed"])))
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    ids = [p.prompt_id for p in prompts]
    duplicates = sorted({prompt_id for prompt_id in ids if ids.count(prompt_id) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate prompt_id values: {', '.join(duplicates)}")
    return prompts


def enabled_generation_configs(config: dict[str, Any], include_disabled: bool = False) -> list[dict[str, Any]]:
    configs = config.get("generation_configs")
    if not isinstance(configs, list) or not configs:
        raise ValueError("experiment config must define generation_configs")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in configs:
        if not isinstance(item, dict):
            raise ValueError("each generation config must be an object")
        config_id = str(item.get("id") or "").strip()
        if not config_id:
            raise ValueError("generation config missing id")
        if config_id in seen:
            raise ValueError(f"duplicate generation config id: {config_id}")
        seen.add(config_id)
        if include_disabled or item.get("enabled", True):
            result.append(item)
    return result


def get_generation_config(config: dict[str, Any], config_id: str) -> dict[str, Any]:
    for item in enabled_generation_configs(config, include_disabled=True):
        if item["id"] == config_id:
            return item
    raise ValueError(f"unknown generation config: {config_id}")


def merged_request(defaults: dict[str, Any], generation_config: dict[str, Any], prompt: Prompt) -> dict[str, Any]:
    request = dict(defaults)
    request.update(generation_config.get("request_overrides") or {})
    request["prompt"] = prompt.prompt
    seed_offset = int(generation_config.get("seed_offset", 0) or 0)
    request["seed"] = int(prompt.seed) + seed_offset
    return request
