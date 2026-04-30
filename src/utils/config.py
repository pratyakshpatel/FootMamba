from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a dictionary.")
    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    default_path = config_path.parent / "default.yaml"

    if config_path.name == "default.yaml":
        return load_yaml(config_path)

    if default_path.exists():
        base = load_yaml(default_path)
        override = load_yaml(config_path)
        return _deep_merge(base, override)
    return load_yaml(config_path)


def resolve_path(path_value: str | Path, root: str | Path = ".") -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(root) / path

