"""Configuration loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Minimal experiment configuration shared by framework runners."""

    name: str
    alpha_values: tuple[float, ...] = (0.10,)
    methods: tuple[str, ...] = field(default_factory=tuple)
    paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Required {label} does not exist: {resolved}")
    return resolved


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a small JSON/YAML experiment config without extra dependencies."""

    config_path = require_path(path, "config file")
    data = _load_mapping(config_path)
    return ExperimentConfig(
        name=str(data["name"]),
        alpha_values=tuple(float(value) for value in data.get("alpha_values", [0.10])),
        methods=tuple(str(value) for value in data.get("methods", [])),
        paths={str(key): str(value) for key, value in data.get("paths", {}).items()},
        metadata=dict(data.get("metadata", {})),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the restricted YAML subset used by repository config files."""

    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip(" "))
            key, sep, value = line.strip().partition(":")
            if not sep:
                raise ValueError(f"Invalid config line in {path}: {raw_line.rstrip()}")

            if indent == 0:
                if value.strip() == "":
                    current_section = {}
                    root[key] = current_section
                else:
                    current_section = None
                    root[key] = _parse_scalar(value.strip())
            elif indent == 2 and current_section is not None:
                current_section[key] = _parse_scalar(value.strip())
            else:
                raise ValueError(f"Unsupported indentation in {path}: {raw_line.rstrip()}")

    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(marker in value for marker in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")
