"""YAML manifest loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from mdprep.config.models import ManifestConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a YAML mapping at the top level")
    return data


def load_manifest(path: str | Path) -> ManifestConfig:
    return ManifestConfig.model_validate(load_yaml(path))

