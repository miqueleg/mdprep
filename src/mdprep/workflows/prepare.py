"""Preparation workflow placeholder."""

from __future__ import annotations

from pathlib import Path


def prepare_system(manifest_path: str | Path) -> None:
    raise NotImplementedError(
        f"mdprep prepare is not implemented yet for {manifest_path}; Task 1 only bootstraps the CLI and config validation."
    )

