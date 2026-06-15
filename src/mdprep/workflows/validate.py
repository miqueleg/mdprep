"""Prepared-system validation placeholder."""

from __future__ import annotations

from pathlib import Path


def validate_system(prmtop: str | Path, inpcrd: str | Path) -> None:
    raise NotImplementedError(
        f"mdprep validate is not implemented yet for {prmtop} and {inpcrd}; Task 1 only bootstraps the CLI and config validation."
    )

