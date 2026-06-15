"""Standalone prepared-system validation."""

from __future__ import annotations

from pathlib import Path

from mdprep.validation.parmed_check import run_parmed_check


def validate_system(prmtop: str | Path, inpcrd: str | Path) -> dict[str, object]:
    topology = Path(prmtop)
    coordinates = Path(inpcrd)
    errors: list[str] = []
    for label, path in [("prmtop", topology), ("inpcrd", coordinates)]:
        if not path.exists():
            errors.append(f"Missing {label}: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"Empty {label}: {path}")
    parmed = None
    if not errors:
        parmed = run_parmed_check(topology, coordinates)
        if parmed.get("status") == "error":
            errors.append(f"ParmEd validation failed: {parmed.get('error')}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "prmtop": str(topology),
        "inpcrd": str(coordinates),
        "parmed": parmed,
    }
