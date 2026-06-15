"""Optional ParmEd validation for final Amber files."""

from __future__ import annotations

from pathlib import Path


def parmed_available() -> bool:
    try:
        import parmed  # noqa: F401
    except Exception:
        return False
    return True


def parmed_version() -> str:
    try:
        import parmed
    except Exception:
        return "not available"
    return getattr(parmed, "__version__", "unknown")


def run_parmed_check(prmtop: str | Path, inpcrd: str | Path) -> dict[str, object]:
    try:
        import parmed
    except Exception as exc:
        return {"available": False, "status": "skipped", "warning": f"ParmEd is unavailable: {exc}"}
    try:
        structure = parmed.load_file(str(prmtop), str(inpcrd))
        total_charge = sum(atom.charge for atom in structure.atoms)
        coordinates = getattr(structure, "coordinates", None)
        return {
            "available": True,
            "status": "ok",
            "atom_count": len(structure.atoms),
            "coordinate_count": len(coordinates) if coordinates is not None else 0,
            "total_charge": total_charge,
            "charge_near_integer": abs(total_charge - round(total_charge)) <= 0.05,
            "version": getattr(parmed, "__version__", "unknown"),
        }
    except Exception as exc:
        return {"available": True, "status": "error", "error": str(exc)}
