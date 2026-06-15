"""Optional OpenMM finite-energy validation."""

from __future__ import annotations

from math import isfinite
from pathlib import Path


def openmm_available() -> bool:
    try:
        import openmm  # noqa: F401
        from openmm import app  # noqa: F401
    except Exception:
        return False
    return True


def openmm_version() -> str:
    try:
        import openmm
    except Exception:
        return "not available"
    return getattr(openmm, "__version__", "unknown")


def run_openmm_energy_check(prmtop: str | Path, inpcrd: str | Path) -> dict[str, object]:
    try:
        import openmm
        from openmm import app, unit
    except Exception as exc:
        return {"available": False, "status": "skipped", "warning": f"OpenMM is unavailable: {exc}"}
    try:
        topology = app.AmberPrmtopFile(str(prmtop))
        coordinates = app.AmberInpcrdFile(str(inpcrd))
        system = topology.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = openmm.Context(system, integrator)
        context.setPositions(coordinates.positions)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        del context
        del integrator
        return {
            "available": True,
            "status": "ok" if isfinite(energy) else "error",
            "potential_energy_kcal_mol": energy,
            "finite": isfinite(energy),
            "version": getattr(openmm, "__version__", "unknown"),
        }
    except Exception as exc:
        return {"available": True, "status": "error", "error": str(exc)}
