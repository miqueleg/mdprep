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
        force_classes = [force.__class__.__name__ for force in system.getForces()]
        custom_nonbonded_expressions = [
            force.getEnergyFunction()
            for force in system.getForces()
            if isinstance(force, openmm.CustomNonbondedForce)
        ]
        amber_12_6_4 = _prmtop_has_flag(prmtop, "LENNARD_JONES_CCOEF")
        openmm_12_6_4 = any(
            "r^4" in expression.replace(" ", "")
            for expression in custom_nonbonded_expressions
        )
        if amber_12_6_4 and not openmm_12_6_4:
            return {
                "available": True,
                "status": "error",
                "error": (
                    "Amber topology contains LENNARD_JONES_CCOEF, but OpenMM did not create "
                    "a CustomNonbondedForce with the r^-4 term."
                ),
                "version": getattr(openmm, "__version__", "unknown"),
                "force_classes": force_classes,
                "custom_nonbonded_expressions": custom_nonbonded_expressions,
                "amber_12_6_4_detected": amber_12_6_4,
                "openmm_12_6_4_force_present": openmm_12_6_4,
            }
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
            "force_classes": force_classes,
            "custom_nonbonded_expressions": custom_nonbonded_expressions,
            "amber_12_6_4_detected": amber_12_6_4,
            "openmm_12_6_4_force_present": openmm_12_6_4,
        }
    except Exception as exc:
        return {"available": True, "status": "error", "error": str(exc)}


def _prmtop_has_flag(path: str | Path, flag: str) -> bool:
    marker = f"%FLAG {flag}"
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        return any(line.rstrip("\r\n") == marker for line in handle)
