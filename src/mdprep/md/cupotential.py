"""Running the MD on NVIDIA's cuPotential (``molecular_dynamics.engine: cupotential``).

cuPotential is NVIDIA's nonbonded engine for OpenMM's CUDA platform, shipped
as ``nv-openmm-plugin``. It replaces three things in a System built the
usual way: the ``NonbondedForce`` (Coulomb by its own lattice sum, Lennard-
Jones, the 1-4 exceptions and the dispersion correction, all copied from the
source force), the ``LangevinMiddleIntegrator`` and the
``MonteCarloBarostat``. Bonded terms, restraints and the 12-6-4 C4 force stay
OpenMM's.

What it asks of the System, and what that changes here:

- It runs the dynamics. The minimisations keep OpenMM's own force: the
  plugin is made for MD on a relaxed box, not for the overlapping atoms of
  one straight from tleap.
- The source force's Ewald error tolerance must be OpenMM's default, 5e-4;
  the plugin sizes its own mesh from it. mdprep's own OpenMM runs use 1e-4.
- Its integrator removes the centre-of-mass motion itself, so OpenMM's
  ``CMMotionRemover`` comes out of the System; left in, both would run and
  the plugin could not capture the step as one CUDA graph.
- It runs on CUDA, in single or mixed precision, and only in a periodic box.

The plugin is NVIDIA proprietary software under NVIDIA's own licence, which
mdprep cannot redistribute: nothing installs it except on request (the
official notebook's section D, or ``pip install nv-openmm-plugin``).
"""

from __future__ import annotations

from typing import Any

# The only tolerance the plugin accepts from its source force.
EWALD_ERROR_TOLERANCE = 5.0e-4
PLUGIN_PACKAGE = "nv-openmm-plugin"
LICENCE_URL = "https://github.com/NVIDIA/cuPotential/blob/main/LICENSE-cupo.txt"


class CuPotentialUnavailable(RuntimeError):
    """The manifest asked for cuPotential and this environment cannot run it."""


def load_plugin() -> Any:
    """Import nv_openmm_plugin and make sure its CUDA kernels registered.

    The package imports softly -- a library that would not load leaves an
    importable module behind -- so the classes are touched here, where the
    real reason surfaces, rather than at the first Context.
    """

    try:
        import nv_openmm_plugin as nv
    except ImportError as exc:
        raise CuPotentialUnavailable(
            f"molecular_dynamics.engine: cupotential needs NVIDIA's {PLUGIN_PACKAGE} "
            f"in the environment the MD runs in ({exc}). Install it yourself -- "
            f"pip install {PLUGIN_PACKAGE}, under NVIDIA's licence ({LICENCE_URL}) -- "
            "or let section D of the official notebook do it; or set engine: openmm."
        ) from exc
    try:
        nv.NonbondedForce  # noqa: B018 - the attribute access is the load
    except Exception as exc:
        raise CuPotentialUnavailable(
            f"{PLUGIN_PACKAGE} is installed but its libraries did not load: {exc}"
        ) from exc
    return nv


def plugin_version() -> str | None:
    from importlib import metadata

    try:
        return metadata.version(PLUGIN_PACKAGE)
    except metadata.PackageNotFoundError:
        return None


def _nonbonded_index(system: Any, mm: Any) -> int:
    indices = [
        index
        for index in range(system.getNumForces())
        if isinstance(system.getForce(index), mm.NonbondedForce)
    ]
    if len(indices) != 1:
        raise CuPotentialUnavailable(
            f"cuPotential replaces exactly one NonbondedForce; this system has {len(indices)}"
        )
    return indices[0]


def use_cupotential_forces(system: Any, nv: Any) -> Any:
    """Swap the System's NonbondedForce for the plugin's copy of it."""

    import openmm as mm

    index = _nonbonded_index(system, mm)
    source = system.getForce(index)
    tolerance = source.getEwaldErrorTolerance()
    if abs(tolerance - EWALD_ERROR_TOLERANCE) > 1e-12:
        raise CuPotentialUnavailable(
            f"cuPotential needs an Ewald error tolerance of {EWALD_ERROR_TOLERANCE}; "
            f"the system was built with {tolerance}"
        )
    replacement = nv.NonbondedForce(source)
    system.removeForce(index)
    system.addForce(replacement)
    return system


def use_cupotential_dynamics(system: Any, nv: Any) -> Any:
    """Make the System ready for the plugin's integrator and barostat.

    The plugin's integrator removes centre-of-mass motion every 100 steps
    itself, and a stock ``MonteCarloBarostat`` alongside the plugin's is
    rejected, so OpenMM's remover goes and its barostat becomes the plugin's,
    with the same pressure, temperature, frequency and seed.
    """

    import openmm as mm
    from openmm import unit

    for index in reversed(range(system.getNumForces())):
        force = system.getForce(index)
        if isinstance(force, mm.CMMotionRemover):
            system.removeForce(index)
        elif isinstance(force, mm.MonteCarloBarostat):
            pressure = force.getDefaultPressure()
            temperature = force.getDefaultTemperature()
            frequency = force.getFrequency()
            seed = force.getRandomNumberSeed()
            system.removeForce(index)
            system.addForce(
                nv.MonteCarloBarostat(
                    pressure.in_units_of(unit.bar),
                    temperature,
                    frequency,
                    seed=seed,
                )
            )
    return system


def langevin_integrator(nv: Any, temperature: Any, friction: Any, timestep: Any) -> Any:
    """The plugin's LangevinMiddleIntegrator: OpenMM's step, fused on the GPU."""

    return nv.LangevinMiddleIntegrator(temperature, friction, timestep)
