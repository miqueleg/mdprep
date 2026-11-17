"""NVIDIA's cuPotential, where it is installed and a CUDA GPU can run it.

Skipped otherwise: the plugin is proprietary, CUDA-only and never a test
dependency. With it, the force mdprep hands the dynamics has to agree with
OpenMM's own on a solvated, relaxed box.
"""

from __future__ import annotations

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
nv = pytest.importorskip("nv_openmm_plugin")

from mdprep.md import cupotential


def cuda_or_skip():
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if "CUDA" not in names:
        pytest.skip("OpenMM has no CUDA platform here")
    try:
        cupotential.load_plugin()
    except cupotential.CuPotentialUnavailable as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def water_box():
    cuda_or_skip()
    forcefield = app.ForceField("amber14/tip3p.xml")
    modeller = app.Modeller(app.Topology(), [])
    modeller.addSolvent(forcefield, model="tip3p", boxSize=mm.Vec3(3.0, 3.0, 3.0) * unit.nanometer)

    def build():
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=0.8 * unit.nanometer,
            constraints=app.HBonds,
            ewaldErrorTolerance=cupotential.EWALD_ERROR_TOLERANCE,
        )
        return system

    # Relaxed with OpenMM's own force, as the Roe-Brooks minimisations are.
    system = build()
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CUDA"))
    context.setPositions(modeller.positions)
    mm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
    positions = context.getState(getPositions=True).getPositions()
    return build, positions


def energy_and_forces(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},
    )
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
    )


def test_the_plugins_force_agrees_with_openmms_on_a_relaxed_box(water_box):
    build, positions = water_box
    stock_energy, stock_forces = energy_and_forces(build(), positions)
    system = cupotential.use_cupotential_forces(build(), cupotential.load_plugin())
    energy, forces = energy_and_forces(system, positions)

    relative = np.sqrt(((forces - stock_forces) ** 2).sum(1).mean() / (stock_forces**2).sum(1).mean())
    assert relative < 5e-3
    assert abs(energy - stock_energy) < 1e-3 * abs(stock_energy)


def test_the_plugins_dynamics_step_and_stay_finite(water_box):
    build, positions = water_box
    plugin = cupotential.load_plugin()
    system = build()
    system.addForce(mm.MonteCarloBarostat(1 * unit.atmosphere, 300 * unit.kelvin, 25))
    cupotential.use_cupotential_forces(system, plugin)
    cupotential.use_cupotential_dynamics(system, plugin)
    integrator = cupotential.langevin_integrator(
        plugin, 300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    context.setPositions(positions)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    integrator.step(500)

    state = context.getState(getEnergy=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    assert state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3) > 20
