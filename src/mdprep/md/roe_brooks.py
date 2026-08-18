"""Manifest-driven Roe and Brooks equilibration protocol for OpenMM.

The implementation follows the staged protocol used by mdprep's development
system while making platform selection, convergence limits, random seeds, and
production reporting explicit and reproducible.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from mdprep.config.models import MolecularDynamicsConfig


class RoeBrooksError(ValueError):
    """Raised when the OpenMM MD protocol cannot complete safely."""


@dataclass(frozen=True)
class RoeBrooksResult:
    output_dir: Path
    platform: str
    step10_elapsed_ps: float
    production_steps: int
    final_pdb: Path
    final_checkpoint: Path
    trajectory: Path
    state_log: Path
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": "roe_brooks_2020",
            "output_dir": str(self.output_dir),
            "platform": self.platform,
            "step10_elapsed_ps": self.step10_elapsed_ps,
            "production_steps": self.production_steps,
            "final_pdb": str(self.final_pdb),
            "final_checkpoint": str(self.final_checkpoint),
            "trajectory": str(self.trajectory),
            "state_log": str(self.state_log),
            "report_path": str(self.report_path),
        }


BACKBONE_NAMES = {
    "N", "CA", "C", "O", "OXT", "P", "OP1", "OP2", "O5'", "C5'",
    "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
}
WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TP3", "SPC", "SPCE", "OPC"}
ION_RESNAMES = {
    "NA", "CL", "K", "RB", "CS", "LI", "MG", "CA", "ZN", "MN", "FE",
    "CU", "CO", "NI",
}
AUTO_PLATFORM_ORDER = ("CUDA", "HIP", "Metal", "OpenCL", "CPU", "Reference")


def steps_from_time(time_q: Any, timestep_q: Any) -> int:
    return int(round(time_q / timestep_q))


def density_plateau_check(
    times_ps: Sequence[float],
    densities_g_ml: Sequence[float],
    *,
    window_ps: float,
    report_interval_ps: float,
    slope_threshold: float,
    mean_difference_threshold: float,
    minimum_points: int,
) -> tuple[bool, dict[str, object]]:
    """Evaluate the final point-based density window."""

    times = np.asarray(times_ps, dtype=float)
    densities = np.asarray(densities_g_ml, dtype=float)
    if len(densities) < minimum_points:
        return False, {"reason": f"need at least {minimum_points} points"}
    window_points = max(
        minimum_points,
        int(round(window_ps / report_interval_ps)) + 1,
    )
    if len(densities) < window_points:
        return False, {"reason": f"need at least {window_points} window points"}
    window_times = times[-window_points:]
    window_densities = densities[-window_points:]
    slope, _ = np.polyfit(window_times, window_densities, 1)
    midpoint = len(window_densities) // 2
    mean_first = float(np.mean(window_densities[:midpoint]))
    mean_second = float(np.mean(window_densities[midpoint:]))
    mean_difference = abs(mean_second - mean_first)
    info: dict[str, object] = {
        "points": window_points,
        "window_ps": float(window_times[-1] - window_times[0]),
        "slope_g_ml_ps": float(slope),
        "slope_threshold_g_ml_ps": slope_threshold,
        "mean_difference_g_ml": mean_difference,
        "mean_difference_threshold_g_ml": mean_difference_threshold,
        "standard_deviation_g_ml": float(np.std(window_densities)),
        "mean_first_half_g_ml": mean_first,
        "mean_second_half_g_ml": mean_second,
    }
    converged = (
        abs(float(slope)) < slope_threshold
        and mean_difference < mean_difference_threshold
    )
    return converged, info


def platform_candidates(requested: str, available: Sequence[str]) -> list[str]:
    """Return deterministic accelerator-first candidates for an OpenMM run."""

    available_set = set(available)
    if requested != "auto":
        if requested not in available_set:
            raise RoeBrooksError(
                f"Requested OpenMM platform {requested!r} is unavailable; "
                f"available platforms: {', '.join(available)}"
            )
        return [requested]
    candidates = [name for name in AUTO_PLATFORM_ORDER if name in available_set]
    if not candidates:
        raise RoeBrooksError("OpenMM reports no usable computational platforms")
    return candidates


def _platform_properties(platform: Any, config: MolecularDynamicsConfig) -> dict[str, str]:
    names = set(platform.getPropertyNames())
    properties: dict[str, str] = {}
    if config.device_index is not None and "DeviceIndex" in names:
        properties["DeviceIndex"] = str(config.device_index)
    if "Precision" in names:
        properties["Precision"] = config.precision
    if config.cpu_threads is not None and "Threads" in names:
        properties["Threads"] = str(config.cpu_threads)
    return properties


def run_roe_brooks(
    *,
    prmtop_path: str | Path,
    inpcrd_path: str | Path,
    output_dir: str | Path,
    config: MolecularDynamicsConfig,
) -> RoeBrooksResult:
    """Run equilibration and production from validated Amber inputs."""

    if not config.enabled or config.production is None:
        raise RoeBrooksError("The molecular_dynamics stage is not enabled")
    try:
        import openmm as mm
        from openmm import Platform, unit
        from openmm.app import (
            AmberInpcrdFile,
            AmberPrmtopFile,
            CheckpointReporter,
            DCDReporter,
            HBonds,
            PDBFile,
            PME,
            Simulation,
            StateDataReporter,
        )
    except ImportError as exc:
        raise RoeBrooksError(
            "OpenMM is required for molecular_dynamics.enabled: true; install the "
            "mdprep MD optional dependencies"
        ) from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    topology = prmtop.topology
    positions = inpcrd.positions
    box_vectors = inpcrd.boxVectors
    if topology.getNumAtoms() != len(positions):
        raise RoeBrooksError("prmtop and inpcrd atom counts differ")
    if box_vectors is None:
        raise RoeBrooksError("Roe--Brooks PME/NPT requires periodic box vectors")

    temperature = config.temperature_kelvin * unit.kelvin
    pressure = config.pressure_atmosphere * unit.atmosphere
    started = perf_counter()

    def build_system(*, constraints: Any, rigid_water: bool, barostat: bool) -> Any:
        system = prmtop.createSystem(
            nonbondedMethod=PME,
            nonbondedCutoff=0.8 * unit.nanometer,
            constraints=constraints,
            rigidWater=rigid_water,
            ewaldErrorTolerance=1.0e-4,
        )
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(True)
        if barostat:
            barostat_force = mm.MonteCarloBarostat(pressure, temperature, 100)
            barostat_force.setRandomNumberSeed(config.random_seed)
            system.addForce(barostat_force)
        return system

    def add_restraints(
        system: Any,
        reference_positions: Any,
        force_constant: float,
        selection: str,
    ) -> None:
        k = (
            force_constant
            * unit.kilocalories_per_mole
            / unit.angstrom**2
        ).in_units_of(unit.kilojoules_per_mole / unit.nanometer**2)
        force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", k)
        for parameter in ("x0", "y0", "z0"):
            force.addPerParticleParameter(parameter)
        reference_nm = reference_positions.value_in_unit(unit.nanometer)
        for atom in topology.atoms():
            name = atom.name.strip()
            is_hydrogen = (
                atom.element is not None and atom.element.symbol == "H"
            ) or name.startswith("H")
            residue_name = atom.residue.name.strip().upper()
            if is_hydrogen or residue_name in WATER_RESNAMES | ION_RESNAMES:
                continue
            if selection == "backbone" and name not in BACKBONE_NAMES:
                continue
            if selection not in {"heavy", "backbone"}:
                raise RoeBrooksError(f"Unknown restraint selection {selection!r}")
            point = reference_nm[atom.index]
            force.addParticle(atom.index, [point.x, point.y, point.z])
        system.addForce(force)

    available = [
        Platform.getPlatform(index).getName()
        for index in range(Platform.getNumPlatforms())
    ]
    trial_system = build_system(constraints=HBonds, rigid_water=True, barostat=False)
    selected_platform = None
    selected_properties: dict[str, str] = {}
    platform_failures: dict[str, str] = {}
    for candidate in platform_candidates(config.platform, available):
        candidate_platform = Platform.getPlatformByName(candidate)
        candidate_properties = _platform_properties(candidate_platform, config)
        trial_integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            trial_simulation = Simulation(
                topology,
                trial_system,
                trial_integrator,
                candidate_platform,
                candidate_properties,
            )
            trial_simulation.context.setPositions(positions)
            trial_simulation.context.setPeriodicBoxVectors(*box_vectors)
            trial_state = trial_simulation.context.getState(getEnergy=True)
            energy = trial_state.getPotentialEnergy().value_in_unit(
                unit.kilojoules_per_mole
            )
            if not math.isfinite(energy):
                raise RoeBrooksError("initial energy is not finite")
        except Exception as exc:  # OpenMM plugins raise several exception types
            platform_failures[candidate] = str(exc)
            if config.platform != "auto":
                raise RoeBrooksError(
                    f"Requested OpenMM platform {candidate!r} failed: {exc}"
                ) from exc
            continue
        selected_platform = candidate_platform
        selected_properties = candidate_properties
        del trial_simulation, trial_integrator
        break
    if selected_platform is None:
        raise RoeBrooksError(
            "No OpenMM platform could initialize: "
            + "; ".join(f"{key}: {value}" for key, value in platform_failures.items())
        )

    stage_records: list[dict[str, object]] = []

    def make_simulation(
        system: Any,
        integrator: Any,
        stage_positions: Any,
        stage_box: Any,
        *,
        velocities: Any | None = None,
        initialize_velocities: bool = False,
    ) -> Any:
        simulation = Simulation(
            topology,
            system,
            integrator,
            selected_platform,
            selected_properties,
        )
        simulation.context.setPositions(stage_positions)
        simulation.context.setPeriodicBoxVectors(*stage_box)
        if velocities is not None:
            simulation.context.setVelocities(velocities)
        elif initialize_velocities:
            simulation.context.setVelocitiesToTemperature(
                temperature, config.random_seed
            )
        return simulation

    def write_pdb(simulation: Any, name: str) -> Path:
        path = output / name
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        with path.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(topology, state.getPositions(), handle, keepIds=True)
        return path

    def record_stage(simulation: Any, label: str) -> Any:
        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
        if not all(math.isfinite(value) for value in (potential, kinetic, volume)):
            raise RoeBrooksError(f"{label} produced a non-finite state")
        stage_records.append(
            {
                "stage": label,
                "potential_energy_kj_mol": potential,
                "kinetic_energy_kj_mol": kinetic,
                "box_volume_nm3": volume,
            }
        )
        return state

    def save_restart(simulation: Any, stem: str) -> Path:
        checkpoint = output / f"{stem}.chk"
        state_path = output / f"{stem}.xml"
        simulation.saveCheckpoint(str(checkpoint))
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True,
            enforcePeriodicBox=True,
        )
        state_path.write_text(mm.XmlSerializer.serialize(state), encoding="utf-8")
        return checkpoint

    dt_min = 1.0 * unit.femtoseconds
    dt_1fs = 1.0 * unit.femtoseconds
    dt_2fs = 2.0 * unit.femtoseconds

    # Steps 1--5. Positional forces are added before Context creation.
    reference_positions = positions
    current_positions = positions
    current_box = box_vectors
    system = build_system(constraints=None, rigid_water=False, barostat=False)
    add_restraints(system, reference_positions, 5.0, "heavy")
    simulation = make_simulation(
        system, mm.VerletIntegrator(dt_min), current_positions, current_box
    )
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=1000)
    write_pdb(simulation, "step01.pdb")
    state = record_stage(simulation, "step01")
    current_positions = state.getPositions()
    current_box = state.getPeriodicBoxVectors()

    # Make the unconstrained minimum compatible with the SHAKE/rigid-water
    # manifold before heating. Some accelerator and CPU constraint solvers
    # otherwise fail on the first integration step for strained input water.
    system = build_system(constraints=HBonds, rigid_water=True, barostat=False)
    add_restraints(system, reference_positions, 5.0, "heavy")
    simulation = make_simulation(
        system, mm.VerletIntegrator(dt_min), current_positions, current_box
    )
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=500)
    write_pdb(simulation, "step01b.pdb")
    state = record_stage(simulation, "step01b")
    current_positions = state.getPositions()
    current_box = state.getPeriodicBoxVectors()

    system = build_system(constraints=HBonds, rigid_water=True, barostat=False)
    add_restraints(system, reference_positions, 5.0, "heavy")
    integrator = mm.LangevinMiddleIntegrator(temperature, 5.0 / unit.picosecond, dt_1fs)
    integrator.setConstraintTolerance(1.0e-6)
    integrator.setRandomNumberSeed(config.random_seed + 2)
    simulation = make_simulation(
        system, integrator, current_positions, current_box, initialize_velocities=True
    )
    simulation.step(steps_from_time(15.0 * unit.picoseconds, dt_1fs))
    write_pdb(simulation, "step02.pdb")
    save_restart(simulation, "after_step02")
    state = record_stage(simulation, "step02")
    current_positions, current_velocities = state.getPositions(), state.getVelocities()
    current_box = state.getPeriodicBoxVectors()

    for label, restraint_k in (("step03", 2.0), ("step04", 0.1)):
        system = build_system(constraints=None, rigid_water=False, barostat=False)
        add_restraints(system, reference_positions, restraint_k, "heavy")
        simulation = make_simulation(
            system, mm.VerletIntegrator(dt_min), current_positions, current_box
        )
        mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=1000)
        write_pdb(simulation, f"{label}.pdb")
        state = record_stage(simulation, label)
        current_positions, current_box = state.getPositions(), state.getPeriodicBoxVectors()

    system = build_system(constraints=None, rigid_water=False, barostat=False)
    simulation = make_simulation(
        system, mm.VerletIntegrator(dt_min), current_positions, current_box
    )
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=1000)
    write_pdb(simulation, "step05.pdb")
    state = record_stage(simulation, "step05")
    current_positions, current_box = state.getPositions(), state.getPeriodicBoxVectors()

    system = build_system(constraints=HBonds, rigid_water=True, barostat=False)
    simulation = make_simulation(
        system, mm.VerletIntegrator(dt_min), current_positions, current_box
    )
    mm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=500)
    write_pdb(simulation, "step05b.pdb")
    state = record_stage(simulation, "step05b")
    current_positions, current_box = state.getPositions(), state.getPeriodicBoxVectors()
    step5_reference = current_positions

    system = build_system(constraints=HBonds, rigid_water=True, barostat=False)
    add_restraints(system, step5_reference, 1.0, "heavy")
    settle_dt = 0.25 * unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(
        temperature, 20.0 / unit.picosecond, settle_dt
    )
    integrator.setConstraintTolerance(1.0e-6)
    integrator.setRandomNumberSeed(config.random_seed + 5)
    simulation = make_simulation(
        system, integrator, current_positions, current_box, initialize_velocities=True
    )
    simulation.step(steps_from_time(0.2 * unit.picoseconds, settle_dt))
    write_pdb(simulation, "step05c.pdb")
    state = record_stage(simulation, "step05c")
    current_positions, current_velocities = state.getPositions(), state.getVelocities()
    current_box = state.getPeriodicBoxVectors()

    def run_md_stage(
        label: str,
        *,
        duration_ps: float,
        timestep: Any,
        restraint_k: float | None,
        restraint_selection: str | None,
        barostat: bool,
        seed_offset: int,
    ) -> None:
        nonlocal current_positions, current_velocities, current_box
        system = build_system(constraints=HBonds, rigid_water=True, barostat=barostat)
        if restraint_k is not None and restraint_selection is not None:
            add_restraints(system, step5_reference, restraint_k, restraint_selection)
        integrator = mm.LangevinMiddleIntegrator(
            temperature, 5.0 / unit.picosecond, timestep
        )
        integrator.setConstraintTolerance(1.0e-6)
        integrator.setRandomNumberSeed(config.random_seed + seed_offset)
        simulation = make_simulation(
            system,
            integrator,
            current_positions,
            current_box,
            velocities=current_velocities,
        )
        simulation.step(steps_from_time(duration_ps * unit.picoseconds, timestep))
        write_pdb(simulation, f"{label}.pdb")
        state = record_stage(simulation, label)
        current_positions, current_velocities = state.getPositions(), state.getVelocities()
        current_box = state.getPeriodicBoxVectors()

    run_md_stage(
        "step06", duration_ps=5.0, timestep=dt_1fs, restraint_k=1.0,
        restraint_selection="heavy", barostat=False, seed_offset=6,
    )
    run_md_stage(
        "step07", duration_ps=5.0, timestep=dt_1fs, restraint_k=0.5,
        restraint_selection="heavy", barostat=True, seed_offset=7,
    )
    run_md_stage(
        "step08", duration_ps=10.0, timestep=dt_1fs, restraint_k=0.5,
        restraint_selection="backbone",
        barostat=config.step08_ensemble == "NPT", seed_offset=8,
    )
    run_md_stage(
        "step09", duration_ps=10.0, timestep=dt_2fs, restraint_k=None,
        restraint_selection=None, barostat=True, seed_offset=9,
    )

    # Step 10: unrestrained NPT until the configured density window converges.
    system = build_system(constraints=HBonds, rigid_water=True, barostat=True)
    integrator = mm.LangevinMiddleIntegrator(
        temperature, 5.0 / unit.picosecond, dt_2fs
    )
    integrator.setConstraintTolerance(1.0e-6)
    integrator.setRandomNumberSeed(config.random_seed + 10)
    simulation = make_simulation(
        system, integrator, current_positions, current_box,
        velocities=current_velocities,
    )
    total_mass_da = sum(
        system.getParticleMass(index).value_in_unit(unit.dalton)
        for index in range(system.getNumParticles())
    )
    density_path = output / "density_step10.csv"
    with density_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(["time_ps", "density_g_ml", "box_volume_nm3"])
    report_steps = max(
        1,
        steps_from_time(config.density.report_interval_ps * unit.picoseconds, dt_2fs),
    )
    chunk_steps = max(
        1,
        steps_from_time(config.density.increment_ns * unit.nanoseconds, dt_2fs),
    )
    maximum_steps = steps_from_time(
        config.density.maximum_duration_ns * unit.nanoseconds, dt_2fs
    )
    elapsed_steps = 0
    times: list[float] = []
    densities: list[float] = []
    convergence: dict[str, object] = {}
    converged = False
    while elapsed_steps < maximum_steps and not converged:
        steps_left = min(chunk_steps, maximum_steps - elapsed_steps)
        while steps_left > 0:
            advance = min(report_steps, steps_left)
            simulation.step(advance)
            elapsed_steps += advance
            steps_left -= advance
            state = simulation.context.getState(enforcePeriodicBox=True)
            volume = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
            density = (total_mass_da / 6.02214076e23) / (volume * 1.0e-21)
            elapsed_ps = (elapsed_steps * dt_2fs).value_in_unit(unit.picoseconds)
            if not all(math.isfinite(value) for value in (volume, density)):
                raise RoeBrooksError("Step 10 produced a non-finite density or box volume")
            times.append(elapsed_ps)
            densities.append(density)
            with density_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow([elapsed_ps, density, volume])
            converged, convergence = density_plateau_check(
                times,
                densities,
                window_ps=config.density.window_ps,
                report_interval_ps=config.density.report_interval_ps,
                slope_threshold=config.density.slope_threshold_g_ml_ps,
                mean_difference_threshold=(
                    config.density.mean_difference_threshold_g_ml
                ),
                minimum_points=config.density.minimum_points,
            )
            if converged:
                break
    if not converged:
        raise RoeBrooksError(
            "Roe--Brooks step 10 did not reach the configured density plateau within "
            f"{config.density.maximum_duration_ns} ns; last check: {convergence}"
        )
    step10_elapsed_ps = times[-1]
    write_pdb(simulation, "step10.pdb")
    save_restart(simulation, "after_step10")
    record_stage(simulation, "step10")

    production = config.production
    trajectory = output / "production.dcd"
    state_log = output / "production.log"
    periodic_checkpoint = output / "production_periodic.chk"
    simulation.reporters.append(
        DCDReporter(str(trajectory), production.trajectory_interval_steps)
    )
    simulation.reporters.append(
        StateDataReporter(
            str(state_log),
            production.state_interval_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            progress=True,
            remainingTime=True,
            totalSteps=simulation.currentStep + production.steps,
            separator="\t",
        )
    )
    simulation.reporters.append(
        CheckpointReporter(str(periodic_checkpoint), production.checkpoint_interval_steps)
    )
    simulation.step(production.steps)
    final_pdb = write_pdb(simulation, "production_last.pdb")
    final_checkpoint = save_restart(simulation, "after_production")
    final_state = record_stage(simulation, "production")
    final_positions = final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if not np.isfinite(final_positions).all():
        raise RoeBrooksError("Production finished with non-finite coordinates")

    report_path = output / "md_report.json"
    report = {
        "protocol": "roe_brooks_2020",
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "atom_count": topology.getNumAtoms(),
        "platform_requested": config.platform,
        "platform_selected": selected_platform.getName(),
        "platform_properties": selected_properties,
        "platform_initialization_failures": platform_failures,
        "random_seed": config.random_seed,
        "temperature_kelvin": config.temperature_kelvin,
        "pressure_atmosphere": config.pressure_atmosphere,
        "step08_ensemble": config.step08_ensemble,
        "step10_elapsed_ps": step10_elapsed_ps,
        "step10_convergence": convergence,
        "production": production.model_dump(mode="json"),
        "stages": stage_records,
        "runtime_seconds": perf_counter() - started,
        "outputs": {
            "trajectory": str(trajectory),
            "state_log": str(state_log),
            "periodic_checkpoint": str(periodic_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "final_pdb": str(final_pdb),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RoeBrooksResult(
        output_dir=output,
        platform=selected_platform.getName(),
        step10_elapsed_ps=step10_elapsed_ps,
        production_steps=production.steps,
        final_pdb=final_pdb,
        final_checkpoint=final_checkpoint,
        trajectory=trajectory,
        state_log=state_log,
        report_path=report_path,
    )
