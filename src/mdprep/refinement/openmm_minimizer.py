"""Force-field hydrogen refinement.

The QM/MM refinement moves hydrogens and nothing else: every heavy atom is
held at its experimental position, and the quantum method is spent entirely on
deciding where the protons sit. That is an expensive way to answer a question
a force field already answers, so this backend answers it with the provisional
AM1-BCC/GAFF2 system the refinement stage has already built.

What it does **not** do is replace the quantum chemistry that produces charges
and parameters. The AM1-BCC charges used here are provisional, exactly as they
are for the QM/MM backend; a QMMESP or MCPB fit still runs afterwards and its
charges and parameters are what reach the final build.

Heavy atoms are pinned by giving them zero mass, which OpenMM's minimizer
honours exactly rather than approximately. The result is verified before it is
returned.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from mdprep.config.models import OpenmmRefinementConfig


class OpenmmRefinementError(ValueError):
    """Raised when the force-field hydrogen refinement cannot complete."""


# A heavy atom must not move at all. The tolerance only absorbs the float32
# round-trip through OpenMM's single-precision paths.
HEAVY_ATOM_TOLERANCE_ANGSTROM = 1.0e-4


@dataclass(frozen=True)
class OpenmmRefinementRun:
    optimized_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    seconds: float
    atom_count: int
    hydrogen_count: int
    held_heavy_atoms: int
    max_heavy_atom_shift_angstrom: float
    hydrogen_rms_displacement_angstrom: float
    hydrogen_max_displacement_angstrom: float
    initial_energy_kj_mol: float
    final_energy_kj_mol: float
    platform: str
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "openmm",
            "seconds": self.seconds,
            "atom_count": self.atom_count,
            "hydrogen_count": self.hydrogen_count,
            "held_heavy_atoms": self.held_heavy_atoms,
            "max_heavy_atom_shift_angstrom": self.max_heavy_atom_shift_angstrom,
            "hydrogen_rms_displacement_angstrom": self.hydrogen_rms_displacement_angstrom,
            "hydrogen_max_displacement_angstrom": self.hydrogen_max_displacement_angstrom,
            "initial_energy_kj_mol": self.initial_energy_kj_mol,
            "final_energy_kj_mol": self.final_energy_kj_mol,
            "platform": self.platform,
        }


def run_openmm_refinement(
    *,
    prmtop_path: str | Path,
    inpcrd_path: str | Path,
    config: OpenmmRefinementConfig,
    work_dir: str | Path,
) -> OpenmmRefinementRun:
    """Relax every hydrogen against the provisional force field.

    Unlike the QM/MM backend there is no active region: a force field is cheap
    enough to relax the whole structure's hydrogens, and freezing the ones
    outside an arbitrary radius would put a seam through the hydrogen-bond
    network for no saving.
    """

    try:
        import numpy as np
        import openmm as mm
        from openmm import unit
        from openmm.app import PME, AmberInpcrdFile, AmberPrmtopFile, NoCutoff, Simulation
    except ImportError as exc:
        raise OpenmmRefinementError(
            "refinement.backend: openmm requires OpenMM; install the mdprep MD "
            "optional dependencies or select another backend."
        ) from exc

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    topology = prmtop.topology
    periodic = inpcrd.boxVectors is not None

    system = prmtop.createSystem(
        nonbondedMethod=PME if periodic else NoCutoff,
        nonbondedCutoff=config.nonbonded_cutoff_nanometer * unit.nanometer,
        # Hydrogens are the only thing being optimised, so nothing about them
        # may be constrained -- including the ones in water.
        constraints=None,
        rigidWater=False,
    )

    hydrogens: list[int] = []
    heavy: list[int] = []
    for atom in topology.atoms():
        # The element decides when the topology states one. Falling through
        # to the name whenever it is present makes a mercury or cadmium ion
        # a hydrogen -- HG1, CD1 -- and a metal counted as a hydrogen is one
        # that is left free to move where the heavy atoms are pinned, and
        # that no heavy-atom displacement check can see.
        is_hydrogen = (
            atom.element.symbol == "H"
            if atom.element is not None
            else atom.name.strip().startswith("H")
        )
        (hydrogens if is_hydrogen else heavy).append(atom.index)
    if not hydrogens:
        raise OpenmmRefinementError(
            "The provisional system contains no hydrogens to refine."
        )
    # Zero mass pins a particle for the minimizer as well as for dynamics.
    for index in heavy:
        system.setParticleMass(index, 0.0)

    simulation, platform = _build_simulation(
        mm,
        unit,
        Simulation,
        topology=topology,
        system=system,
        config=config,
    )
    simulation.context.setPositions(inpcrd.positions)
    if periodic:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

    before = simulation.context.getState(
        getPositions=True, getEnergy=True
    )
    initial_positions = np.asarray(
        before.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    initial_energy = before.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=config.tolerance_kj_mol_per_nm
        * unit.kilojoules_per_mole
        / unit.nanometer,
        maxIterations=config.max_iterations,
    )
    elapsed = time.perf_counter() - started

    after = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = np.asarray(
        after.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    final_energy = after.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    if not _is_finite(final_energy):
        raise OpenmmRefinementError(
            "Force-field hydrogen refinement produced a non-finite energy; the "
            "provisional parameters or the input coordinates are unusable."
        )

    heavy_shift = (
        float(np.abs(final_positions[heavy] - initial_positions[heavy]).max())
        if heavy
        else 0.0
    )
    if heavy_shift > HEAVY_ATOM_TOLERANCE_ANGSTROM:
        raise OpenmmRefinementError(
            f"Force-field refinement moved a heavy atom by {heavy_shift:.6f} A. "
            "Only hydrogens may move; the experimental heavy-atom coordinates "
            "must be preserved exactly."
        )

    moved = np.linalg.norm(
        final_positions[hydrogens] - initial_positions[hydrogens], axis=1
    )
    run = OpenmmRefinementRun(
        optimized_coordinates_angstrom=tuple(
            (float(x), float(y), float(z)) for x, y, z in final_positions
        ),
        seconds=elapsed,
        atom_count=topology.getNumAtoms(),
        hydrogen_count=len(hydrogens),
        held_heavy_atoms=len(heavy),
        max_heavy_atom_shift_angstrom=heavy_shift,
        hydrogen_rms_displacement_angstrom=float(np.sqrt((moved**2).mean())),
        hydrogen_max_displacement_angstrom=float(moved.max()),
        initial_energy_kj_mol=float(initial_energy),
        final_energy_kj_mol=float(final_energy),
        platform=platform.getName(),
        report_path=work / "openmm_refinement.json",
    )
    run.report_path.write_text(
        json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


@dataclass(frozen=True)
class OpenmmSiteMinimizationRun:
    """Coordinates after minimizing some atoms with every other one held."""

    coordinates_angstrom: tuple[tuple[float, float, float], ...]
    initial_energy_kj_mol: float
    final_energy_kj_mol: float
    movable_atoms: int
    held_atoms: int
    platform: str
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_energy_kj_mol": self.initial_energy_kj_mol,
            "final_energy_kj_mol": self.final_energy_kj_mol,
            "movable_atoms": self.movable_atoms,
            "held_atoms": self.held_atoms,
            "platform": self.platform,
            "seconds": self.seconds,
        }


def run_openmm_site_minimization(
    *,
    prmtop_path: str | Path,
    inpcrd_path: str | Path,
    movable: Sequence[int],
    tolerance_kj_mol_per_nm: float = 1.0,
    max_iterations: int = 0,
    platform: str = "auto",
    nonbonded_cutoff_angstrom: float = 15.0,
) -> OpenmmSiteMinimizationRun:
    """Minimise the ``movable`` atoms against a force field, holding the rest.

    The same pinning as the hydrogen refinement -- zero mass, which OpenMM's
    minimizer honours exactly -- on every atom outside the set, heavy or not.
    What it is for is a metal site's heavy atoms: a fit at a stationary point
    of the provisional nonbonded model rather than at the input geometry.
    """

    try:
        import numpy as np
        import openmm as mm
        from openmm import unit
        from openmm.app import (
            PME,
            AmberInpcrdFile,
            AmberPrmtopFile,
            CutoffNonPeriodic,
            Simulation,
        )
    except ImportError as exc:
        raise OpenmmRefinementError(
            "Minimising a metal site with the force field requires OpenMM."
        ) from exc

    config = OpenmmRefinementConfig(
        platform=platform,  # type: ignore[arg-type]
        tolerance_kj_mol_per_nm=tolerance_kj_mol_per_nm,
        max_iterations=max_iterations,
        nonbonded_cutoff_nanometer=nonbonded_cutoff_angstrom / 10.0,
    )
    prmtop = AmberPrmtopFile(str(prmtop_path))
    inpcrd = AmberInpcrdFile(str(inpcrd_path))
    periodic = inpcrd.boxVectors is not None
    system = prmtop.createSystem(
        nonbondedMethod=PME if periodic else CutoffNonPeriodic,
        nonbondedCutoff=config.nonbonded_cutoff_nanometer * unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    free = set(movable)
    if not free:
        raise OpenmmRefinementError("No atoms were selected to minimise.")
    for index in range(system.getNumParticles()):
        if index not in free:
            system.setParticleMass(index, 0.0)
    _drop_held_overlaps(
        system,
        np.asarray(inpcrd.positions.value_in_unit(unit.angstrom)),
        free,
        [f"{atom.residue.name}{atom.residue.id}@{atom.name}" for atom in prmtop.topology.atoms()],
    )
    simulation, chosen = _build_simulation(
        mm, unit, Simulation, topology=prmtop.topology, system=system, config=config
    )
    simulation.context.setPositions(inpcrd.positions)
    if periodic:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
    before = simulation.context.getState(getEnergy=True)
    started = time.perf_counter()
    mm.LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=config.tolerance_kj_mol_per_nm * unit.kilojoules_per_mole / unit.nanometer,
        maxIterations=config.max_iterations,
    )
    elapsed = time.perf_counter() - started
    after = simulation.context.getState(getPositions=True, getEnergy=True)
    final_energy = after.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    if not _is_finite(final_energy):
        raise OpenmmRefinementError(
            "Minimising the metal site produced a non-finite energy; the provisional "
            "parameters or the coordinates are unusable."
        )
    positions = np.asarray(after.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    return OpenmmSiteMinimizationRun(
        coordinates_angstrom=tuple((float(x), float(y), float(z)) for x, y, z in positions),
        initial_energy_kj_mol=float(
            before.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ),
        final_energy_kj_mol=float(final_energy),
        movable_atoms=len(free),
        held_atoms=system.getNumParticles() - len(free),
        platform=chosen.getName(),
        seconds=elapsed,
    )


# Two atoms closer than this are one on top of the other: tleap building a
# missing atom where another already is.
OVERLAP_ANGSTROM = 0.5


def _drop_held_overlaps(system: Any, positions: Any, free: set[int], labels: Sequence[str]) -> None:
    """Switch off the interaction of two held atoms that sit on each other.

    With no cutoff every pair counts, and one pair at zero distance -- a
    lysine NZ tleap rebuilt onto a water's hydrogen, far from the site --
    makes the whole energy NaN, and the minimiser with it. Both are held, so
    what they do to each other is a constant that no movable atom feels.
    An overlap involving an atom that moves is a broken site, and stops.
    """

    import numpy as np

    cells: dict[tuple[int, int, int], list[int]] = {}
    for index, (x, y, z) in enumerate(positions):
        cells.setdefault((int(x // 1.0), int(y // 1.0), int(z // 1.0)), []).append(index)
    pairs: list[tuple[int, int]] = []
    for (cx, cy, cz), members in cells.items():
        nearby = [
            other
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            for other in cells.get((cx + dx, cy + dy, cz + dz), ())
        ]
        for i in members:
            for j in nearby:
                if j > i and np.linalg.norm(positions[i] - positions[j]) < OVERLAP_ANGSTROM:
                    pairs.append((i, j))
    if not pairs:
        return
    moving = [(i, j) for i, j in pairs if i in free or j in free]
    if moving:
        i, j = moving[0]
        raise OpenmmRefinementError(
            f"{labels[i]} and {labels[j]} sit {np.linalg.norm(positions[i] - positions[j]):.2f} A "
            "apart inside the region being minimised; the structure has an atom on top "
            "of another there."
        )
    import openmm as mm

    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            for i, j in pairs:
                force.addException(i, j, 0.0, 1.0, 0.0, True)
        elif isinstance(force, mm.CustomNonbondedForce):
            excluded = {
                tuple(sorted(force.getExclusionParticles(k)))
                for k in range(force.getNumExclusions())
            }
            for i, j in pairs:
                if (i, j) not in excluded:
                    force.addExclusion(i, j)


# Fastest first. A platform being registered is not the same as it working:
# OpenMM ships the CUDA plugin wherever it was built with CUDA, so a machine
# with no GPU -- a CI runner, a cluster login node -- still lists it, and only
# fails when a context is created on it.
# CPU first, which is not the fastest but is the one that gives the same
# answer twice. Measured on the reference system: CPU with a pinned thread
# count minimises to bit-identical positions across runs, while CUDA -- even
# with DeterministicForces -- landed on final energies differing by 0.14
# kJ/mol, and the third decimal of a few hundred coordinates with it. Those
# coordinates are what tleap packs solvent around, so the run that follows
# builds a different system.
#
# This is a preparation tool: the minimisation is a few seconds of a run
# measured in tens, and a GPU is still there for anyone who asks for it by
# name and would rather have the speed than the repeatability.
AUTO_PLATFORM_ORDER = ("CPU", "CUDA", "HIP", "Metal", "OpenCL", "Reference")


def platform_candidates(requested: str, available: Sequence[str]) -> list[str]:
    """Platforms to try, in order, for a refinement."""

    available_set = set(available)
    if requested != "auto":
        if requested not in available_set:
            raise OpenmmRefinementError(
                f"Requested OpenMM platform {requested!r} is unavailable; "
                f"available platforms: {', '.join(available)}"
            )
        return [requested]
    candidates = [name for name in AUTO_PLATFORM_ORDER if name in available_set]
    if not candidates:
        raise OpenmmRefinementError("OpenMM reports no usable computational platforms")
    return candidates


def deterministic_platform_properties(platform_name: str) -> dict[str, str]:
    """Platform settings that make a minimisation repeatable.

    A minimiser has no random numbers in it, so the same input should give
    the same output -- and it did not. Two runs of one manifest differed in
    the third decimal of a few hundred coordinates, which is the signature
    of a floating-point sum taken in a different order, not of a different
    calculation. Forces are accumulated per thread and then reduced, so the
    order follows the thread count; on a GPU the reduction is unordered
    unless asked otherwise.

    Those coordinates are what tleap is handed, and tleap packs its solvent
    around them deterministically -- so a last-decimal difference in the
    solute comes out as a different number of waters, and the prepared
    system is not the one the last run built. That is the whole reason to
    pin this: AGENTS.md asks for a preparation that can be repeated.

    One thread costs time on a large system. It is the price of the answer
    being the same twice, and the minimisation is not the slow stage.
    """

    if platform_name == "CPU":
        return {"Threads": "1"}
    if platform_name in {"CUDA", "HIP", "OpenCL"}:
        return {"DeterministicForces": "true"}
    return {}


def _build_simulation(
    mm: Any,
    unit: Any,
    Simulation: Any,
    *,
    topology: Any,
    system: Any,
    config: OpenmmRefinementConfig,
) -> tuple[Any, Any]:
    """Create the simulation on the first platform that actually works.

    Selecting a platform by name and hoping is how a GPU-less machine with the
    CUDA plugin installed gets an unhandled driver error instead of a run on
    the CPU. Each candidate is tried, and an explicitly requested one still
    fails loudly rather than silently falling back to something slower.
    """

    available = [
        mm.Platform.getPlatform(index).getName()
        for index in range(mm.Platform.getNumPlatforms())
    ]
    failures: dict[str, str] = {}
    for candidate in platform_candidates(config.platform, available):
        platform = mm.Platform.getPlatformByName(candidate)
        try:
            simulation = Simulation(
                topology,
                system,
                mm.VerletIntegrator(1.0 * unit.femtoseconds),
                platform,
                deterministic_platform_properties(candidate),
            )
        except Exception as exc:  # OpenMM plugins raise several exception types
            failures[candidate] = str(exc)
            if config.platform != "auto":
                raise OpenmmRefinementError(
                    f"Requested OpenMM platform {candidate!r} failed: {exc}"
                ) from exc
            continue
        return simulation, platform
    raise OpenmmRefinementError(
        "No OpenMM platform could initialize: "
        + "; ".join(f"{key}: {value}" for key, value in failures.items())
    )


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
