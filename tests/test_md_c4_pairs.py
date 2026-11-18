"""The 12-6-4 C4 term evaluated only where it is nonzero.

OpenMM builds a prmtop's C4 table into a CustomNonbondedForce over every
pair in the box, and the table is zero unless one atom is a 12-6-4 ion. An
interaction group of the ions against everyone has to give the same forces
-- including between two ions, which sit in both sets -- and differ in the
energy only through OpenMM's long-range correction.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdprep.md.c4_pairs import _covering_types, restrict_c4_to_ion_pairs

mm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

ION, OXYGEN, CARBON = 0, 1, 2


def c4_table(ion_c4=(0.0, 120.0, 40.0), ion_ion=8.0):
    """Types ION, OXYGEN, CARBON; only pairs with an ion are nonzero."""

    size = 3
    values = [0.0] * size * size
    for other, value in zip((ION, OXYGEN, CARBON), ion_c4):
        values[ION + size * other] = value
        values[other + size * ION] = value
    values[ION + size * ION] = ion_ion
    return size, values


def system_with_c4(types, positions, table):
    size, values = table
    system = mm.System()
    box = 3.0
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box, 0, 0), mm.Vec3(0, box, 0), mm.Vec3(0, 0, box)
    )
    force = mm.CustomNonbondedForce("-c/r^4; c=ccoef(type1, type2)")
    force.addTabulatedFunction("ccoef", mm.Discrete2DFunction(size, size, values))
    force.addPerParticleParameter("type")
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.9)
    force.setUseLongRangeCorrection(True)
    for kind in types:
        system.addParticle(16.0)
        force.addParticle([float(kind)])
    force.addExclusion(0, 1)
    system.addForce(force)
    return system


def energy_and_forces(system, positions):
    context = mm.Context(
        system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
    )


@pytest.fixture
def crowd():
    rng = np.random.default_rng(7)
    # Two ions, among oxygens and carbons, close enough to see each other.
    types = [ION, OXYGEN, ION] + [OXYGEN, CARBON] * 30
    positions = rng.uniform(0.0, 3.0, size=(len(types), 3))
    positions[2] = positions[0] + [0.35, 0.0, 0.0]
    return types, positions


def pair(types, positions, *, long_range_correction=True):
    full = system_with_c4(types, positions, c4_table())
    restricted = system_with_c4(types, positions, c4_table())
    for system in (full, restricted):
        system.getForce(0).setUseLongRangeCorrection(long_range_correction)
    return full, restricted


def test_restricted_energy_and_forces_are_the_full_sums(crowd):
    types, positions = crowd
    full, restricted = pair(types, positions, long_range_correction=False)

    ions = restrict_c4_to_ion_pairs(restricted)

    assert ions == [0, 2]
    e_full, f_full = energy_and_forces(full, positions)
    e_restricted, f_restricted = energy_and_forces(restricted, positions)
    assert np.abs(f_restricted - f_full).max() <= 1e-9 * np.abs(f_full).max()
    # The ion-ion pair is in both sets of the group and counted once.
    assert e_restricted == pytest.approx(e_full, rel=1e-12)


def test_only_the_long_range_correction_is_averaged_differently(crowd):
    """OpenMM averages the tail over the pairs it is told about, self-pairs
    included, so with a group the average moves by about 1/N of the tail:
    1e-3 kJ/mol on a 27,000-atom box, a constant at a given volume."""

    types, positions = crowd
    full, restricted = pair(types, positions)
    restrict_c4_to_ion_pairs(restricted)
    without_tail, _ = energy_and_forces(pair(types, positions, long_range_correction=False)[0], positions)

    e_full, f_full = energy_and_forces(full, positions)
    e_restricted, f_restricted = energy_and_forces(restricted, positions)

    assert np.abs(f_restricted - f_full).max() <= 1e-9 * np.abs(f_full).max()
    tail = e_full - without_tail
    assert abs(e_restricted - e_full) < abs(tail) / len(types)


def test_the_restriction_is_one_group_on_the_c4_force(crowd):
    types, positions = crowd
    system = system_with_c4(types, positions, c4_table())

    restrict_c4_to_ion_pairs(system)

    force = system.getForce(0)
    assert force.getNumInteractionGroups() == 1
    first, second = force.getInteractionGroupParameters(0)
    assert sorted(first) == [0, 2] and sorted(second) == list(range(len(types)))


def test_other_custom_forces_are_left_alone(crowd):
    types, positions = crowd
    system = system_with_c4(types, positions, c4_table())
    other = mm.CustomNonbondedForce("eps*r; eps=1")
    for _ in types:
        other.addParticle([])
    system.addForce(other)

    restrict_c4_to_ion_pairs(system)

    assert system.getForce(1).getNumInteractionGroups() == 0


def test_a_table_of_zeros_is_left_alone(crowd):
    types, positions = crowd
    system = system_with_c4(types, positions, c4_table(ion_c4=(0.0, 0.0, 0.0), ion_ion=0.0))

    assert restrict_c4_to_ion_pairs(system) == []
    assert system.getForce(0).getNumInteractionGroups() == 0


def test_a_system_without_c4_is_untouched():
    system = mm.System()
    system.addParticle(1.0)
    system.addForce(mm.NonbondedForce())

    assert restrict_c4_to_ion_pairs(system) == []


def test_the_cover_is_the_ion_when_the_ion_touches_every_type():
    size, values = c4_table()

    assert _covering_types(values, size) == {ION}


def test_two_ion_types_both_end_up_in_the_cover():
    """Two 12-6-4 metals, say a zinc and a magnesium, each with their own row."""

    size = 4
    values = [0.0] * size * size
    for ion in (0, 3):
        for other in range(size):
            values[ion + size * other] = values[other + size * ion] = 1.0 + other

    assert _covering_types(values, size) == {0, 3}
