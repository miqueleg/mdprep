"""OpenMM-level regression for the Roe--Brooks coordinate frame.

The published protocol restrains solute heavy atoms to the coordinates of the
input structure. Handing wrapped coordinates to the next stage while the
restraint references stay unwrapped puts a restrained particle that crossed a
periodic face a full box length from its reference; at 5 kcal/mol/A^2 that is a
restraint force of order 1e5 kJ/mol/nm and the next Langevin stage NaNs.
"""

import math

import pytest

from mdprep.md.roe_brooks import periodic_output_state, stage_handoff_state


pytestmark = [pytest.mark.external, pytest.mark.openmm]

BOX_NM = 2.0
OUTSIDE_X_NM = 2.5  # one box length past the primary image of x = 0.5 nm


def _periodic_context():
    mm = pytest.importorskip("openmm", reason="OpenMM is required for this test")
    from openmm import Vec3, unit

    system = mm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(
        Vec3(BOX_NM, 0, 0) * unit.nanometer,
        Vec3(0, BOX_NM, 0) * unit.nanometer,
        Vec3(0, 0, BOX_NM) * unit.nanometer,
    )
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.5 * unit.nanometer)
    for _ in range(2):
        nonbonded.addParticle(0.0, 0.2 * unit.nanometer, 0.0)
    system.addForce(nonbonded)

    context = mm.Context(system, mm.VerletIntegrator(1.0 * unit.femtoseconds))
    positions = [Vec3(OUTSIDE_X_NM, 1.0, 1.0), Vec3(1.0, 1.0, 1.0)] * unit.nanometer
    context.setPositions(positions)
    return mm, unit, context


def test_stage_handoff_preserves_the_restraint_reference_frame():
    _, unit, context = _periodic_context()

    handoff = stage_handoff_state(context)
    x_nm = handoff.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[0][0]

    assert math.isclose(x_nm, OUTSIDE_X_NM, abs_tol=1e-6)


def test_written_output_is_wrapped_into_the_primary_image():
    _, unit, context = _periodic_context()

    wrapped = periodic_output_state(context)
    x_nm = wrapped.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[0][0]

    # Same particle, one box length in: this is correct for a viewer and wrong
    # as the seed for a restrained stage.
    assert math.isclose(x_nm, OUTSIDE_X_NM - BOX_NM, abs_tol=1e-6)


def test_wrapped_handoff_would_displace_a_restrained_particle_by_a_box_length():
    """Pins the mechanism: wrapping costs a full box length of restraint error.

    On the 13.4 nm box this was first seen on, the same 5 kcal/mol/A^2 restraint
    produced forces of ~3e5 kJ/mol/nm, so the assertion is written relative to
    the box edge rather than as an absolute magnitude.
    """
    _, unit, context = _periodic_context()

    reference = context.getState(getPositions=True, enforcePeriodicBox=False)
    reference_nm = reference.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    force_constant = (
        5.0 * unit.kilocalories_per_mole / unit.angstrom**2
    ).value_in_unit(unit.kilojoules_per_mole / unit.nanometer**2)

    seeded_unwrapped = stage_handoff_state(context).getPositions(
        asNumpy=True
    ).value_in_unit(unit.nanometer)
    seeded_wrapped = periodic_output_state(context).getPositions(
        asNumpy=True
    ).value_in_unit(unit.nanometer)

    error_unwrapped = abs(seeded_unwrapped[0][0] - reference_nm[0][0])
    error_wrapped = abs(seeded_wrapped[0][0] - reference_nm[0][0])

    assert error_unwrapped < 1.0e-6
    assert math.isclose(error_wrapped, BOX_NM, abs_tol=1.0e-6)
    assert force_constant * error_unwrapped < 1.0e-3
    assert force_constant * error_wrapped > 0.99 * force_constant * BOX_NM
