"""OpenMM-level regression for what a per-step PDB says about its box.

Two defects, both visible in a real Colab run of a four-chain halogenase in
a truncated octahedron:

- Every snapshot carried the pre-NPT cell. `PDBFile` takes CRYST1 from the
  topology, and the topology was the one loaded at the start, so step 7
  onwards declared 83.4 A while the system had compressed to 79.9 A.
- OpenMM's `enforcePeriodicBox` wraps molecules into the parallelepiped
  spanned by the cell vectors, corner at the origin. Same lattice, but a
  sheared prism with the solute off to one side: water that sat beside the
  protein was drawn 70 A away, against 54 A in the octahedron tleap built.
"""

import pytest

from mdprep.md.imaging import write_imaged_pdb


pytestmark = [pytest.mark.external, pytest.mark.openmm]

EDGE_NM = 4.0


def _octahedral_system():
    """A solute atom and four solvent atoms in a truncated octahedron."""

    mm = pytest.importorskip("openmm", reason="OpenMM is required for this test")
    import numpy as np
    from openmm import Vec3, unit
    from openmm.app import Element, Topology

    d = EDGE_NM
    vectors = (
        Vec3(d, 0.0, 0.0),
        Vec3(-d / 3.0, 2.0 * np.sqrt(2.0) / 3.0 * d, 0.0),
        Vec3(-d / 3.0, -np.sqrt(2.0) / 3.0 * d, np.sqrt(6.0) / 3.0 * d),
    )

    system = mm.System()
    topology = Topology()
    chain = topology.addChain()
    names = [("LIG", "C"), ("WAT", "O"), ("WAT", "O"), ("WAT", "O"), ("WAT", "O")]
    for resname, symbol in names:
        residue = topology.addResidue(resname, chain)
        topology.addAtom(symbol, Element.getBySymbol(symbol), residue)
        system.addParticle(12.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*[v * unit.nanometer for v in vectors])
    topology.setPeriodicBoxVectors([v * unit.nanometer for v in vectors])

    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.5 * unit.nanometer)
    for _ in names:
        force.addParticle(0.0, 0.2 * unit.nanometer, 0.0)
    system.addForce(force)

    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    return context, topology, vectors


def _cryst1(path):
    for line in path.read_text().splitlines():
        if line.startswith("CRYST1"):
            return [float(line[6 + 9 * i : 15 + 9 * i]) for i in range(3)] + [
                float(line[33 + 7 * i : 40 + 7 * i]) for i in range(3)
            ]
    raise AssertionError(f"no CRYST1 in {path}")


def test_the_snapshot_declares_the_box_the_state_is_in_not_the_one_it_started_in(
    tmp_path,
):
    """The NPT steps shrink the cell. A header still naming the starting one
    is what made the periodic images come out wrong."""

    from openmm import Vec3, unit

    context, topology, vectors = _octahedral_system()
    context.setPositions([Vec3(0, 0, 0)] * 5 * unit.nanometer)

    # As the NPT barostat would: a smaller cell, same shape.
    shrunk = [v * 0.9 for v in vectors]
    context.setPeriodicBoxVectors(*[v * unit.nanometer for v in shrunk])
    state = context.getState(getPositions=True, enforcePeriodicBox=True)

    path = write_imaged_pdb(
        tmp_path / "snap.pdb", topology, state, [[0], [1], [2], [3], [4]], [0]
    )

    a, b, c, alpha, beta, gamma = _cryst1(path)
    assert a == pytest.approx(EDGE_NM * 0.9 * 10.0, abs=0.01)
    assert (b, c) == (pytest.approx(a, abs=0.01), pytest.approx(a, abs=0.01))
    # Still a truncated octahedron, not quietly turned into something else.
    for angle in (alpha, beta, gamma):
        assert angle == pytest.approx(109.47, abs=0.01)


def test_the_solute_is_at_the_centre_and_the_solvent_is_around_it(tmp_path):
    """The complaint this fixes: an octahedron was asked for and a prism with
    the protein in a corner came back."""

    import numpy as np
    from openmm import Vec3, unit

    context, topology, vectors = _octahedral_system()
    cell = np.array([[v.x, v.y, v.z] for v in vectors])
    # Four solvent atoms placed one lattice vector away from the solute --
    # physically beside it, but wrapped to the far side of the cell.
    positions = [np.zeros(3)]
    for shift in (cell[0], cell[1], cell[2], cell[0] + cell[1]):
        positions.append(np.array([0.3, 0.0, 0.0]) + shift)
    context.setPositions([Vec3(*p) for p in positions] * unit.nanometer)
    state = context.getState(getPositions=True, enforcePeriodicBox=True)

    path = write_imaged_pdb(
        tmp_path / "snap.pdb", topology, state, [[0], [1], [2], [3], [4]], [0]
    )

    coordinates = []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            coordinates.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    coordinates = np.array(coordinates)

    assert np.allclose(coordinates[0], [0.0, 0.0, 0.0], atol=1e-3)
    # Every solvent atom is back beside the solute, not a cell away.
    assert np.linalg.norm(coordinates[1:], axis=1).max() < 5.0  # angstrom
