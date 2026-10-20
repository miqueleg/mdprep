"""mdprep's DCD reader against the DCD writer it exists to read: OpenMM's."""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")

from mdprep.analysis.dcd import read_dcd  # noqa: E402

pytestmark = [pytest.mark.external, pytest.mark.openmm]


def test_positions_and_a_truncated_octahedral_cell_round_trip(tmp_path):
    from openmm import Vec3, unit
    from openmm.app import DCDFile, Element, Topology

    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("WAT", chain)
    for name, element in (("O", "O"), ("H1", "H"), ("H2", "H")):
        topology.addAtom(name, Element.getBySymbol(element), residue)
    # tleap's octahedron, as OpenMM reduces it.
    side = 4.0
    vectors = (
        Vec3(side, 0, 0),
        Vec3(-side / 3, side * np.sqrt(8) / 3, 0),
        Vec3(-side / 3, -side * np.sqrt(2) / 3, side * np.sqrt(6) / 3),
    )
    topology.setPeriodicBoxVectors(vectors)
    frames = [np.array([[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.1, 0.3, 0.3]]) + 0.01 * i for i in range(3)]
    path = tmp_path / "t.dcd"
    with open(path, "wb") as handle:
        dcd = DCDFile(handle, topology, 0.002 * unit.picoseconds, interval=500)
        for frame in frames:
            dcd.writeModel(frame * unit.nanometer, periodicBoxVectors=vectors)

    data = read_dcd(path)

    assert np.allclose(data.positions, np.array(frames) * 10.0, atol=1e-4)
    expected = np.array([[v.x, v.y, v.z] for v in vectors]) * 10.0
    assert np.allclose(data.cells[0], expected, atol=1e-3)
    assert data.timestep_ps == pytest.approx(1.0, rel=1e-4)
