"""A snapshot has to come back in the box the manifest asked for.

A truncated octahedron is stored as a triclinic cell. OpenMM's
``enforcePeriodicBox`` wraps molecules into the parallelepiped spanned by
that cell's vectors with a corner at the origin -- the same lattice, but a
sheared prism rather than an octahedron, with the solute wherever it
happened to be and water that was beside the protein wrapped to the far
corner. On a real 40 ps run, water reached 54 A from the solute in tleap's
octahedron and 70 A in the snapshot written afterwards.
"""

import numpy as np
import pytest

from mdprep.md.imaging import (
    image_molecules_about_solute,
    solute_atom_indices,
    solvent_residue_names,
)


def truncated_octahedron(edge: float) -> np.ndarray:
    """Amber's cell for solvateOct, as OpenMM reduces it: three vectors of
    equal length at 109.4712 degrees."""

    d = edge
    return np.array(
        [
            [d, 0.0, 0.0],
            [-d / 3.0, 2.0 * np.sqrt(2.0) / 3.0 * d, 0.0],
            [-d / 3.0, -np.sqrt(2.0) / 3.0 * d, np.sqrt(6.0) / 3.0 * d],
        ]
    )


CELL = truncated_octahedron(83.359)


def test_the_solute_ends_up_at_the_origin():
    positions = np.array([[10.0, 20.0, 30.0], [12.0, 20.0, 30.0], [70.0, 5.0, 5.0]])

    imaged = image_molecules_about_solute(positions, CELL, [[0, 1], [2]], [0, 1])

    assert np.allclose(imaged[[0, 1]].mean(axis=0), [0.0, 0.0, 0.0], atol=1e-9)


def test_a_molecule_wrapped_to_the_far_corner_comes_back_beside_the_solute():
    """The actual complaint: water next to the protein, drawn 70 A away."""

    solute = np.array([[0.0, 0.0, 0.0]])
    # One cell vector away is the same place, physically.
    far = solute[0] + CELL[0] + CELL[1]
    positions = np.vstack([solute, far])

    imaged = image_molecules_about_solute(positions, CELL, [[0], [1]], [0])

    assert np.linalg.norm(imaged[1]) < 1e-9


def test_nothing_ends_up_further_away_than_the_octahedron_reaches():
    """Every molecule lands in the Wigner-Seitz cell of the lattice, which
    for the body-centred lattice a truncated octahedron tiles is that
    octahedron. Its circumradius is sqrt(5)/4 of the conventional cubic
    lattice constant, and the cell vectors are sqrt(3)/2 of it.

    This is the measurement the bug was visible in: water reached 54 A from
    the solute in tleap's octahedron and 70 A once OpenMM had wrapped the
    same frame into the parallelepiped.
    """

    rng = np.random.default_rng(0)
    fractional = rng.uniform(-3.0, 3.0, size=(400, 3))
    positions = np.vstack([[[0.0, 0.0, 0.0]], fractional @ CELL])
    molecules = [[i] for i in range(len(positions))]

    imaged = image_molecules_about_solute(positions, CELL, molecules, [0])

    radii = np.linalg.norm(imaged, axis=1)
    cubic_constant = 2.0 * np.linalg.norm(CELL[0]) / np.sqrt(3.0)
    circumradius = np.sqrt(5.0) / 4.0 * cubic_constant
    assert radii.max() <= circumradius + 1e-6
    # Comfortably inside the full diagonal of the parallelepiped, which is
    # how far a corner-origin wrap can throw a molecule.
    assert radii.max() < np.linalg.norm(CELL.sum(axis=0))


def test_a_molecule_is_moved_whole():
    """Translating atoms independently would tear a water apart, and the
    tear would only show up as a bad bond in whatever read the file."""

    solute = np.array([[0.0, 0.0, 0.0]])
    water = np.array([[0.0, 0.0, 0.0], [0.096, 0.0, 0.0], [-0.03, 0.09, 0.0]])
    positions = np.vstack([solute, water + CELL[2]])

    imaged = image_molecules_about_solute(positions, CELL, [[0], [1, 2, 3]], [0])

    before = positions[[1, 2, 3]] - positions[1]
    after = imaged[[1, 2, 3]] - imaged[1]
    assert np.allclose(before, after)


def test_a_molecule_already_nearest_is_left_exactly_alone():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])

    imaged = image_molecules_about_solute(positions, CELL, [[0], [1]], [0])

    assert np.allclose(imaged[1], [1.0, 2.0, 3.0])


def test_a_cubic_box_is_handled_too():
    """Not every manifest asks for an octahedron."""

    cubic = np.diag([50.0, 50.0, 50.0])
    positions = np.array([[0.0, 0.0, 0.0], [49.0, 1.0, 1.0]])

    imaged = image_molecules_about_solute(positions, cubic, [[0], [1]], [0])

    assert np.allclose(imaged[1], [-1.0, 1.0, 1.0])


def test_the_input_is_not_modified():
    positions = np.array([[0.0, 0.0, 0.0], [90.0, 0.0, 0.0]])
    original = positions.copy()

    image_molecules_about_solute(positions, CELL, [[0], [1]], [0])

    assert np.allclose(positions, original)


def test_a_bad_cell_is_refused():
    with pytest.raises(ValueError, match="3x3"):
        image_molecules_about_solute(np.zeros((1, 3)), np.eye(2), [[0]], [0])


def test_no_solute_is_refused_rather_than_centred_on_nothing():
    with pytest.raises(ValueError, match="No solute"):
        image_molecules_about_solute(np.zeros((1, 3)), CELL, [[0]], [])


def test_water_and_the_ions_tleap_adds_are_solvent_and_a_ligand_is_not():
    solvent = solvent_residue_names()

    assert {"WAT", "HOH", "NA+", "CL-"} <= solvent
    assert "LIG" not in solvent
    assert "LG2" not in solvent


def test_the_solute_is_everything_that_is_not_added_solvent():
    class _Residue:
        def __init__(self, name):
            self.name = name

    class _Atom:
        def __init__(self, index, residue):
            self.index = index
            self.residue = residue

    class _Topology:
        def __init__(self, names):
            self._atoms = [_Atom(i, _Residue(n)) for i, n in enumerate(names)]

        def atoms(self):
            return iter(self._atoms)

    topology = _Topology(["ALA", "LG2", "WAT", "WAT", "Na+", "Cl-", "FE1"])

    assert solute_atom_indices(topology) == [0, 1, 6]


def test_a_system_of_nothing_but_water_still_centres_on_something():
    class _T:
        def atoms(self):
            class A:
                def __init__(self, i):
                    self.index = i
                    self.residue = type("R", (), {"name": "WAT"})()

            return iter([A(0), A(1)])

    assert solute_atom_indices(_T()) == [0, 1]
