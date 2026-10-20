"""1-4 pairs through a metal with nothing to stop them collapsing.

SyrB2 (PDB 2FCT) binds a chloride and a water on the same iron. Cl-Fe-O-H
makes the chloride and the water's hydrogen a 1-4 pair; MCPB.py zeroes the
torsions through the metal, O...Cl is excluded as 1-3 through the iron, and a
TIP3P hydrogen has no Lennard-Jones radius. The first minimisation put that
hydrogen 0.14 A from the chloride and the heating step went to NaN.

These build a five-atom topology in memory with ParmEd's raw format, so they
need ParmEd but no AmberTools.
"""

import pytest

pytest.importorskip("parmed")
from parmed.amber import AmberFormat  # noqa: E402

from mdprep.metals.through_metal import switch_off_unshielded_through_metal_14  # noqa: E402

pytestmark = [pytest.mark.parmed]

# Atoms: 0 Cl (res CL1), 1 Fe (res FE1), 2 O (res WT1), 3 H (res WT1, no LJ),
# 4 N (res HD1), 5 C (res HD1). LJ types: 1 shielded, 2 unshielded.
NAMES = ["CL", "FE", "O", "H1", "NE2", "CE1"]
RESIDUES = ["CL1", "FE1", "WT1", "HD1"]
RESIDUE_POINTER = [1, 2, 3, 5]
ATOMIC_NUMBER = [17, 26, 8, 1, 7, 6]
TYPE_INDEX = [1, 1, 1, 2, 1, 1]


def _dihedral(a, b, c, d, kind=1):
    return [3 * a, 3 * b, 3 * c, 3 * d, kind]


def _write(path, inc_h, without_h):
    parm = AmberFormat()
    pointers = [0] * 31
    pointers[0], pointers[1] = len(NAMES), 2
    parm.add_flag("POINTERS", "10I8", data=pointers)
    parm.add_flag("ATOM_NAME", "20a4", data=NAMES)
    parm.add_flag("ATOMIC_NUMBER", "10I8", data=ATOMIC_NUMBER)
    parm.add_flag("ATOM_TYPE_INDEX", "10I8", data=TYPE_INDEX)
    # 2 types -> 3 unique pairs; (2,2) has a zero A coefficient.
    parm.add_flag("NONBONDED_PARM_INDEX", "10I8", data=[1, 2, 2, 3])
    parm.add_flag("LENNARD_JONES_ACOEF", "5E16.8", data=[1.0e6, 0.0, 0.0])
    parm.add_flag("LENNARD_JONES_BCOEF", "5E16.8", data=[1.0e3, 0.0, 0.0])
    parm.add_flag("RESIDUE_LABEL", "20a4", data=RESIDUES)
    parm.add_flag("RESIDUE_POINTER", "10I8", data=RESIDUE_POINTER)
    parm.add_flag("DIHEDRALS_INC_HYDROGEN", "10I8", data=inc_h)
    parm.add_flag("DIHEDRALS_WITHOUT_HYDROGEN", "10I8", data=without_h)
    parm.write_parm(str(path))


def test_the_water_hydrogen_and_the_chloride_stop_attracting(tmp_path):
    path = tmp_path / "site.prmtop"
    # Cl-Fe-O-H: through the metal, different residues, H unshielded.
    _write(path, _dihedral(0, 1, 2, 3), [])

    switched = switch_off_unshielded_through_metal_14(path)

    assert [(pair.first, pair.metal, pair.last) for pair in switched] == [
        ("CL1 1:CL", "FE1 2:FE", "WT1 3:H1")
    ]
    flags = AmberFormat(str(path)).parm_data["DIHEDRALS_INC_HYDROGEN"]
    assert flags[2] < 0, "the third index carries the 'ignore 1-4' flag"


def test_a_pair_with_repulsion_on_both_sides_is_left_alone(tmp_path):
    """Cl-Fe-NE2-CE1 crosses the metal too, but both ends have a radius, so
    the 1-4 term has a wall under it and stays as tleap wrote it."""

    path = tmp_path / "site.prmtop"
    _write(path, [], _dihedral(0, 1, 4, 5))

    assert switch_off_unshielded_through_metal_14(path) == []
    assert AmberFormat(str(path)).parm_data["DIHEDRALS_WITHOUT_HYDROGEN"][2] > 0


def test_a_dihedral_that_does_not_cross_a_metal_is_left_alone(tmp_path):
    """An unshielded hydrogen in an ordinary torsion is the usual Amber case,
    held away by the torsion and its own residue's geometry."""

    path = tmp_path / "site.prmtop"
    # NE2-CE1-... a dihedral with no metal in the middle positions.
    _write(path, _dihedral(4, 5, 2, 3), [])

    assert switch_off_unshielded_through_metal_14(path) == []


def test_a_topology_without_a_metal_is_not_rewritten(tmp_path):
    path = tmp_path / "site.prmtop"
    _write(path, _dihedral(0, 1, 2, 3), [])
    data = AmberFormat(str(path))
    data.parm_data["ATOMIC_NUMBER"][1] = 6
    data.write_parm(str(path))
    before = path.read_bytes()

    assert switch_off_unshielded_through_metal_14(path) == []
    assert path.read_bytes() == before

