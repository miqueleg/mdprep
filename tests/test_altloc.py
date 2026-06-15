from pathlib import Path

import pytest

from mdprep.structure.pdb import PdbParseError, read_pdb


DATA = Path("tests/data")


def test_highest_occupancy_selects_expected_atom():
    structure = read_pdb(DATA / "protein_altloc.pdb", altloc_policy="highest_occupancy")
    ca = next(atom for atom in structure.atoms if atom.name == "CA")

    assert ca.altloc == "B"
    assert ca.x == 2.0


def test_first_selects_first_altloc_atom():
    structure = read_pdb(DATA / "protein_altloc.pdb", altloc_policy="first")
    ca = next(atom for atom in structure.atoms if atom.name == "CA")

    assert ca.altloc == "A"
    assert ca.x == 1.0


def test_fail_rejects_unresolved_altlocs():
    with pytest.raises(PdbParseError) as excinfo:
        read_pdb(DATA / "protein_altloc.pdb", altloc_policy="fail")

    assert "Unresolved alternate locations" in str(excinfo.value)

