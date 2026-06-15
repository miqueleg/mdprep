from pathlib import Path

import pytest

from mdprep.structure.pdb import read_pdb
from mdprep.structure.selectors import (
    SelectorError,
    parse_atom_selector,
    parse_residue_selector,
    resolve_atom_selector,
    resolve_residue_selector,
)


DATA = Path("tests/data")


def test_parse_residue_selector_with_resname():
    selector = parse_residue_selector("A:HIS64")

    assert selector.chain_id == "A"
    assert selector.resname == "HIS"
    assert selector.resid == 64


def test_parse_residue_selector_without_resname():
    selector = parse_residue_selector("A:64")

    assert selector.chain_id == "A"
    assert selector.resname is None
    assert selector.resid == 64


def test_parse_blank_chain_selectors():
    selector = parse_residue_selector(":HIS64")
    atom_selector = parse_atom_selector(":64@ND1")

    assert selector.chain_id == ""
    assert selector.resname == "HIS"
    assert atom_selector.residue.chain_id == ""
    assert atom_selector.atom_name == "ND1"


def test_parse_atom_selector():
    selector = parse_atom_selector("A:CYS45@SG")

    assert selector.residue.chain_id == "A"
    assert selector.residue.resname == "CYS"
    assert selector.residue.resid == 45
    assert selector.atom_name == "SG"


def test_structured_selector_resolves_correct_residue():
    structure = read_pdb(DATA / "protein_with_waters.pdb")
    residue = resolve_residue_selector(
        structure,
        {"chain": "A", "resname": "ALA", "resid": 1, "icode": None},
    )

    assert residue.id.resname == "ALA"
    assert residue.atoms[0].name == "N"


def test_atom_selector_resolves_correct_atom():
    structure = read_pdb(DATA / "protein_disulfide.pdb")
    atom = resolve_atom_selector(structure, "A:CYS10@SG")

    assert atom.name == "SG"
    assert atom.resid == 10


def test_missing_selector_raises_selector_error():
    structure = read_pdb(DATA / "protein_with_waters.pdb")

    with pytest.raises(SelectorError) as excinfo:
        resolve_residue_selector(structure, "A:HIS99")

    assert "No residue matched selector" in str(excinfo.value)


def test_ambiguous_selector_raises_selector_error():
    structure = read_pdb(DATA / "protein_with_waters.pdb")

    with pytest.raises(SelectorError) as excinfo:
        resolve_residue_selector(structure, "A:1")

    assert "matched 2 residues" in str(excinfo.value)

