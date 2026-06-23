from pathlib import Path

import pytest

from mdprep.structure.inspect import inspect_pdb_structure
from mdprep.structure.pdb import infer_element, read_pdb


DATA = Path("tests/data")


def test_parses_atoms_residues_and_preserves_order_and_names():
    structure = read_pdb(DATA / "protein_with_waters.pdb")

    assert len(structure.atoms) == 18
    assert len(structure.residues) == 5
    assert [atom.serial for atom in structure.atoms[:3]] == [1, 2, 3]
    assert [atom.name for atom in structure.atoms[:5]] == ["N", "CA", "C", "O", "CB"]
    assert structure.residues[0].id.resname == "ALA"
    assert structure.residues[-1].id.resname == "SO4"


def test_counts_waters_and_likely_ligands():
    summary = inspect_pdb_structure(DATA / "protein_with_waters.pdb")

    assert len(summary.water_residues) == 1
    assert len(summary.likely_ligands) == 1
    assert summary.likely_ligands[0].id.resname == "SO4"


def test_atom_record_nonstandard_blank_chain_ligand_is_parsed_and_detected():
    summary = inspect_pdb_structure(DATA / "protein_atom_record_ligand_blank_chain.pdb")
    ligand = summary.likely_ligands[0]

    assert ligand.id.chain_id == ""
    assert ligand.id.resname == "5NB"
    assert ligand.id.resid == 301
    assert ligand.atom_names() == ["C1x", "N1x", "O1x", "C2x"]
    assert "ATOM" in ligand.record_names


def test_handles_blank_chain_ids():
    structure = read_pdb(DATA / "protein_blank_chain.pdb")

    assert structure.residues[0].id.chain_id == ""
    assert structure.residues[0].id.resname == "HIS"


def test_handles_insertion_codes():
    structure = read_pdb(DATA / "protein_insertion_code.pdb")

    assert structure.residues[0].id.icode == "A"
    assert structure.residues[0].id.resid == 10


def test_reports_model_count_and_uses_first_model():
    structure = read_pdb(DATA / "protein_with_waters.pdb")

    assert structure.model_count == 2
    assert structure.used_model == 1
    assert structure.atoms[-1].serial == 18
    assert "using MODEL 1 only" in structure.warnings[0]


def test_infers_alpha_carbon_as_carbon_for_standard_protein_atoms():
    assert infer_element("CA", resname="ALA", record_name="ATOM") == "C"
    assert infer_element("CD1", resname="ILE", record_name="ATOM") == "C"
    assert infer_element("NE2", resname="HIS", record_name="ATOM") == "N"
    assert infer_element("SG", resname="CYS", record_name="ATOM") == "S"


def test_infers_calcium_for_heterogen_ca_when_element_column_missing():
    assert infer_element("CA", resname="CA", record_name="HETATM") == "Ca"


def test_pdb_atom_name_alignment_disambiguates_blank_element_ca():
    assert infer_element("CA", resname="LIG", record_name="HETATM", atom_field=" CA ") == "C"
    assert infer_element("CA", resname="CA", record_name="HETATM", atom_field="CA  ") == "Ca"


@pytest.mark.parametrize(
    ("atom_name", "atom_field", "element"),
    [
        ("CL1", "CL1 ", "Cl"),
        ("BR1", "BR1 ", "Br"),
        ("I1", " I1 ", "I"),
        ("FE", "FE  ", "Fe"),
        ("AU", "AU  ", "Au"),
        ("ZN", "ZN  ", "Zn"),
    ],
)
def test_pdb_blank_element_fallback_handles_common_ligand_elements(atom_name, atom_field, element):
    assert infer_element(atom_name, resname="LIG", record_name="HETATM", atom_field=atom_field) == element
