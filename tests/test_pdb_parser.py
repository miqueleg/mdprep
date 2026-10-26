from pathlib import Path

import pytest

from mdprep.structure.inspect import inspect_pdb_structure
from mdprep.structure.pdb import PdbParseError, infer_element, read_pdb


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
    assert (
        infer_element(
            "FE",
            resname="FE",
            record_name="HETATM",
            atom_field=" FE ",
        )
        == "Fe"
    )


def test_infers_ambertools_atom_record_iron_when_element_column_missing():
    assert (
        infer_element(
            "FE",
            resname="FE",
            record_name="ATOM",
            atom_field=" FE ",
        )
        == "Fe"
    )


def test_infers_iron_for_an_amber_ion_residue_named_for_its_oxidation_state():
    """Amber names an ion residue after its oxidation state: FE2 is Fe(II).

    tleap then writes the atom as " FE " in a file with no element column, and
    by the column convention a two-letter symbol starts in column 13 -- so
    " FE " is literally fluorine with remoteness E. The residue name settles
    it, but only once reduced to its symbol. Before that, a prepared
    structure's iron came back as fluorine and the refinement mapping could
    not match the metal to itself.
    """

    for resname in ("FE2", "FE3"):
        assert (
            infer_element("FE", resname=resname, record_name="ATOM", atom_field=" FE ")
            == "Fe"
        )
    assert (
        infer_element("CU", resname="CU1", record_name="HETATM", atom_field=" CU ")
        == "Cu"
    )
    assert (
        infer_element("YB", resname="YB2", record_name="HETATM", atom_field=" YB ")
        == "Yb"
    )


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


def _unlabelled_alternates(tmp_path, occupancy: float) -> Path:
    """An aspartate's OD2 written twice, as a format converter writes two conformers."""
    lines = [
        f"ATOM      1  OD2 ASP A 265      -8.289  52.710  -3.481  {occupancy:4.2f}  0.00           O",
        f"ATOM      2  OD2 ASP A 265      -9.909  51.341  -3.371  {occupancy:4.2f}  0.00           O",
        "END",
    ]
    path = tmp_path / "unlabelled.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("policy", ["highest_occupancy", "first"])
def test_copies_at_partial_occupancy_without_labels_are_read_as_alternates(tmp_path, policy):
    """PDBFixer used to keep one of the two without saying which."""
    structure = read_pdb(_unlabelled_alternates(tmp_path, 0.50), altloc_policy=policy)

    assert [(atom.name, atom.x) for atom in structure.atoms] == [("OD2", -8.289)]
    assert any("A:ASP265 OD2" in warning for warning in structure.warnings)


def test_unlabelled_alternates_still_fail_under_the_fail_policy(tmp_path):
    with pytest.raises(PdbParseError, match="alternate locations"):
        read_pdb(_unlabelled_alternates(tmp_path, 0.50), altloc_policy="fail")


def test_copies_at_full_occupancy_are_left_as_written(tmp_path):
    """Those are different atoms with one name -- a converter's hydrogens, all H."""
    structure = read_pdb(_unlabelled_alternates(tmp_path, 1.00))

    assert len(structure.atoms) == 2
    assert not any("alternate locations" in warning for warning in structure.warnings)
