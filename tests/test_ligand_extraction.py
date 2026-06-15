import pytest

from mdprep.ligands.extract import LigandExtractionError, extract_configured_ligands
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_manifest(input_structure: str):
    data = manifest_data(input_structure)
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
            "charge_method": "user_mol2",
        }
    ]
    return make_manifest(data)


def test_extract_configured_ligand_preserves_identity(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    manifest = ligand_manifest("tests/data/protein_two_ligands.pdb")

    extracted = extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert len(extracted) == 1
    ligand = extracted[0]
    assert [atom.name for atom in ligand.atoms] == ["C1", "O1"]
    assert [(atom.x, atom.y, atom.z) for atom in ligand.atoms] == [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0)]
    assert ligand.residue.id.resname == "SUB"
    assert ligand.residue.id.chain_id == "B"
    assert ligand.pdb_path.exists()
    assert ligand.identity_path.exists()


def test_missing_ligand_selector_fails_clearly(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("missing", "B", "SUB", 999)]
    manifest = make_manifest(data)

    with pytest.raises(LigandExtractionError) as excinfo:
        extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert "selector did not resolve exactly one residue" in str(excinfo.value)


def test_multiple_ligands_are_extracted_independently(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        ligand_entry("sub_501", "B", "SUB", 501),
        ligand_entry("cof_601", "C", "COF", 601),
    ]
    manifest = make_manifest(data)

    extracted = extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert [ligand.config.id for ligand in extracted] == ["sub_501", "cof_601"]
    assert [atom.name for atom in extracted[0].atoms] == ["C1", "O1"]
    assert [atom.name for atom in extracted[1].atoms] == ["N1", "C1"]
