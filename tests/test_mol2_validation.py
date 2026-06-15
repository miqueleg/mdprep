import csv

import pytest

from mdprep.ambertools.mol2 import Mol2Error, read_mol2, validate_and_write_final_mol2
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_config(**updates):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    entry.update(updates)
    data["ligands"] = [entry]
    return make_manifest(data).ligands[0]


def extracted_atoms():
    structure = read_pdb("tests/data/ligands/ligand_sub.pdb")
    return structure.residues[0].atoms


def validate(tmp_path, mol2_path, **updates):
    return validate_and_write_final_mol2(
        mol2_path=mol2_path,
        extracted_atoms=extracted_atoms(),
        ligand=ligand_config(**updates),
        final_mol2_path=tmp_path / "final.mol2",
        charges_csv_path=tmp_path / "charges.csv",
        validation_json_path=tmp_path / "validation.json",
    )


def test_good_mol2_validates_against_extracted_ligand(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.good.mol2")

    assert result.charge_sum_final == pytest.approx(0.0)
    assert result.coordinate_max_deviation == pytest.approx(0.0)


def test_small_charge_residual_is_corrected_and_recorded(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.small_residual.mol2")
    final = read_mol2(result.final_mol2_path)

    assert result.charge_correction_applied == pytest.approx(-0.005)
    assert final.total_charge == pytest.approx(0.0)
    with result.charges_csv_path.open("r", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))


def test_large_charge_mismatch_fails(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.bad_charge.mol2")

    assert "charge sum" in str(excinfo.value)


def test_atom_count_mismatch_fails(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.atom_count_mismatch.mol2")

    assert "atom-count mismatch" in str(excinfo.value)


def test_coordinate_deviation_fails_when_not_allowed(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.shifted_coords.mol2")

    assert "coordinates deviate" in str(excinfo.value)


def test_renamed_atoms_are_restored_when_order_is_compatible(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.renamed.mol2")
    final = read_mol2(result.final_mol2_path)

    assert [atom.name for atom in final.atoms] == ["C1", "O1"]
    assert result.atom_names_preserved
