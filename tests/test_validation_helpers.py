import pytest

from mdprep.validation.topology import FinalValidationError, validate_final_outputs
from tests.test_structure_normalize import make_manifest, manifest_data


def test_missing_final_files_fail(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    manifest = make_manifest(data)

    with pytest.raises(FinalValidationError) as excinfo:
        validate_final_outputs(
            manifest=manifest,
            prmtop=tmp_path / "missing.prmtop",
            inpcrd=tmp_path / "missing.inpcrd",
            pdb=tmp_path / "missing.pdb",
        )

    assert "Missing final prmtop" in str(excinfo.value)


def test_zero_size_final_file_fails(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    manifest = make_manifest(data)
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    pdb = tmp_path / "system.pdb"
    prmtop.write_text("", encoding="utf-8")
    inpcrd.write_text("x", encoding="utf-8")
    pdb.write_text("END\n", encoding="utf-8")

    with pytest.raises(FinalValidationError) as excinfo:
        validate_final_outputs(manifest=manifest, prmtop=prmtop, inpcrd=inpcrd, pdb=pdb)

    assert "empty" in str(excinfo.value)
