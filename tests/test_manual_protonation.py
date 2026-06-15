import pytest

from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_structure_normalize import make_manifest, manifest_data


def override(chain: str, resname: str, resid: int, state: str, reason: str = "test") -> dict:
    return {
        "selector": {"chain": chain, "resname": resname, "resid": resid, "icode": None},
        "state": state,
        "reason": reason,
    }


def protonate(data: dict):
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    return apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )


def test_manual_asp_to_ash_renames_residue():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["protonation"]["overrides"] = [override("A", "ASP", 3, "ASH")]

    result = protonate(data)

    assert "ASH" in [residue.id.resname for residue in result.structure.residues]
    assert result.manual_overrides_applied[0].changed


def test_manual_his_to_hie_renames_residue():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["structure"]["remove_input_hydrogens"] = False
    data["protonation"]["overrides"] = [override("A", "HIS", 2, "HIE")]

    result = protonate(data)

    assert "HIE" in [residue.id.resname for residue in result.structure.residues]


def test_manual_cys_to_cym_renames_residue():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["disulfides"]["auto_detect"] = False
    data["protonation"]["overrides"] = [override("A", "CYS", 30, "CYM")]

    result = protonate(data)

    assert "CYM" in [residue.id.resname for residue in result.structure.residues]


def test_explicit_same_state_assignment_is_recorded():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["protonation"]["overrides"] = [override("A", "ASP", 3, "ASP")]

    result = protonate(data)

    record = result.manual_overrides_applied[0]
    assert record.original_resname == "ASP"
    assert record.final_resname == "ASP"
    assert not record.changed


def test_missing_selector_fails():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["protonation"]["overrides"] = [override("A", "ASP", 99, "ASH")]

    with pytest.raises(ProtonationApplicationError) as excinfo:
        protonate(data)

    assert "did not resolve exactly one residue" in str(excinfo.value)


def test_incompatible_override_fails_clearly():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["protonation"]["overrides"] = [override("A", "ASP", 3, "HIE")]

    with pytest.raises(ProtonationApplicationError) as excinfo:
        protonate(data)

    message = str(excinfo.value)
    assert "current residue ASP" in message
    assert "requested HIE" in message


def test_atom_names_and_coordinates_are_preserved():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["protonation"]["method"] = "manual_only"
    data["structure"]["remove_input_hydrogens"] = False
    data["protonation"]["overrides"] = [override("A", "HIS", 2, "HIE")]
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)

    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    assert [atom.name for atom in result.structure.atoms] == [atom.name for atom in normalized.normalized_structure.atoms]
    assert [(atom.x, atom.y, atom.z) for atom in result.structure.atoms] == [
        (atom.x, atom.y, atom.z) for atom in normalized.normalized_structure.atoms
    ]
