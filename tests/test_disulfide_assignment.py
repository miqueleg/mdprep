import pytest

from mdprep.protonation.apply import ProtonationApplicationError
from tests.test_manual_protonation import override, protonate
from tests.test_structure_normalize import manifest_data


def test_auto_detected_disulfide_renames_both_cys_to_cyx():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    result = protonate(data)

    residues = {(residue.id.resid, residue.id.resname) for residue in result.structure.residues}
    assert (10, "CYX") in residues
    assert (20, "CYX") in residues
    assert (30, "CYS") in residues


def test_forced_disulfide_renames_both_residues_to_cyx():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["disulfides"]["auto_detect"] = False
    data["disulfides"]["force"] = [
        {
            "a": {"chain": "A", "resname": "CYS", "resid": 10, "icode": None},
            "b": {"chain": "A", "resname": "CYS", "resid": 20, "icode": None},
            "reason": "manual pair",
        }
    ]

    result = protonate(data)

    assert [record.source for record in result.disulfide_assignments_applied] == [
        "forced_disulfide",
        "forced_disulfide",
    ]
    assert {record.final_resname for record in result.disulfide_assignments_applied} == {"CYX"}


def test_forbidden_disulfide_prevents_auto_assignment():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["disulfides"]["forbid"] = [
        {
            "a": {"chain": "A", "resname": "CYS", "resid": 10, "icode": None},
            "b": {"chain": "A", "resname": "CYS", "resid": 20, "icode": None},
            "reason": "do not assign",
        }
    ]

    result = protonate(data)

    assert [residue.id.resname for residue in result.structure.residues] == ["CYS", "CYS", "CYS"]
    assert result.disulfide_assignments_applied == []


def test_manual_cym_override_conflicts_with_disulfide_assignment():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["protonation"]["overrides"] = [override("A", "CYS", 10, "CYM")]

    with pytest.raises(ProtonationApplicationError) as excinfo:
        protonate(data)

    assert "requires CYX" in str(excinfo.value)


def test_already_cyx_disulfide_is_accepted_and_reported():
    data = manifest_data("tests/data/protein_cyx_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"

    result = protonate(data)

    assert len(result.disulfide_assignments_applied) == 2
    assert {record.final_resname for record in result.disulfide_assignments_applied} == {"CYX"}
    assert {record.changed for record in result.disulfide_assignments_applied} == {False}
