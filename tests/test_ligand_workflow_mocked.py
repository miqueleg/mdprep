from pathlib import Path

import pytest

from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.external.runner import CommandResult
from mdprep.ligands.workflow import LigandWorkflowError, run_ligand_stage
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def command_run(output_path: Path, name: str) -> AmberToolRun:
    return AmberToolRun(
        command_result=CommandResult(
            command=(name, "fake"),
            cwd=str(output_path.parent),
            returncode=0,
            stdout="",
            stderr="",
            runtime_seconds=0.01,
        ),
        stdout_path=output_path.parent / f"{name}_stdout.txt",
        stderr_path=output_path.parent / f"{name}_stderr.txt",
        output_path=output_path,
    )


def manifest_with_ligand(entry: dict):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["structure"]["remove_unknown_heterogens"] = True
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [entry]
    return make_manifest(data)


def normalized_structure(manifest):
    return normalize_structure_stage(manifest).normalized_structure


def test_am1bcc_workflow_writes_parameter_files_with_fake_tools(monkeypatch, tmp_path):
    entry = ligand_entry("sub_501", "B", "SUB", 501)
    manifest = manifest_with_ligand(entry)

    def fake_antechamber(*, ligand, input_pdb, output_mol2, residue_name, work_dir):
        output = Path(output_mol2)
        output.write_text(Path("tests/data/ligands/ligand_sub.good.mol2").read_text(encoding="utf-8"), encoding="utf-8")
        return command_run(output, "antechamber")

    def fake_parmchk2(*, ligand, input_mol2, output_frcmod, work_dir):
        output = Path(output_frcmod)
        output.write_text(Path("tests/data/ligands/ligand_sub.frcmod").read_text(encoding="utf-8"), encoding="utf-8")
        return command_run(output, "parmchk2")

    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    item = result.ligands[0]
    assert item.final_mol2_path and item.final_mol2_path.exists()
    assert item.final_frcmod_path and item.final_frcmod_path.exists()
    assert item.validation and item.validation.validation_json_path.exists()
    assert item.validation.charges_csv_path.exists()


def test_user_mol2_workflow_copies_and_validates_mol2(tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    manifest = manifest_with_ligand(entry)

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    item = result.ligands[0]
    assert item.final_mol2_path and item.final_mol2_path.exists()
    assert item.final_frcmod_path and item.final_frcmod_path.exists()
    assert item.parmchk2 is None


def test_user_frcmod_is_copied_and_parmchk2_is_skipped(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr(
        "mdprep.ligands.workflow.run_parmchk2",
        lambda **kwargs: (_ for _ in ()).throw(AmberToolsError("should not run")),
    )

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert result.ligands[0].final_frcmod_path is not None


def test_user_mol2_without_user_frcmod_fails_if_parmchk2_unavailable(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr(
        "mdprep.ligands.workflow.run_parmchk2",
        lambda **kwargs: (_ for _ in ()).throw(AmberToolsError("AmberTools executable not found: parmchk2")),
    )

    with pytest.raises(LigandWorkflowError) as excinfo:
        run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert "parmchk2" in str(excinfo.value)


@pytest.mark.parametrize("method", ["gas_resp_pyscf", "qmmesp_pyscf"])
def test_pyscf_charge_methods_fail_clearly(method, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": method,
    }
    if method == "qmmesp_pyscf":
        entry["qmmesp"] = {
            "qm_engine": "pyscf",
            "method": "HF",
            "basis": "6-31G*",
            "embedding_cutoff_angstrom": 12.0,
            "resp_fitting": {"backend": "auto", "total_charge_constraint": True, "stage_2": True},
        }
    manifest = manifest_with_ligand(entry)

    with pytest.raises(LigandWorkflowError) as excinfo:
        run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert "PySCF RESP/QMMESP ligand charges are not implemented yet" in str(excinfo.value)
