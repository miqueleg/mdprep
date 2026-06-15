import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from tests.test_ligand_workflow_mocked import (
    fake_antechamber,
    fake_parmchk2,
    fake_point_charges,
    fake_pyscf_derivation,
    fake_tleap,
    qmmesp_block,
)
from tests.test_prepare_tleap_stage import fake_validation
from tests.test_structure_normalize import ligand_entry, manifest_data
from tests.test_tleap_workflow_mocked import fake_tleap as fake_final_tleap


def write_pyscf_manifest(tmp_path, *, method="gas_resp_pyscf", output_name="prepared"):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / output_name)
    data["structure"]["remove_unknown_heterogens"] = True
    data["protonation"]["method"] = "manual_only"
    data["solvation"]["enabled"] = False
    data["solvation"]["salt_concentration_molar"] = 0.0
    data["validation"]["run_openmm_energy_check"] = False
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": method,
            "qmmesp": qmmesp_block(),
        }
    ]
    manifest = tmp_path / f"{method}.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest, tmp_path / output_name


def patch_pyscf_ligand_tools(monkeypatch):
    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)
    monkeypatch.setattr("mdprep.ligands.workflow.derive_pyscf_charges", fake_pyscf_derivation)


def test_prepare_ligands_stage_with_mocked_gas_resp_pyscf(monkeypatch, tmp_path):
    manifest, output_dir = write_pyscf_manifest(tmp_path, method="gas_resp_pyscf")
    patch_pyscf_ligand_tools(monkeypatch)

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "ligands"])

    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    ligand = report["ligands"][0]
    assert ligand["charge_method"] == "gas_resp_pyscf"
    assert ligand["qm"]["fit_result"]["charge_sum_final"] == 0.0
    assert Path(ligand["final_mol2_path"]).exists()


def test_prepare_ligands_stage_with_mocked_qmmesp_pyscf(monkeypatch, tmp_path):
    manifest, output_dir = write_pyscf_manifest(tmp_path, method="qmmesp_pyscf")
    patch_pyscf_ligand_tools(monkeypatch)
    monkeypatch.setattr("mdprep.ligands.workflow.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.ligands.workflow.extract_point_charges_from_prmtop", fake_point_charges)

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "ligands"])

    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    ligand = report["ligands"][0]
    assert ligand["charge_method"] == "qmmesp_pyscf"
    assert ligand["qm"]["embedding_summary"]["point_charge_count_after_cutoff"] == 1
    assert Path(ligand["final_mol2_path"]).exists()


def test_prepare_tleap_uses_mocked_qmmesp_final_mol2(monkeypatch, tmp_path):
    manifest, output_dir = write_pyscf_manifest(tmp_path, method="qmmesp_pyscf")
    patch_pyscf_ligand_tools(monkeypatch)
    monkeypatch.setattr("mdprep.ligands.workflow.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.ligands.workflow.extract_point_charges_from_prmtop", fake_point_charges)
    monkeypatch.setattr("mdprep.leap.builder.run_tleap", fake_final_tleap)
    monkeypatch.setattr("mdprep.workflows.prepare.validate_final_outputs", fake_validation)

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])

    assert result.exit_code == 0, result.output
    ligand_report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    tleap_script = output_dir / "leap" / "dry" / "tleap.in"
    assert ligand_report["ligands"][0]["final_mol2_path"] in tleap_script.read_text(encoding="utf-8")
    assert (output_dir / "final" / "system.prmtop").exists()
