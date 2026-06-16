import json
from pathlib import Path

from typer.testing import CliRunner

from mdprep.cli import app
from tests.test_ligand_workflow_mocked import fake_point_charges, fake_pyscf_derivation, fake_tleap
from tests.test_prepare_pyscf_ligands_stage import _loadmol2_path, write_pyscf_manifest
from tests.test_prepare_tleap_stage import fake_validation
from tests.test_qmmesp_correctness import fake_qmmesp_antechamber, fake_qmmesp_parmchk2
from tests.test_tleap_workflow_mocked import fake_tleap as fake_final_tleap


def test_final_tleap_uses_final_fitted_mol2_not_provisional(monkeypatch, tmp_path):
    manifest, output_dir = write_pyscf_manifest(tmp_path, method="qmmesp_pyscf")
    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_qmmesp_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_qmmesp_parmchk2)
    monkeypatch.setattr("mdprep.ligands.workflow.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.ligands.workflow.extract_point_charges_from_prmtop", fake_point_charges)
    monkeypatch.setattr("mdprep.ligands.workflow.derive_pyscf_charges", fake_pyscf_derivation)
    monkeypatch.setattr("mdprep.leap.builder.run_tleap", fake_final_tleap)
    monkeypatch.setattr("mdprep.workflows.prepare.validate_final_outputs", fake_validation)

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])

    assert result.exit_code == 0, result.output
    ligand_report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    ligand = ligand_report["ligands"][0]
    final_mol2 = ligand["final_mol2_path"]
    provisional_mol2 = ligand["provisional_mol2_path"]
    tleap_script_path = Path(output_dir / "leap" / "dry" / "tleap.in")
    tleap_script = tleap_script_path.read_text(encoding="utf-8")
    lock = json_like_yaml(output_dir / "manifest.lock.yaml")

    assert (tleap_script_path.parent / _loadmol2_path(tleap_script_path)).resolve() == Path(final_mol2).resolve()
    assert provisional_mol2 not in tleap_script
    assert lock["resolved"]["ligands"][0]["final_mol2_path"] == final_mol2
    assert lock["resolved"]["ligands"][0]["provisional_mol2_path"] == provisional_mol2


def json_like_yaml(path):
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
