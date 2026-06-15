import json

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from tests.test_prepare_ligands_stage import write_ligand_manifest
from tests.test_tleap_workflow_mocked import fake_tleap


def fake_validation(*, manifest, prmtop, inpcrd, pdb):
    return {
        "final_prmtop_path": str(prmtop),
        "final_inpcrd_path": str(inpcrd),
        "final_pdb_path": str(pdb),
        "file_checks": {},
        "final_atom_count": 2,
        "final_residue_count": 1,
        "ligand_presence_checks": [],
        "water_presence": {"water_count": 0, "solvation_enabled": manifest.solvation.enabled},
        "parmed": {"status": "skipped"},
        "openmm": {"status": "disabled"},
        "warnings": [],
        "errors": [],
    }


def write_tleap_manifest(tmp_path, *, output_name="prepared"):
    manifest, output_dir = write_ligand_manifest(tmp_path, output_name=output_name)
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    data["solvation"]["enabled"] = False
    data["solvation"]["salt_concentration_molar"] = 0.0
    data["validation"]["run_openmm_energy_check"] = False
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest, output_dir


def test_prepare_tleap_stage_creates_expected_files(monkeypatch, tmp_path):
    manifest, output_dir = write_tleap_manifest(tmp_path)
    monkeypatch.setattr("mdprep.leap.builder.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.workflows.prepare.validate_final_outputs", fake_validation)

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])

    assert result.exit_code == 0, result.output
    for path in [
        output_dir / "final" / "system.prmtop",
        output_dir / "final" / "system.inpcrd",
        output_dir / "final" / "system.pdb",
        output_dir / "reports" / "tleap_report.json",
        output_dir / "reports" / "tleap_report.md",
        output_dir / "reports" / "validation_report.json",
        output_dir / "reports" / "validation_report.md",
    ]:
        assert path.exists(), path
    report = json.loads((output_dir / "reports" / "tleap_report.json").read_text(encoding="utf-8"))
    assert report["final_outputs"]["prmtop"].endswith("system.prmtop")


def test_full_prepare_defaults_to_tleap(monkeypatch, tmp_path):
    manifest, output_dir = write_tleap_manifest(tmp_path)
    monkeypatch.setattr("mdprep.leap.builder.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.workflows.prepare.validate_final_outputs", fake_validation)

    result = CliRunner().invoke(app, ["prepare", str(manifest)])

    assert result.exit_code == 0, result.output
    assert (output_dir / "final" / "system.prmtop").exists()


def test_previous_stop_stages_still_work(tmp_path):
    manifest, output_dir = write_tleap_manifest(tmp_path)
    structure = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "structure"])
    assert structure.exit_code == 0
    protonation = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "protonation", "--overwrite"])
    assert protonation.exit_code == 0
    assert (output_dir / "intermediate" / "01_protonation_assigned.pdb").exists()
