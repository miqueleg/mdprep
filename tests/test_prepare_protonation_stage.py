import csv
import json

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.protonation.propka import PropkaExecutionError
from mdprep.structure.pdb import read_pdb
from tests.test_manual_protonation import override
from tests.test_structure_normalize import manifest_data


def write_manifest(tmp_path, *, method="manual_only", output_name="prepared"):
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["project"]["output_dir"] = str(tmp_path / output_name)
    data["protonation"]["method"] = method
    data["protonation"]["overrides"] = [override("A", "ASP", 3, "ASH")]
    manifest = tmp_path / f"{method}.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest, tmp_path / output_name


def test_prepare_protonation_stage_writes_expected_files(tmp_path):
    manifest, output_dir = write_manifest(tmp_path)
    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "protonation"],
    )

    assert result.exit_code == 0
    expected = [
        output_dir / "manifest.input.yaml",
        output_dir / "manifest.lock.yaml",
        output_dir / "versions.json",
        output_dir / "intermediate" / "00_input_normalized.pdb",
        output_dir / "intermediate" / "01_protonation_assigned.pdb",
        output_dir / "reports" / "structure_report.json",
        output_dir / "reports" / "structure_report.md",
        output_dir / "reports" / "protonation_report.json",
        output_dir / "reports" / "protonation_report.csv",
        output_dir / "reports" / "protonation_report.md",
    ]
    for path in expected:
        assert path.exists(), path

    assigned = read_pdb(output_dir / "intermediate" / "01_protonation_assigned.pdb")
    assert "ASH" in [residue.id.resname for residue in assigned.residues]


def test_protonation_reports_contain_changes_and_csv_columns(tmp_path):
    manifest, output_dir = write_manifest(tmp_path)
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "protonation"])

    assert result.exit_code == 0
    report = json.loads((output_dir / "reports" / "protonation_report.json").read_text(encoding="utf-8"))
    assert report["residues_changed"][0]["final_resname"] == "ASH"

    with (output_dir / "reports" / "protonation_report.csv").open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "chain",
            "resid",
            "icode",
            "original_resname",
            "final_resname",
            "source",
            "pka",
            "ph",
            "reason",
            "changed",
        ]


def test_automated_propka_method_fails_clearly_if_propka_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: (_ for _ in ()).throw(
            PropkaExecutionError("PropKa executable not found. Searched: propka3, propka")
        ),
    )
    manifest, _ = write_manifest(tmp_path, method="propka")
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "protonation"])

    assert result.exit_code != 0
    assert "PropKa executable not found" in result.output


def test_automated_propka_xtb_method_fails_clearly_if_propka_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: (_ for _ in ()).throw(
            PropkaExecutionError("PropKa executable not found. Searched: propka3, propka")
        ),
    )
    manifest, _ = write_manifest(tmp_path, method="propka_xtb_his")
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "protonation"])

    assert result.exit_code != 0
    assert "PropKa executable not found" in result.output


def test_structure_stage_still_works_with_future_automated_method(tmp_path):
    manifest, output_dir = write_manifest(tmp_path, method="propka_xtb_his")
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "structure"])

    assert result.exit_code == 0
    assert (output_dir / "intermediate" / "00_input_normalized.pdb").exists()


def test_full_prepare_without_stop_after_fails_clearly(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    result = CliRunner().invoke(app, ["prepare", str(manifest)])

    assert result.exit_code != 0
    assert "Full Amber preparation is not implemented yet" in result.output
