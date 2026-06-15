import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, manifest_data


def write_manifest(tmp_path: Path, output_dir: Path) -> Path:
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(output_dir)
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        ligand_entry("SUB_501", "B", "SUB", 501),
        ligand_entry("COF_601", "C", "COF", 601),
    ]
    manifest = tmp_path / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest


def test_prepare_structure_stage_writes_outputs(tmp_path):
    output_dir = tmp_path / "prepared"
    manifest = write_manifest(tmp_path, output_dir)

    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "structure"],
    )

    assert result.exit_code == 0
    assert (output_dir / "manifest.input.yaml").exists()
    assert (output_dir / "manifest.lock.yaml").exists()
    assert (output_dir / "versions.json").exists()
    normalized = output_dir / "intermediate" / "00_input_normalized.pdb"
    assert normalized.exists()
    assert (output_dir / "reports" / "structure_report.json").exists()
    assert (output_dir / "reports" / "structure_report.md").exists()

    reparsed = read_pdb(normalized)
    assert [residue.id.resname for residue in reparsed.residues] == ["ALA", "SUB", "COF"]
    report = json.loads((output_dir / "reports" / "structure_report.json").read_text(encoding="utf-8"))
    assert report["atom_count_after"] == 7


def test_prepare_output_dir_not_overwritten_without_flag(tmp_path):
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    manifest = write_manifest(tmp_path, output_dir)

    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "structure"],
    )

    assert result.exit_code != 0
    assert "Output directory already exists" in result.output


def test_prepare_overwrite_works(tmp_path):
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    manifest = write_manifest(tmp_path, output_dir)

    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "structure", "--overwrite"],
    )

    assert result.exit_code == 0
    assert (output_dir / "intermediate" / "00_input_normalized.pdb").exists()

