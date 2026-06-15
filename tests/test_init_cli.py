from pathlib import Path

from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.config.loader import load_manifest


def test_init_generates_valid_manifest(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(
        app,
        ["init", "tests/data/protein_with_waters.pdb", "-o", str(output)],
    )

    assert result.exit_code == 0
    manifest = load_manifest(output)
    assert manifest.project.input_structure == "tests/data/protein_with_waters.pdb"

    config_check = CliRunner().invoke(app, ["config-check", str(output)])
    assert config_check.exit_code == 0


def test_init_comments_likely_ligands(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(
        app,
        ["init", "tests/data/protein_two_ligands.pdb", "-o", str(output)],
    )

    assert result.exit_code == 0
    text = output.read_text(encoding="utf-8")
    assert "# Likely ligands/cofactors found:" in text
    assert "#   - B:SUB501" in text
    assert "#   - C:COF601" in text


def test_init_does_not_overwrite_without_flag(tmp_path):
    output = tmp_path / "system.yaml"
    output.write_text("existing: true\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["init", "tests/data/protein_with_waters.pdb", "-o", str(output)],
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert output.read_text(encoding="utf-8") == "existing: true\n"


def test_init_overwrite_replaces_existing_file(tmp_path):
    output = tmp_path / "system.yaml"
    output.write_text("existing: true\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["init", "tests/data/protein_with_waters.pdb", "-o", str(output), "--overwrite"],
    )

    assert result.exit_code == 0
    assert "project:" in output.read_text(encoding="utf-8")


def test_init_accepts_manual_only_protonation_method(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(
        app,
        [
            "init",
            "tests/data/protein_with_waters.pdb",
            "-o",
            str(output),
            "--protonation-method",
            "manual_only",
        ],
    )

    assert result.exit_code == 0
    manifest = load_manifest(output)
    assert manifest.protonation.method == "manual_only"


def test_init_rejects_invalid_protonation_method(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(
        app,
        [
            "init",
            "tests/data/protein_with_waters.pdb",
            "-o",
            str(output),
            "--protonation-method",
            "bad",
        ],
    )

    assert result.exit_code != 0
    assert "--protonation-method" in result.output
