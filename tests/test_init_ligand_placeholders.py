from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.config.loader import load_manifest


def test_init_include_ligand_placeholders_generates_valid_active_blocks(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(
        app,
        [
            "init",
            "tests/data/protein_two_ligands.pdb",
            "-o",
            str(output),
            "--include-ligand-placeholders",
        ],
    )

    assert result.exit_code == 0
    manifest = load_manifest(output)
    assert [ligand.id for ligand in manifest.ligands] == ["sub_501", "cof_601"]
    text = output.read_text(encoding="utf-8")
    assert "Check every net_charge" in text


def test_init_default_comments_ligand_placeholders_without_active_blocks(tmp_path):
    output = tmp_path / "system.yaml"
    result = CliRunner().invoke(app, ["init", "tests/data/protein_two_ligands.pdb", "-o", str(output)])

    assert result.exit_code == 0
    manifest = load_manifest(output)
    assert manifest.ligands == []
    text = output.read_text(encoding="utf-8")
    assert "Detected ligand placeholder suggestions are commented out by default" in text
    assert "CHECK THIS VALUE" in text
