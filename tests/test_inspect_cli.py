import json

from typer.testing import CliRunner

from mdprep.cli import app


def test_inspect_cli_exits_zero_for_pdb():
    result = CliRunner().invoke(app, ["inspect", "tests/data/protein_with_waters.pdb"])

    assert result.exit_code == 0
    assert "Structure summary" in result.output
    assert "Likely ligands/cofactors" in result.output


def test_inspect_cli_json_contains_likely_ligands():
    result = CliRunner().invoke(app, ["inspect", "tests/data/protein_two_ligands.pdb", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert [entry["resname"] for entry in data["likely_ligands"]] == ["SUB", "COF"]


def test_inspect_cli_missing_file_exits_nonzero_with_message():
    result = CliRunner().invoke(app, ["inspect", "missing.pdb"])

    assert result.exit_code != 0
    assert "Input structure not found" in result.output

