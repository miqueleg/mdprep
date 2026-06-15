from pathlib import Path

from typer.testing import CliRunner

from mdprep.cli import app


def test_cli_help_works():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "config-check" in result.output
    assert "selftest" in result.output


def test_cli_config_check_examples():
    paths = [str(path) for path in sorted(Path("examples").glob("*.yaml"))]

    result = CliRunner().invoke(app, ["config-check", *paths])

    assert result.exit_code == 0
    assert "PASS" in result.output

