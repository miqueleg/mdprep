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



def test_residue_names_are_not_rendered_as_emoji():
    """rich reads ":CL" as an emoji shortcode.

    A chloride at A:CL303 printed as "A<emoji>303", which makes a diagnostic
    about a metal-site halide unreadable.
    """
    from typer.testing import CliRunner

    from mdprep.cli import app

    result = CliRunner().invoke(app, ["config-check", "/nope/A:CL303/x.yaml"])

    assert "CL303" in result.output
    assert "\N{SQUARED CL}" not in result.output


def test_square_brackets_in_a_message_are_not_read_as_style_markup():
    from typer.testing import CliRunner

    from mdprep.cli import app

    result = CliRunner().invoke(app, ["plan", "/nope/[red]bogus/x.pdb"])

    assert "[red]" in result.output
