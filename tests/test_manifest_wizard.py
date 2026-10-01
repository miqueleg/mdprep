"""Tests for the interactive manifest wizard and its CLI surface."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.config.decisions import plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.workflows.wizard import (
    collect_answers,
    format_question,
    render_manifest_yaml,
    run_wizard,
)


DATA = Path("tests/data")
runner = CliRunner()


def scripted_prompter(answers_by_id: dict[str, str]):
    """Answer by decision id; anything not listed takes its default."""

    asked: list[str] = []

    def prompt(decision) -> str:
        asked.append(decision.id)
        return answers_by_id.get(decision.id, "")

    prompt.asked = asked  # type: ignore[attr-defined]
    return prompt


def test_wizard_writes_a_manifest_that_passes_config_check(tmp_path):
    output = tmp_path / "generated.yaml"
    prompt = scripted_prompter(
        {
            "structure.remove_unknown_heterogens": "no",
            "ligand.sub_501.net_charge": "-1",
            "ligand.cof_601.net_charge": "0",
        }
    )

    result = run_wizard(
        DATA / "protein_two_ligands.pdb", output_path=output, prompt=prompt
    )

    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    ManifestConfig.model_validate(manifest)
    assert result.manifest_path == output
    assert manifest["ligands"][0]["net_charge"] == -1
    assert manifest["ligands"][1]["net_charge"] == 0


def test_wizard_never_accepts_a_default_for_a_blocking_question(tmp_path):
    """An empty answer to a blocking question must be re-asked, not defaulted."""
    attempts: list[str] = []

    def prompt(decision) -> str:
        attempts.append(decision.id)
        if decision.id == "ligand.sub_501.net_charge":
            # Two empty answers, then a real one.
            if attempts.count(decision.id) < 3:
                return ""
            return "2"
        if decision.requires_user_input:
            return "no" if decision.kind == "boolean" else "0"
        return ""

    run_wizard(
        DATA / "protein_two_ligands.pdb",
        output_path=tmp_path / "m.yaml",
        prompt=prompt,
    )

    assert attempts.count("ligand.sub_501.net_charge") == 3


def test_wizard_rejects_an_invalid_answer_and_asks_again(tmp_path):
    seen: list[str] = []

    def prompt(decision) -> str:
        seen.append(decision.id)
        if decision.id == "protein.forcefield":
            return "ff99SB" if seen.count(decision.id) == 1 else "ff19SB"
        if decision.requires_user_input:
            return "no" if decision.kind == "boolean" else "0"
        return ""

    run_wizard(
        DATA / "protein_two_ligands.pdb",
        output_path=tmp_path / "m.yaml",
        prompt=prompt,
    )

    assert seen.count("protein.forcefield") == 2


def test_unreachable_questions_are_never_asked(tmp_path):
    prompt = scripted_prompter(
        {
            "structure.remove_unknown_heterogens": "no",
            "ligand.sub_501.include": "no",
            "ligand.cof_601.net_charge": "0",
        }
    )

    run_wizard(
        DATA / "protein_two_ligands.pdb",
        output_path=tmp_path / "m.yaml",
        prompt=prompt,
    )

    asked = prompt.asked  # type: ignore[attr-defined]
    assert "ligand.sub_501.net_charge" not in asked
    assert "ligand.cof_601.net_charge" in asked


def test_preset_answers_are_not_asked(tmp_path):
    prompt = scripted_prompter({"structure.remove_unknown_heterogens": "no"})

    run_wizard(
        DATA / "protein_with_waters.pdb",
        output_path=tmp_path / "m.yaml",
        prompt=prompt,
        preset={"project.output_dir": "/scratch/run1", "ligand.so4_1.net_charge": -2},
    )

    asked = prompt.asked  # type: ignore[attr-defined]
    assert "project.output_dir" not in asked
    manifest = yaml.safe_load((tmp_path / "m.yaml").read_text(encoding="utf-8"))
    assert manifest["project"]["output_dir"] == "/scratch/run1"
    assert manifest["ligands"][0]["net_charge"] == -2


def test_wizard_refuses_to_overwrite_without_permission(tmp_path):
    output = tmp_path / "existing.yaml"
    output.write_text("already here\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_wizard(
            DATA / "protein_blank_chain.pdb",
            output_path=output,
            prompt=scripted_prompter({}),
        )

    assert output.read_text(encoding="utf-8") == "already here\n"


def test_mcpb_choice_is_flagged_rather_than_written(tmp_path):
    prompt = scripted_prompter({"metal.zn_500_site.model": "bonded_mcpb"})

    result = run_wizard(
        DATA / "protein_histidine_zinc.pdb",
        output_path=tmp_path / "m.yaml",
        prompt=prompt,
    )

    assert result.mcpb_sites_needing_manual_work == ("zn_500_site",)
    text = (tmp_path / "m.yaml").read_text(encoding="utf-8")
    assert "bonded_mcpb" in text  # the note
    assert yaml.safe_load(text)["metals"] == []


def test_rendered_manifest_records_that_values_were_entered_not_inferred():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")
    answers, _ = collect_answers(plan, scripted_prompter({}))

    text = render_manifest_yaml(plan, answers)

    assert "not inferred" in text
    assert str(plan.structure_path) in text


def test_question_text_shows_evidence_and_rationale():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    text = format_question(plan.by_id("ligand.sub_501.net_charge"))

    assert "B:SUB501" in text
    assert "cannot be guessed" in text
    assert "press Enter to accept" not in text


def test_question_text_marks_the_default_choice():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    text = format_question(plan.by_id("protein.forcefield"))

    assert "* ff14SB" in text
    assert "press Enter to accept: ff14SB" in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_plan_command_json_is_machine_readable():
    result = runner.invoke(app, ["plan", str(DATA / "protein_two_ligands.pdb"), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["decision_count"] > 0
    assert payload["blocking_count"] > 0
    assert payload["findings"]["ligands"] == ["B:SUB501", "C:COF601"]


def test_plan_command_table_names_the_blocking_decisions():
    result = runner.invoke(app, ["plan", str(DATA / "protein_histidine_zinc.pdb")])

    assert result.exit_code == 0, result.output
    assert "must be answered by a person" in result.output


def test_plan_command_reports_a_missing_file_clearly():
    result = runner.invoke(app, ["plan", "does_not_exist.pdb"])

    assert result.exit_code == 1
    assert "Error" in result.output


def test_init_interactive_writes_a_valid_manifest(tmp_path):
    output = tmp_path / "wizard.yaml"
    # Answers in decision order; empty lines accept the shown default.
    keystrokes = "\n".join(
        [
            "demo",           # project.name
            "",               # project.output_dir
            "no",             # structure.remove_unknown_heterogens
            "",               # protein.forcefield
            "",               # protein.water_model
            "",               # protonation.method
            "",               # ligand.sub_501.include
            "-1",             # ligand.sub_501.net_charge
            "",               # ligand.sub_501.atom_types
            "",               # ligand.sub_501.charge_method
            "",               # ligand.cof_601.include
            "0",              # ligand.cof_601.net_charge
            "",               # ligand.cof_601.atom_types
            "",               # ligand.cof_601.charge_method
            "",               # solvation.enabled
            "",               # solvation.box
            "",               # solvation.buffer_angstrom
            "",               # solvation.salt_concentration_molar
            "",               # molecular_dynamics.enabled
        ]
    ) + "\n"

    result = runner.invoke(
        app,
        ["init", str(DATA / "protein_two_ligands.pdb"), "-i", "-o", str(output)],
        input=keystrokes,
    )

    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load(output.read_text(encoding="utf-8"))
    ManifestConfig.model_validate(manifest)
    assert manifest["project"]["name"] == "demo"
    assert manifest["ligands"][0]["net_charge"] == -1

    check = runner.invoke(app, ["config-check", str(output)])
    assert check.exit_code == 0, check.output


def test_init_without_interactive_still_uses_the_original_generator(tmp_path):
    """The non-interactive starter manifest must keep working unchanged."""
    output = tmp_path / "starter.yaml"

    result = runner.invoke(
        app, ["init", str(DATA / "protein_two_ligands.pdb"), "-o", str(output)]
    )

    assert result.exit_code == 0, result.output
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# mdprep starter manifest")
    assert "CHECK THIS VALUE" in text


def test_a_front_end_that_cannot_answer_fails_instead_of_spinning(tmp_path):
    """A blocking question is re-asked, but not forever.

    Without a bound, any non-interactive driver that returns an empty answer
    hangs the process.
    """
    from mdprep.config.decisions import DecisionError

    with pytest.raises(DecisionError, match="no usable answer"):
        run_wizard(
            DATA / "protein_two_ligands.pdb",
            output_path=tmp_path / "m.yaml",
            prompt=lambda decision: "",
            max_attempts=3,
        )

    assert not (tmp_path / "m.yaml").exists()


def test_plan_json_is_valid_on_a_narrow_terminal(monkeypatch):
    """rich reflows to the console width, which would corrupt the JSON."""
    monkeypatch.setenv("COLUMNS", "40")

    result = runner.invoke(app, ["plan", str(DATA / "protein_two_ligands.pdb"), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["decision_count"] > 0


def test_inspect_json_is_valid_on_a_narrow_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "40")

    result = runner.invoke(
        app, ["inspect", str(DATA / "protein_two_ligands.pdb"), "--json"]
    )

    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
