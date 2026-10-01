"""Tests for the structure-aware manifest decision engine.

The engine exists so that every front end -- CLI wizard, web form, notebook --
asks the same short list of questions. The guarantees worth protecting are that
the list is pruned to what the structure needs, and that a chemistry-sensitive
value can never acquire a default.
"""

from pathlib import Path

import pytest
import yaml

from mdprep.config.decisions import (
    Decision,
    DecisionError,
    DecisionPlan,
    build_manifest,
    coerce_answer,
    decision_sections,
    plan_decisions,
    resolve_answers,
    unresolved_mcpb_sites,
)
from mdprep.config.models import ManifestConfig


DATA = Path("tests/data")

# Values a caller must supply because the structure cannot imply them.
BLOCKING_SUFFIXES = (".net_charge", ".model", ".charge", "production.steps")


def answer_blocking(plan: DecisionPlan, **overrides: object) -> dict[str, object]:
    """Mechanically answer every blocking decision, for tests about other things."""

    answers: dict[str, object] = {}
    for decision in plan.blocking():
        if decision.id.endswith(".net_charge"):
            answers[decision.id] = 0
        elif decision.id.endswith(".model"):
            answers[decision.id] = "nonbonded"
        elif decision.id.endswith(".charge"):
            answers[decision.id] = 2
        elif decision.id.endswith("remove_unknown_heterogens"):
            answers[decision.id] = False
        elif decision.id.endswith("production.steps"):
            answers[decision.id] = 500_000
    answers.update(overrides)
    return answers


def test_plan_is_far_smaller_than_the_schema():
    """The point of the engine: ~20 questions, not the 250+ fields available."""
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    assert len(plan.decisions) < 30
    assert len(ManifestConfig.model_json_schema()["$defs"]) > 30


def test_detected_ligands_each_get_their_own_decisions():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    ligand_ids = {
        decision.id for decision in plan.decisions if decision.section == "ligands"
    }
    assert "ligand.sub_501.net_charge" in ligand_ids
    assert "ligand.cof_601.net_charge" in ligand_ids
    assert plan.findings["ligands"] == ["B:SUB501", "C:COF601"]


def test_a_structure_without_ligands_is_not_asked_about_them():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    assert [d for d in plan.decisions if d.section == "ligands"] == []
    assert plan.findings["ligands"] == []


def test_ligand_net_charge_has_no_default_and_is_blocking():
    """A guessed ligand charge silently produces a wrong system."""
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("ligand.sub_501.net_charge")

    assert decision.default is None
    assert decision.requires_user_input is True


def test_metal_oxidation_state_has_no_default_and_is_blocking():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    model = plan.by_id("metal.zn_500_site.model")
    charge = plan.by_id("metal.zn_500_site.charge")

    assert model.default is None and model.requires_user_input
    assert charge.default is None and charge.requires_user_input


def test_unknown_heterogen_removal_is_blocking():
    """Normalization must never drop an unknown heterogen without explicit config."""
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("structure.remove_unknown_heterogens")

    assert decision.default is None
    assert decision.requires_user_input is True
    assert decision.evidence  # the residues at stake are named


def test_metal_ions_are_not_offered_as_gaff_ligands():
    """A zinc is parameterised as a metal site, never by antechamber."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    assert plan.findings["metal_ions"] == ["Z:ZN500"]
    assert [d for d in plan.decisions if d.section == "ligands"] == []


def test_build_manifest_refuses_to_default_a_blocking_decision():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    with pytest.raises(DecisionError, match="net_charge"):
        build_manifest(plan, {"structure.remove_unknown_heterogens": False})


@pytest.mark.parametrize(
    "structure",
    sorted(path.name for path in DATA.glob("*.pdb")),
)
def test_every_test_structure_produces_a_valid_manifest(structure):
    plan = plan_decisions(DATA / structure)

    manifest = build_manifest(plan, answer_blocking(plan))

    ManifestConfig.model_validate(manifest)


def test_generated_manifest_round_trips_through_yaml():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")
    manifest = build_manifest(plan, answer_blocking(plan))

    reloaded = yaml.safe_load(yaml.safe_dump(manifest))

    ManifestConfig.model_validate(reloaded)


def test_ligand_selector_identifies_the_detected_residue():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    selectors = {entry["id"]: entry["selector"] for entry in manifest["ligands"]}
    assert selectors["sub_501"] == {
        "chain": "B",
        "resname": "SUB",
        "resid": 501,
        "icode": None,
    }


def test_blank_chain_ligand_selector_is_preserved():
    plan = plan_decisions(DATA / "protein_atom_record_ligand_blank_chain.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    assert manifest["ligands"][0]["selector"]["chain"] == ""
    ManifestConfig.model_validate(manifest)


def test_declining_a_ligand_removes_its_dependent_questions():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")
    answers = answer_blocking(plan)
    answers["ligand.sub_501.include"] = False
    answers.pop("ligand.sub_501.net_charge")

    manifest = build_manifest(plan, answers)

    assert [entry["id"] for entry in manifest["ligands"]] == ["cof_601"]


def test_conditional_decisions_are_pruned_by_earlier_answers():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    without = plan.applicable({"solvation.enabled": False})
    with_solvent = plan.applicable({"solvation.enabled": True})

    assert not any(d.id == "solvation.box" for d in without)
    assert any(d.id == "solvation.box" for d in with_solvent)


def test_ph_is_only_asked_for_propka_methods():
    plan = plan_decisions(DATA / "protein_histidine_ring.pdb")

    manual = plan.applicable({"protonation.method": "manual_only"})
    propka = plan.applicable({"protonation.method": "propka"})

    assert not any(d.id == "protonation.ph" for d in manual)
    assert any(d.id == "protonation.ph" for d in propka)


def test_xtb_thread_bound_is_written_for_the_xtb_histidine_method():
    plan = plan_decisions(DATA / "protein_histidine_ring.pdb")
    answers = answer_blocking(plan)
    answers["protonation.method"] = "propka_xtb_his"

    manifest = build_manifest(plan, answers)

    assert manifest["protonation"]["histidine"]["xtb"]["num_threads"] == 1
    ManifestConfig.model_validate(manifest)


def test_blocking_count_respects_answers_already_given():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    everything = plan.blocking()
    after_mcpb = plan.blocking({"metal.zn_500_site.model": "bonded_mcpb"})

    assert any(d.id == "metal.zn_500_site.charge" for d in everything)
    # The nonbonded ion charge is not asked once MCPB is chosen.
    assert not any(d.id == "metal.zn_500_site.charge" for d in after_mcpb)


def test_mcpb_sites_are_reported_rather_than_guessed():
    """MCPB needs external QM artifacts, so the engine must not invent a block."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    answers = {
        "structure.remove_unknown_heterogens": False,
        "metal.zn_500_site.model": "bonded_mcpb",
    }

    manifest = build_manifest(plan, answers)

    assert manifest["metals"] == []
    assert unresolved_mcpb_sites(plan, answers) == ["zn_500_site"]


def test_md_block_is_omitted_until_enabled():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    assert manifest["molecular_dynamics"] == {"enabled": False}


def test_enabling_md_requires_an_explicit_production_length():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    with pytest.raises(DecisionError, match="production.steps"):
        build_manifest(plan, {"molecular_dynamics.enabled": True})

    manifest = build_manifest(
        plan,
        {"molecular_dynamics.enabled": True, "molecular_dynamics.production.steps": 1000},
    )
    assert manifest["molecular_dynamics"]["production"]["steps"] == 1000
    ManifestConfig.model_validate(manifest)


def test_reporting_intervals_never_exceed_a_short_production_run():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    manifest = build_manifest(
        plan,
        {"molecular_dynamics.enabled": True, "molecular_dynamics.production.steps": 100},
    )

    ManifestConfig.model_validate(manifest)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("yes", True), ("y", True), ("true", True), ("no", False), ("n", False), ("0", False)],
)
def test_boolean_answers_accept_common_spellings(raw, expected):
    decision = Decision(
        id="x", section="s", question="q", kind="boolean", why="w", default="true"
    )

    assert coerce_answer(decision, raw) is expected


def test_choice_answers_are_validated_against_the_allowed_set():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")
    decision = plan.by_id("protein.forcefield")

    with pytest.raises(DecisionError, match="ff99SB"):
        coerce_answer(decision, "ff99SB")


def test_integer_bounds_are_enforced():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    decision = plan.by_id("metal.zn_500_site.charge")

    with pytest.raises(DecisionError, match="at most 4"):
        coerce_answer(decision, 9)


def test_resolve_answers_fills_defaults_for_everything_else():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    resolved = resolve_answers(plan, answer_blocking(plan))

    assert resolved["protein.forcefield"] == "ff14SB"
    assert resolved["solvation.buffer_angstrom"] == 10.0


def test_plan_serialises_to_json_safe_primitives():
    """The JSON form is the contract a web form or notebook consumes."""
    import json

    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    payload = json.loads(json.dumps(plan.to_dict()))

    assert payload["blocking_count"] == len(plan.blocking())
    assert payload["decision_count"] == len(plan.decisions)
    first = payload["decisions"][0]
    assert {"id", "section", "question", "kind", "default"} <= set(first)


def test_sections_are_reported_in_presentation_order():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    sections = decision_sections(plan.decisions)

    assert sections[0] == "project"
    assert "ligands" in sections
    assert sections.index("project") < sections.index("solvation")
