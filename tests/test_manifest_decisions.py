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
    assert "ligand.b_sub_501.net_charge" in ligand_ids
    assert "ligand.c_cof_601.net_charge" in ligand_ids
    assert plan.findings["ligands"] == ["B:SUB501", "C:COF601"]


def test_a_structure_without_ligands_is_not_asked_about_them():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    assert [d for d in plan.decisions if d.section == "ligands"] == []
    assert plan.findings["ligands"] == []


def test_ligand_net_charge_has_no_default_and_is_blocking():
    """A guessed ligand charge silently produces a wrong system."""
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("ligand.b_sub_501.net_charge")

    assert decision.default is None
    assert decision.requires_user_input is True


def test_metal_oxidation_state_has_no_default_and_is_blocking():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    model = plan.by_id("metal.z_zn_500_site.model")
    charge = plan.by_id("metal.z_zn_500_site.charge")

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
    assert selectors["b_sub_501"] == {
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
    answers["ligand.b_sub_501.include"] = False
    answers.pop("ligand.b_sub_501.net_charge")

    manifest = build_manifest(plan, answers)

    assert [entry["id"] for entry in manifest["ligands"]] == ["c_cof_601"]


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

    nonbonded = plan.blocking({"metal.z_zn_500_site.model": "nonbonded"})
    bonded = plan.blocking(
        {
            "metal.z_zn_500_site.model": "bonded_mcpb",
            "metal.z_zn_500_site.confirm_donors": "true",
        }
    )

    bonded_ids = {d.id for d in bonded}
    nonbonded_ids = {d.id for d in nonbonded}
    # The oxidation state is needed either way.
    assert "metal.z_zn_500_site.charge" in bonded_ids & nonbonded_ids
    # The capped-model questions only arise for the bonded model.
    assert "metal.z_zn_500_site.model_charge" in bonded_ids
    assert "metal.z_zn_500_site.model_charge" not in nonbonded_ids


def test_a_rejected_coordination_shell_is_reported_not_invented():
    """mdprep must never make up bonds the user declined to confirm."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    answers = {
        "structure.remove_unknown_heterogens": False,
        "metal.z_zn_500_site.model": "bonded_mcpb",
        "metal.z_zn_500_site.charge": 2,
        "metal.z_zn_500_site.confirm_donors": False,
    }

    manifest = build_manifest(plan, answers)

    assert manifest["metals"] == []
    assert unresolved_mcpb_sites(plan, answers) == ["z_zn_500_site"]


def test_a_confirmed_coordination_shell_is_generated():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    answers = {
        "structure.remove_unknown_heterogens": False,
        "metal.z_zn_500_site.model": "bonded_mcpb",
        "metal.z_zn_500_site.charge": 2,
        "metal.z_zn_500_site.confirm_donors": True,
        "metal.z_zn_500_site.multiplicity": 1,
        "metal.z_zn_500_site.model_charge": 2,
    }

    manifest = build_manifest(plan, answers)

    assert unresolved_mcpb_sites(plan, answers) == []
    site = manifest["metals"][0]
    assert site["model"] == "bonded_mcpb"
    assert site["mcpb"]["bonds"]
    assert site["mcpb"]["pyscf"]["hessian_backend"] == "xtb"
    assert site["mcpb"]["pyscf"]["xtb"]["model"] == "gfn2"
    ManifestConfig.model_validate(manifest)


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
    decision = plan.by_id("metal.z_zn_500_site.charge")

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


# --------------------------------------------------------------------------
# Viewer targets: what a 3D front end highlights for each question
# --------------------------------------------------------------------------


def test_every_decision_declares_a_focus_mode():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    assert all(
        decision.focus in {"none", "targets", "structure"}
        for decision in plan.decisions
    )


def test_targeted_decisions_actually_carry_something_to_show():
    """focus=targets with nothing to highlight would leave the viewer blank."""
    for structure in sorted(path.name for path in DATA.glob("*.pdb")):
        plan = plan_decisions(DATA / structure)
        for decision in plan.decisions:
            if decision.focus == "targets":
                assert decision.targets or decision.target_resnames, (
                    f"{structure}: {decision.id} focuses on targets but has none"
                )


def test_project_questions_have_no_structural_focus():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    assert plan.by_id("project.name").focus == "none"
    assert plan.by_id("project.output_dir").focus == "none"


def test_ligand_questions_point_at_that_ligand():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    charge = plan.by_id("ligand.b_sub_501.net_charge")

    assert charge.focus == "targets"
    assert [(t.resname, t.resid) for t in charge.targets] == [("SUB", 501)]
    assert charge.subject.resname == "SUB"


def test_disulfide_question_points_at_both_partners():
    plan = plan_decisions(DATA / "protein_disulfide.pdb")

    decision = plan.by_id("disulfides.auto_detect")

    assert len(decision.targets) == 2
    assert {target.resname for target in decision.targets} <= {"CYS", "CYX"}


def test_unknown_heterogen_question_points_at_each_residue_at_stake():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("structure.remove_unknown_heterogens")

    assert {(t.resname, t.resid) for t in decision.targets} == {
        ("SUB", 501),
        ("COF", 601),
    }


def test_waters_are_selected_by_residue_name_not_enumerated():
    """Enumerating thousands of waters would bloat the plan for no benefit."""
    plan = plan_decisions(DATA / "protein_with_waters.pdb")

    decision = plan.by_id("structure.keep_crystal_waters")

    assert decision.target_resnames == ("HOH",)
    assert decision.targets == ()


def test_protonation_question_points_at_the_histidines():
    plan = plan_decisions(DATA / "protein_histidine_ring.pdb")

    decision = plan.by_id("protonation.method")

    assert decision.focus == "targets"
    assert all(target.resname.startswith("HI") for target in decision.targets)


def test_protonation_focuses_on_the_whole_structure_without_histidines():
    plan = plan_decisions(DATA / "protein_disulfide.pdb")
    assert plan.findings["histidines"] == []

    assert plan.by_id("protonation.method").focus == "structure"


def test_metal_question_shows_the_ion_first_then_its_surroundings():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    decision = plan.by_id("metal.z_zn_500_site.model")

    assert decision.subject.resname == "ZN"
    assert decision.subject.element == "Zn"
    # The contacting residue is context, so it follows the subject.
    assert len(decision.targets) > 1
    assert any(target.resname.startswith("HI") for target in decision.targets[1:])


def test_contact_search_is_framing_not_a_coordination_claim():
    """mdprep must not imply it has determined coordination."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    evidence = " ".join(plan.by_id("metal.z_zn_500_site.model").evidence)

    assert "within 3.0 A" in evidence
    assert "coordinat" not in evidence.lower()


def test_contact_residues_respects_the_cutoff():
    from mdprep.config.decisions import contact_residues
    from mdprep.structure.inspect import inspect_pdb_structure

    summary = inspect_pdb_structure(DATA / "protein_histidine_zinc.pdb")
    zinc = summary.metal_ions[0]

    near = contact_residues(summary.structure.residues, zinc, cutoff_angstrom=3.0)
    none_at_all = contact_residues(
        summary.structure.residues, zinc, cutoff_angstrom=0.1
    )

    assert near
    assert zinc.id not in {residue.id for residue in near}
    assert none_at_all == []


def test_viewer_fields_survive_the_json_round_trip():
    import json

    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    payload = json.loads(json.dumps(plan.to_dict()))
    metal = next(
        d for d in payload["decisions"] if d["id"] == "metal.z_zn_500_site.model"
    )

    assert metal["focus"] == "targets"
    assert metal["targets"][0]["resname"] == "ZN"
    assert metal["targets"][0]["element"] == "Zn"
    assert "target_resnames" in metal


def test_ids_are_unique_across_repeated_chains():
    """A crystal structure with several copies repeats every residue number.

    Without the chain in the id, every copy of a ligand collides and the
    manifest is rejected for duplicate ids before it can be used.
    """
    import tempfile
    from pathlib import Path as _Path

    source = (DATA / "protein_two_ligands.pdb").read_text(encoding="utf-8")
    # Duplicate the ligands into a second chain at the same residue numbers.
    extra = []
    for line in source.splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[21] in {"B", "C"}:
            extra.append(line[:21] + "X" + line[22:])
    with tempfile.TemporaryDirectory() as tmp:
        doubled = _Path(tmp) / "doubled.pdb"
        doubled.write_text(source + "\n".join(extra) + "\nEND\n", encoding="utf-8")
        plan = plan_decisions(doubled)

        manifest = build_manifest(plan, answer_blocking(plan))

    ligand_ids = [entry["id"] for entry in manifest["ligands"]]
    assert len(ligand_ids) == len(set(ligand_ids))
    assert "b_sub_501" in ligand_ids and "x_sub_501" in ligand_ids
    ManifestConfig.model_validate(manifest)


def test_blank_chain_ids_do_not_gain_an_empty_prefix():
    plan = plan_decisions(DATA / "protein_atom_record_ligand_blank_chain.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    assert not manifest["ligands"][0]["id"].startswith("_")


# --------------------------------------------------------------------------
# Chain selection
# --------------------------------------------------------------------------


def test_a_single_chain_structure_is_not_asked_about_chains():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    assert not any(d.id == "structure.chains" for d in plan.decisions)


def test_a_chain_holding_only_a_metal_is_not_offered_for_selection():
    """Chain Z is the zinc; discarding it would break the structure."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    assert plan.findings["protein_chains"] == ["A"]
    assert plan.findings["non_protein_chains"] == ["Z"]
    assert not any(d.id == "structure.chains" for d in plan.decisions)


def test_hetero_only_chains_are_kept_whichever_protein_chain_is_chosen():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    # No selection was made, so every chain is kept.
    assert manifest["structure"]["keep_chains"] == []
