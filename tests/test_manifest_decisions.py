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
        elif decision.id.endswith(".smiles"):
            answers[decision.id] = "C"
        elif decision.id.endswith(".model"):
            answers[decision.id] = "nonbonded"
        elif decision.id.endswith(".charge"):
            answers[decision.id] = 2
        elif decision.id.endswith("remove_unknown_heterogens"):
            answers[decision.id] = False
        elif decision.id.endswith("add_missing_heavy_atoms"):
            answers[decision.id] = True
        elif decision.id == "structure.chains":
            answers[decision.id] = "all"
        elif decision.id.endswith(".confirm_donors"):
            answers[decision.id] = False
        elif decision.id.endswith("multiplicity"):
            # Covers the per-ion question and the QM region total, which
            # mdprep asks rather than deriving from the ions.
            answers[decision.id] = 1
        elif decision.id.endswith(".model_charge"):
            answers[decision.id] = 0
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


def test_an_oxidation_state_the_name_settles_is_proposed_and_not_blocking():
    """A residue called ZN is zinc(II) in the PDB dictionary, so asking the
    user to supply what the file already says is friction, not rigour. It is
    still asked, with the proposal filled in."""

    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    charge = plan.by_id("metal.z_zn_500_site.charge")

    assert charge.default == "2"
    assert not charge.requires_user_input
    assert "chemical component dictionary" in charge.why
    # And it is still a question, so a non-standard site can be corrected.
    assert charge.id in {decision.id for decision in plan.decisions}


def test_a_coordinated_metal_is_proposed_as_a_bonded_site():
    """A nonbonded model lets a coordination shell come apart in dynamics, so
    an ion with donors in range is proposed as bonded. The question is still
    asked -- this is a modelling choice, not something to settle silently."""

    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")

    model = plan.by_id("metal.z_zn_500_site.model")

    assert model.default == "bonded_mcpb"
    # The reason is shown, with the evidence it rests on.
    assert "donor atom(s) were found" in model.why
    # Accepting it is not a silent commitment: the questions a bonded site
    # then needs cannot be defaulted. The oxidation state is not among them
    # here only because this residue is called ZN, which settles it.
    blocking = {decision.id for decision in plan.blocking()}
    assert "metal.z_zn_500_site.confirm_donors" in blocking
    assert "refinement.total_qm_multiplicity" in blocking


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
    # Declining it makes it an unconfigured heterogen, so it has to be allowed
    # to go.
    answers["structure.remove_unknown_heterogens"] = True

    manifest = build_manifest(plan, answers)

    assert [entry["id"] for entry in manifest["ligands"]] == ["c_cof_601"]
    assert manifest["structure"]["remove_unknown_heterogens"] is True


def test_declining_a_ligand_while_keeping_unknown_heterogens_is_caught_here():
    """That pair of answers gives a manifest normalisation refuses, and it
    used to be found only in Colab -- after the setup cell had run."""

    plan = plan_decisions(DATA / "protein_two_ligands.pdb")
    answers = answer_blocking(plan)
    answers["ligand.b_sub_501.include"] = False
    answers.pop("ligand.b_sub_501.net_charge")
    answers["structure.remove_unknown_heterogens"] = False

    with pytest.raises(DecisionError, match="declined B:SUB501"):
        build_manifest(plan, answers)


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
    # Zinc's oxidation state is settled by its residue name, so it is not
    # blocking either way; what a bonded site adds is what it alone needs.
    assert "metal.z_zn_500_site.charge" not in bonded_ids | nonbonded_ids
    assert "refinement.total_qm_multiplicity" in bonded_ids - nonbonded_ids
    # The capped-model questions only arise for the bonded model.
    assert "metal.z_zn_500_site.model_charge" in bonded_ids
    assert "metal.z_zn_500_site.model_charge" not in nonbonded_ids


def test_a_rejected_coordination_shell_is_reported_not_invented():
    """mdprep must never make up bonds the user declined to confirm."""
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    answers = {
        "structure.remove_unknown_heterogens": False,
        "structure.repair.add_missing_heavy_atoms": True,
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
        "structure.repair.add_missing_heavy_atoms": True,
        "metal.z_zn_500_site.model": "bonded_mcpb",
        "metal.z_zn_500_site.charge": 2,
        "metal.z_zn_500_site.confirm_donors": True,
        "refinement.total_qm_multiplicity": 1,
        "metal.z_zn_500_site.model_charge": 2,
    }

    manifest = build_manifest(plan, answers)

    assert unresolved_mcpb_sites(plan, answers) == []
    site = manifest["metals"][0]
    assert site["model"] == "bonded_mcpb"
    assert site["mcpb"]["bonds"]
    assert site["mcpb"]["pyscf"]["hessian_backend"] == "xtb"
    assert site["mcpb"]["pyscf"]["xtb"]["model"] == "gxtb"
    assert site["mcpb"]["pyscf"]["esp_backend"] == "xtb"
    ManifestConfig.model_validate(manifest)


def test_md_block_is_omitted_until_enabled():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    manifest = build_manifest(plan, answer_blocking(plan))

    assert manifest["molecular_dynamics"] == {"enabled": False}


def test_enabling_md_requires_an_explicit_production_length():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    with pytest.raises(DecisionError, match="production.steps"):
        build_manifest(
            plan,
            {
                "structure.repair.add_missing_heavy_atoms": True,
                "molecular_dynamics.enabled": True,
            },
        )

    manifest = build_manifest(
        plan,
        {
            "structure.repair.add_missing_heavy_atoms": True,
            "molecular_dynamics.enabled": True,
            "molecular_dynamics.production.steps": 1000,
        },
    )
    assert manifest["molecular_dynamics"]["production"]["steps"] == 1000
    ManifestConfig.model_validate(manifest)


def test_reporting_intervals_never_exceed_a_short_production_run():
    plan = plan_decisions(DATA / "protein_blank_chain.pdb")

    manifest = build_manifest(
        plan,
        {
            "structure.repair.add_missing_heavy_atoms": True,
            "molecular_dynamics.enabled": True,
            "molecular_dynamics.production.steps": 100,
        },
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
    for structure in (DATA / "protein_histidine_zinc.pdb", Path("examples/tutorials/syrb2_2fct/2FCT.pdb")):
        plan = plan_decisions(structure)

        assert all(
            decision.focus in {"none", "targets", "structure", "chains"}
            for decision in plan.decisions
        )


def test_the_chain_questions_show_each_chain_in_its_own_colour():
    """Which chains to keep, and whether the copies match, are questions
    about the whole assembly: a rainbow per chain makes two copies look
    like one, and a close-up of one cofactor shows neither."""

    plan = plan_decisions("examples/tutorials/syrb2_2fct/2FCT.pdb")

    assert plan.by_id("structure.chains").focus == "chains"
    sharing = plan.by_id("structure.same_parametrization")
    assert sharing.focus == "chains"
    assert {target.chain for target in sharing.targets} == {"A", "B"}


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


def test_a_ligand_without_hydrogens_is_asked_for_a_smiles_and_not_nagged_after():
    """A net charge does not determine a molecule, so the protonation has to
    be stated, and it is asked for. Once it has been given, the builder must
    not then tell the student to go and add a SMILES -- which it did, to
    every student whose ligand came without hydrogens."""

    from mdprep.config.decisions import ligands_needing_hydrogens

    plan = plan_decisions(str(DATA / "protein_two_ligands.pdb"))

    asked = {d.id for d in plan.decisions if d.id.endswith(".smiles")}
    assert asked == {"ligand.b_sub_501.smiles", "ligand.c_cof_601.smiles"}
    assert all(plan.by_id(item).requires_user_input for item in asked)

    assert set(plan.findings["ligands_without_hydrogens"]) == {
        "b_sub_501",
        "c_cof_601",
    }
    answers = answer_blocking(plan)
    assert answers["ligand.b_sub_501.smiles"]
    assert ligands_needing_hydrogens(plan, answers) == []


def test_a_ligand_that_already_has_hydrogens_is_neither_asked_nor_reported():
    """mdprep leaves an already-protonated ligand alone, so asking for its
    SMILES would invite exactly the double-hydrogenation it avoids."""

    plan = plan_decisions(str(DATA / "protein_histidine_ring_hydrogenated.pdb"))

    assert [d.id for d in plan.decisions if d.id.endswith(".smiles")] == []
    assert plan.findings["ligands_without_hydrogens"] == []


def test_the_capped_model_charge_is_summed_from_answers_already_given(besd_structure):
    """mdprep asks for the metal's oxidation state and each ligand's net
    charge, so by the time it asks for the model charge it has the terms.
    Making the user add them up again invites an arithmetic slip that gives
    converged, wrong parameters."""

    from mdprep.config.decisions import proposed_model_charge

    plan = plan_decisions(besd_structure)

    value, terms = proposed_model_charge(
        plan,
        {
            "metal.a_fe2_302_site.charge": 2,
            "ligand.a_akg_304.net_charge": -2,
        },
        "a_fe2_302_site",
    )

    # Fe(2+) + aKG(2-) + Cl(1-) + two neutral histidines.
    assert value == -1
    assert " ".join(terms) == "FE2 +2 AKG304 -2 HIS204 +0 HIS137 +0 CL303 -1"


def test_a_bidentate_cofactor_is_counted_once(besd_structure):
    """alpha-ketoglutarate donates two oxygens and appears twice among the
    donors; counting its charge twice would be off by its whole charge."""

    from mdprep.config.decisions import proposed_model_charge

    plan = plan_decisions(besd_structure)
    confirm = plan.by_id("metal.a_fe2_302_site.confirm_donors")

    donating_atoms = [t for t in confirm.targets[1:] if t.resname == "AKG"]
    assert len(donating_atoms) == 2  # O1 and O5

    _, terms = proposed_model_charge(
        plan,
        {"metal.a_fe2_302_site.charge": 2, "ligand.a_akg_304.net_charge": -2},
        "a_fe2_302_site",
    )
    assert len([term for term in terms if term.startswith("AKG")]) == 1


def test_no_charge_is_proposed_while_a_term_is_still_unknown(besd_structure):
    """Half a sum is worse than no suggestion: it looks like an answer."""

    from mdprep.config.decisions import proposed_answers, proposed_model_charge

    plan = plan_decisions(besd_structure)

    # The cofactor's charge has not been given yet.
    value, terms = proposed_model_charge(
        plan, {"metal.a_fe2_302_site.charge": 2}, "a_fe2_302_site"
    )
    assert value is None
    assert any(term.endswith("?") for term in terms)
    assert "metal.a_fe2_302_site.model_charge" not in proposed_answers(
        plan, {"metal.a_fe2_302_site.charge": 2}
    )

    # An oxidation state left at its proposal (FE2 is +2) counts as known, as
    # the GUI sends it; the cofactor's missing charge still blocks the sum.
    value, terms = proposed_model_charge(plan, {}, "a_fe2_302_site")
    assert value is None and terms[0] == "FE2 +2"

    # And none at all once the metal's own oxidation state has been cleared.
    assert proposed_model_charge(
        plan, {"metal.a_fe2_302_site.charge": ""}, "a_fe2_302_site"
    ) == (None, [])


def test_an_oxidation_state_the_residue_name_settles_is_proposed():
    """The PDB dictionary names an ion after its state where there is more
    than one, so FE2 is evidence and FE is different evidence."""

    from mdprep.config.decisions import proposed_oxidation_state

    assert proposed_oxidation_state("FE2", "Fe")[0] == 2
    assert proposed_oxidation_state("FE", "Fe")[0] == 3
    assert proposed_oxidation_state("CU1", "Cu")[0] == 1
    assert proposed_oxidation_state("ZN", "Zn")[0] == 2


def test_an_ambiguous_metal_with_an_uninformative_name_gets_no_proposal():
    """Boltz names a ligand given as SMILES `LIG`, which says nothing. Iron
    is II or III, and they differ in spin as well as charge, so guessing
    would give a converged calculation with the wrong parameters."""

    from mdprep.config.decisions import proposed_oxidation_state

    value, why = proposed_oxidation_state("LIG", "Fe")

    assert value is None
    assert "more than one common state" in why
    # Zinc is still settled, because it has only the one.
    assert proposed_oxidation_state("LIG", "Zn")[0] == 2


def test_the_model_charge_is_still_proposed_for_a_prediction_naming_everything_lig():
    """The metal's own state cannot be proposed there, but once the user has
    given it the rest of the sum is known -- which is the case that matters,
    because it is the arithmetic rather than the chemistry."""

    from mdprep.config.decisions import proposed_model_charge

    plan = plan_decisions(DATA / "boltz_smiles_ligands.cif")
    site = next(
        decision.id.split(".")[1]
        for decision in plan.decisions
        if decision.section == "metals" and decision.id.endswith(".model")
    )

    # No proposal for the iron itself.
    assert plan.by_id(f"metal.{site}.charge").default is None

    # Found rather than spelled out: the chain this ligand sits in is renamed
    # on the way in, because mmCIF allows a chain id wider than a PDB column.
    cofactor = next(
        decision.id
        for decision in plan.decisions
        if decision.id.endswith("_lig_21.net_charge")
    )

    # The aspartate on the iron is asked its state, and the sum waits for it.
    (aspartate,) = [
        decision.id
        for decision in plan.decisions
        if decision.id.startswith(f"metal.{site}.donor_state.")
    ]
    assert proposed_model_charge(
        plan, {f"metal.{site}.charge": 2, cofactor: -2}, site
    )[0] is None

    value, terms = proposed_model_charge(
        plan,
        {f"metal.{site}.charge": 2, cofactor: -2, aspartate: "ASP"},
        site,
    )

    # Fe(2+) + Asp(-1) + two neutral His + aKG(2-), the cofactor counted once
    # though it donates through two oxygens.
    assert value == -1
    assert len([term for term in terms if term.startswith("LIG21")]) == 1

    # A neutral aspartate counts as one.
    assert proposed_model_charge(
        plan,
        {f"metal.{site}.charge": 2, cofactor: -2, aspartate: "ASH"},
        site,
    )[0] == 0


def test_the_output_directory_is_marked_as_needing_a_filesystem():
    """A browser has no path the user could usefully name: the page hands
    back a bundle, and whoever runs the manifest chooses where output lands.
    The question stays in the plan for front ends that do have a filesystem,
    and keeps a default so the manifest is complete either way."""

    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("project.output_dir")

    assert decision.needs_local_filesystem
    assert decision.default == "prepared/protein_two_ligands"
    assert not decision.requires_user_input
    # It is the only one: nothing else asks for a path.
    assert [d.id for d in plan.decisions if d.needs_local_filesystem] == [
        "project.output_dir"
    ]


def test_the_flag_reaches_a_front_end_through_the_plan():
    """The page filters on this, so it has to survive serialisation."""

    plan = plan_decisions(DATA / "protein_two_ligands.pdb")
    serialised = {item["id"]: item for item in plan.to_dict()["decisions"]}

    assert serialised["project.output_dir"]["needs_local_filesystem"] is True
    assert serialised["project.name"]["needs_local_filesystem"] is False


# --------------------------------------------------------------------------
# A titratable side chain on the metal
#
# Preparation refuses to build a bonded site whose ASP, GLU, CYS, LYS or ARG
# donor has no stated protonation -- coordination does not distinguish a
# carboxylate from a neutral acid, and the two differ by a unit of charge on
# the capped model the parameters come from. That refusal is right. Nothing
# asked the question, so the first a student heard of it was the run
# stopping in Colab, after the install and most of the preparation.
# --------------------------------------------------------------------------


def test_a_titratable_metal_donor_is_asked_about():
    plan = plan_decisions(DATA / "boltz_wide_names.cif")

    asked = [d for d in plan.decisions if ".donor_state." in d.id]

    assert len(asked) == 1, [d.id for d in asked]
    decision = asked[0]
    assert "ASP131" in decision.question
    assert [choice.value for choice in decision.choices] == ["ASP", "ASH"]
    # No default: the geometry shows the contact, not the proton.
    assert decision.default is None
    assert decision.requires_user_input


def test_the_answer_reaches_the_structure_the_refusal_was_about():
    """End to end: the question the builder asks becomes an override that
    resolves to the residue preparation was refusing to guess."""

    from mdprep.config.decisions import proposed_model_charge
    from mdprep.config.models import ManifestConfig
    from mdprep.protonation.overrides import resolve_manual_overrides
    from mdprep.structure.reader import read_structure

    plan = plan_decisions(DATA / "boltz_wide_names.cif")
    answers = {
        "project.name": "t",
        "structure.remove_unknown_heterogens": False,
        "structure.repair.add_missing_heavy_atoms": True,
        "ligand.k_lg2_1.net_charge": 0,
        "ligand.k_lg2_1.smiles": "OCC=O",
        "metal.f_fe_1_site.charge": 2,
        "metal.f_fe_1_site.confirm_donors": True,
        "refinement.total_qm_multiplicity": 5,
        "molecular_dynamics.enabled": False,
        "metal.f_fe_1_site.donor_state.a_asp_131": "ASH",
    }
    answers["metal.f_fe_1_site.model_charge"] = proposed_model_charge(
        plan, answers, "f_fe_1_site"
    )[0]

    manifest = ManifestConfig.model_validate(build_manifest(plan, answers))
    resolved = resolve_manual_overrides(
        read_structure(DATA / "boltz_wide_names.cif"), manifest
    )

    assert [(item.residue.id.display(), item.requested_state) for item in resolved] == [
        ("A:ASP131", "ASH")
    ]


def test_every_titratable_residue_that_becomes_a_bond_is_asked_about():
    """The list the questions come from must be the list the bonds come from.

    They were two different lists with two different radii: bonds are built
    from the confirmed shell at 3.0 A, while the questions were generated
    from the 2.6 A donor list. A side chain between the two was written into
    the manifest as a bond and never asked about, so preparation refused it
    in Colab exactly as before. boltz_far_donor.cif holds an aspartate at
    2.8 A, which is in the gap.
    """

    from mdprep.config.decisions import COORDINATING_DONOR_STATES

    for name in (
        "boltz_far_donor.cif",
        "boltz_wide_names.cif",
        "boltz_smiles_ligands.cif",
        "protein_histidine_zinc.pdb",
    ):
        plan = plan_decisions(DATA / name)
        asked = {
            decision.id.split(".donor_state.")[1]
            for decision in plan.decisions
            if ".donor_state." in decision.id
        }
        for decision in plan.decisions:
            if not decision.id.endswith(".confirm_donors"):
                continue
            for target in decision.targets[1:]:  # exactly what becomes a bond
                resname = target.resname.strip().upper()
                if resname not in COORDINATING_DONOR_STATES:
                    continue
                key = "_".join(
                    part
                    for part in (
                        target.chain.strip().lower(),
                        resname.lower(),
                        str(target.resid),
                    )
                    if part
                )
                assert key in asked, (
                    f"{name}: {target.chain}:{resname}{target.resid} is bonded "
                    "to the metal but nothing asks its protonation"
                )


def test_the_aspartate_in_the_radius_gap_is_asked_about():
    plan = plan_decisions(DATA / "boltz_far_donor.cif")

    asked = [d for d in plan.decisions if ".donor_state." in d.id]

    assert [d.id for d in asked] == ["metal.f_fe_1_site.donor_state.a_asp_131"]
    assert [c.value for c in asked[0].choices] == ["ASP", "ASH"]
