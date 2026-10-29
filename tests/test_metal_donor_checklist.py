"""Which residues near a metal are bonded to it: a checklist, every box ticked.

A substrate docked too close to an iron with no water of its own sits within
bonding distance, and was made one of the iron's bonds with no way to say
otherwise short of writing the site by hand. Each residue in range now has a
box, ticked to begin with; an unticked one stays where it is, unbonded, and
MCPB.py's bonding cutoff is placed below it, since MCPB.py bonds every donor
within its cutoff whatever the bond list says.

The fixture is the SyrB2 iron site with the water on the iron replaced by
ethanol, docked where the water was; see tests/data/metal_sites/README.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdprep.config.decisions import (
    DecisionError,
    answer_conflicts,
    build_manifest,
    coerce_answer,
    mcpb_cutoff_for,
    plan_decisions,
    proposed_answers,
    resolve_answers,
    unresolved_mcpb_sites,
)
from mdprep.config.models import ManifestConfig
from mdprep.metals.coordination import MetalCoordinationError, resolve_metal_sites
from mdprep.structure.pdb import read_pdb
from mdprep.workflows.wizard import collect_answers, format_question


DOCKED = Path("tests/data/metal_sites/syrb2_site_docked_substrate.pdb")
SITE = "a_fe2_900_site"
CHECKLIST = f"metal.{SITE}.confirm_donors"
WITHOUT_SUBSTRATE = ["A:AKG897", "A:HIS116", "A:HIS235", "A:CL901"]


@pytest.fixture(scope="module")
def plan():
    return plan_decisions(DOCKED)


def _answers(**extra) -> dict:
    answers = {
        "structure.remove_unknown_heterogens": "true",
        "structure.repair.add_missing_heavy_atoms": "true",
        "ligand.a_akg_897.net_charge": "-2",
        "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
        "ligand.a_sub_950.net_charge": "0",
        "ligand.a_sub_950.smiles": "CCO",
        "metal.a_fe2_900_site.charge": "2",
        "refinement.total_qm_multiplicity": "5",
    }
    answers.update(extra)
    return answers


def _built(plan, **extra) -> dict:
    answers = _answers(**extra)
    answers.setdefault(
        f"metal.{SITE}.model_charge", proposed_answers(plan, answers)[f"metal.{SITE}.model_charge"]
    )
    manifest = build_manifest(plan, answers)
    ManifestConfig.model_validate(manifest)
    return manifest


def _bonded(manifest) -> list[str]:
    site = manifest["metals"][0]["mcpb"]
    return [
        f"{bond['coordinator']['chain']}:{bond['coordinator']['resname']}"
        f"{bond['coordinator']['resid']} {bond['coordinator']['atom_name']}"
        for bond in site["bonds"]
    ]


# --------------------------------------------------------------------------
# The question
# --------------------------------------------------------------------------


def test_each_residue_in_range_has_a_box_nearest_first_all_ticked(plan):
    checklist = plan.by_id(CHECKLIST)

    assert checklist.kind == "multichoice"
    assert [choice.value for choice in checklist.choices] == [
        "A:AKG897", "A:HIS116", "A:HIS235", "A:SUB950", "A:CL901",
    ]
    # A bidentate cofactor is one box with both of its atoms.
    assert checklist.choices[0].detail == "O2 at 2.03 A, O5 at 2.19 A"
    assert coerce_answer(checklist, checklist.default) == tuple(
        choice.value for choice in checklist.choices
    )
    # Still the user's to confirm: the boxes are a proposal, not an answer.
    assert checklist.requires_user_input


@pytest.mark.parametrize(
    ("raw", "ticked"),
    [
        (True, ("A:AKG897", "A:HIS116", "A:HIS235", "A:SUB950", "A:CL901")),
        ("true", ("A:AKG897", "A:HIS116", "A:HIS235", "A:SUB950", "A:CL901")),
        (False, ()),
        ("none", ()),
        (["A:CL901", "A:HIS116"], ("A:HIS116", "A:CL901")),
        ("A:HIS116, A:CL901", ("A:HIS116", "A:CL901")),
    ],
)
def test_an_answer_reads_as_the_boxes_it_ticks(plan, raw, ticked):
    """Yes and no tick all or none, as answers given to the yes/no question did."""
    assert coerce_answer(plan.by_id(CHECKLIST), raw) == ticked


def test_a_residue_not_offered_cannot_be_ticked(plan):
    with pytest.raises(DecisionError, match="A:HIS999"):
        coerce_answer(plan.by_id(CHECKLIST), ["A:HIS999"])


# --------------------------------------------------------------------------
# What unticking does to the manifest
# --------------------------------------------------------------------------


def test_everything_ticked_builds_the_site_as_before(plan):
    manifest = _built(plan, **{CHECKLIST: "true"})

    assert manifest["metals"][0]["mcpb"]["cutoff_angstrom"] == 2.6
    assert "A:SUB950 O1" in _bonded(manifest)


def test_an_unticked_substrate_is_left_out_of_the_bonds(plan):
    manifest = _built(
        plan,
        **{CHECKLIST: WITHOUT_SUBSTRATE, "ligand.a_sub_950.charge_method": "am1bcc"},
    )

    assert "A:SUB950 O1" not in _bonded(manifest)
    assert set(_bonded(manifest)) == {
        "A:AKG897 O2", "A:AKG897 O5", "A:HIS116 NE2", "A:HIS235 NE2", "A:CL901 CL",
    }
    # Kept as a ligand of its own, still in the QM region of the refinement.
    assert {"a_sub_950"} <= set(manifest["refinement"]["qm_components"]["ligands"])


def test_mcpb_cutoff_drops_below_what_was_left_out(plan):
    """MCPB.py bonds every donor within its cutoff; the substrate's O1 is 2.24 A away."""
    manifest = _built(
        plan,
        **{CHECKLIST: WITHOUT_SUBSTRATE, "ligand.a_sub_950.charge_method": "am1bcc"},
    )

    assert manifest["metals"][0]["mcpb"]["cutoff_angstrom"] == 2.19


def test_the_manifest_passes_the_check_that_mcpb_bonds_nothing_undeclared(plan):
    """And would not with the usual cutoff, which is the point of lowering it."""
    manifest = _built(
        plan,
        **{CHECKLIST: WITHOUT_SUBSTRATE, "ligand.a_sub_950.charge_method": "am1bcc"},
    )
    structure = read_pdb(DOCKED)

    sites = resolve_metal_sites(structure, ManifestConfig.model_validate(manifest))

    found = {
        f"{bond.coordinator.resname}{bond.coordinator.resid} {bond.coordinator.name}": bond.discovered_by_cutoff
        for bond in sites[0].bonds
    }
    # Within 2.19 A MCPB.py finds them itself; beyond it they are handed to
    # MCPB.py by name (add_bonded_pairs), so lowering the cutoff loses none.
    assert found == {
        "AKG897 O2": True,
        "HIS116 NE2": True,
        "AKG897 O5": True,
        "HIS235 NE2": False,
        "CL901 CL": False,
    }
    manifest["metals"][0]["mcpb"]["cutoff_angstrom"] = 2.6
    with pytest.raises(MetalCoordinationError, match="SUB950"):
        resolve_metal_sites(structure, ManifestConfig.model_validate(manifest))


def test_the_model_charge_proposal_counts_only_what_is_ticked(plan):
    answers = _answers(**{CHECKLIST: ["A:AKG897", "A:HIS116", "A:HIS235"]})

    proposal = proposed_answers(plan, answers)[f"metal.{SITE}.model_charge"]

    # Fe(II) +2, AKG -2, two neutral histidines; the chloride is unticked.
    assert proposal == "0"


def test_a_substrate_left_unticked_cannot_take_its_charges_from_the_fit(plan):
    """Said at the checklist, where it was unticked, with what to do."""
    conflict = answer_conflicts(
        plan, _answers(**{CHECKLIST: WITHOUT_SUBSTRATE, "ligand.a_sub_950.charge_method": "mcpb_resp_pyscf"})
    )

    assert conflict is not None
    assert conflict.decision_id == CHECKLIST
    assert "you left it unticked" in str(conflict)
    assert "such as AM1-BCC" in str(conflict)


def test_the_charge_method_question_says_what_to_do_for_a_ligand_only_near_the_metal(plan):
    why = plan.by_id("ligand.a_sub_950.charge_method").why

    assert "docked against it -- choose AM1-BCC" in why


def test_nothing_ticked_leaves_the_site_to_be_written_by_hand(plan):
    answers = _answers(**{CHECKLIST: [], "ligand.a_sub_950.charge_method": "am1bcc",
                          "ligand.a_akg_897.charge_method": "am1bcc"})

    assert unresolved_mcpb_sites(plan, answers) == [SITE]
    assert CHECKLIST in resolve_answers(plan, answers)


def test_an_unticked_atom_nearer_than_a_contact_is_refused(plan):
    """An atom 1.4 A from a metal is a clash, which unticking would hide."""
    from dataclasses import replace

    findings = dict(plan.findings)
    findings["metal_donors"] = {
        SITE: [
            {**item, "distance": 1.4} if item["residue"] == "A:SUB950" else item
            for item in plan.findings["metal_donors"][SITE]
        ]
    }
    clashing = replace(plan, findings=findings)
    resolved = resolve_answers(
        clashing, _answers(**{CHECKLIST: WITHOUT_SUBSTRATE, "ligand.a_sub_950.charge_method": "am1bcc",
                              f"metal.{SITE}.model_charge": "-1"})
    )

    with pytest.raises(DecisionError, match="clash"):
        mcpb_cutoff_for(clashing, resolved, SITE)


# --------------------------------------------------------------------------
# Copies of one protein
# --------------------------------------------------------------------------


def test_a_copy_unticks_its_own_counterpart():
    """B's chloride is unticked when A's is, so the shared fit is one site's."""
    plan = plan_decisions("examples/tutorials/syrb2_2fct/2FCT.pdb")
    answers = {
        "structure.chains": "all",
        "structure.same_parametrization": "true",
        "structure.remove_unknown_heterogens": "true",
        "structure.repair.add_missing_heavy_atoms": "true",
        "ligand.a_dsu_896.include": "false",
        "ligand.a_akg_897.net_charge": "-2",
        "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
        "metal.a_fe2_900_site.charge": "2",
        "metal.a_fe2_900_site.confirm_donors": ["A:AKG897", "A:HIS116", "A:HIS235", "A:HOH1072"],
        "refinement.total_qm_multiplicity": "5",
    }
    answers["metal.a_fe2_900_site.model_charge"] = proposed_answers(plan, answers)[
        "metal.a_fe2_900_site.model_charge"
    ]

    manifest = build_manifest(plan, answers)

    sites = {site["id"]: site for site in manifest["metals"]}
    copy = sites["b_fe2_899_site"]
    assert not any(
        bond["coordinator"]["resname"] == "CL" for bond in copy["mcpb"]["bonds"]
    )
    assert copy["mcpb"]["cutoff_angstrom"] < 2.414


# --------------------------------------------------------------------------
# The command-line wizard
# --------------------------------------------------------------------------


def test_the_wizard_shows_the_boxes_ticked(plan):
    text = format_question(plan.by_id(CHECKLIST))

    assert "[x] A:SUB950: O1 at 2.24 A" in text
    assert "press Enter to keep the ticked ones" in text


def test_the_wizard_takes_enter_as_the_boxes_shown_and_a_list_as_those_ticked(plan):
    def prompter(answers):
        def prompt(decision):
            return answers.get(decision.id, "")
        return prompt

    preset = {
        key: value
        for key, value in _answers(**{"ligand.a_sub_950.charge_method": "am1bcc"}).items()
    }
    taken, _ = collect_answers(plan, prompter({}), preset=preset)
    assert taken[CHECKLIST] == ("A:AKG897", "A:HIS116", "A:HIS235", "A:SUB950", "A:CL901")

    taken, _ = collect_answers(
        plan, prompter({CHECKLIST: "A:AKG897,A:HIS116,A:HIS235,A:CL901"}), preset=preset
    )
    assert taken[CHECKLIST] == tuple(WITHOUT_SUBSTRATE)
