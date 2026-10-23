"""The spin state of the QM region is stated by the user, not inferred.

mdprep used to take the region's multiplicity from the first open-shell
component it found. For BesD that gives 5, and it is right: the high-spin
Fe(II) is the only unpaired centre there. It is right by accident. The QM
region is one quantum system with one spin state, and once a second
open-shell component is in it -- a substrate radical, a second metal -- the
total depends on whether the spins align, which is a fact about the chemistry
and not a function of the parts.

So it is asked, in the GUI and in the wizard, and a manifest that leaves it
out of an open-shell region is refused. It is the only spin question: one site
is fitted per preparation, so the region's spin is also the spin of the site's
MCPB.py models, and asking the metal's as well asked the same thing twice.
"""

import pytest
from pydantic import ValidationError

from mdprep.config.decisions import (
    DecisionError,
    build_manifest,
    plan_decisions,
    proposed_answers,
    resolve_answers,
)
from mdprep.config.models import ManifestConfig


STRUCTURE = "tests/data/protein_histidine_zinc.pdb"

BONDED = {
    "structure.chains": "all",
    "structure.repair.add_missing_heavy_atoms": True,
    "metal.z_zn_500_site.model": "bonded_mcpb",
    "metal.z_zn_500_site.charge": 2,
    "metal.z_zn_500_site.confirm_donors": True,
    "metal.z_zn_500_site.model_charge": 2,
}


def _plan():
    return plan_decisions(STRUCTURE)


def test_the_question_is_asked_and_has_no_default():
    decision = _plan().by_id("refinement.total_qm_multiplicity")

    assert decision.requires_user_input is True
    assert decision.default is None
    assert decision.minimum == 1
    assert "QM region" in decision.question


def test_it_is_the_only_spin_question():
    plan = _plan()

    assert not [d.id for d in plan.decisions if d.id.endswith(".multiplicity") and d.section == "metals"]
    assert "high-spin" in plan.by_id("refinement.total_qm_multiplicity").why


def test_it_is_blocking_until_answered():
    plan = _plan()
    answers = dict(BONDED)
    answers.pop("structure.chains")

    blocking = {decision.id for decision in plan.blocking(BONDED)}

    assert "refinement.total_qm_multiplicity" in blocking
    with pytest.raises(DecisionError, match="refinement.total_qm_multiplicity"):
        resolve_answers(plan, BONDED)


def test_it_is_not_asked_when_no_bonded_site_will_be_refined():
    """A nonbonded ion runs no QM/MM refinement, so there is no region to
    have a spin state, and a question with no use is a question not asked."""

    plan = _plan()
    nonbonded = {**BONDED, "metal.z_zn_500_site.model": "nonbonded"}

    asked = {decision.id for decision in plan.applicable(nonbonded)}

    assert "refinement.total_qm_multiplicity" not in asked


def test_it_is_not_asked_when_the_coordination_shell_was_rejected():
    """A rejected shell generates no site, so nothing is refined either."""

    plan = _plan()
    rejected = {**BONDED, "metal.z_zn_500_site.confirm_donors": False}

    asked = {decision.id for decision in plan.applicable(rejected)}

    assert "refinement.total_qm_multiplicity" not in asked


def test_the_answer_is_the_region_s_and_the_site_s():
    """The site's ion and its MCPB.py models take the region's spin: one
    site is fitted, and the cofactors in it are closed shell."""

    manifest = build_manifest(
        _plan(), {**BONDED, "refinement.total_qm_multiplicity": 3}
    )

    assert manifest["refinement"]["total_qm_multiplicity"] == 3
    [site] = manifest["metals"]
    assert site["ions"][0]["multiplicity"] == 3
    assert site["mcpb"]["small_model_spin"] == site["mcpb"]["large_model_spin"] == 3
    ManifestConfig.model_validate(manifest)


def test_nothing_is_proposed_for_it():
    """The spin is chemistry the structure does not show."""

    assert "refinement.total_qm_multiplicity" not in proposed_answers(_plan(), BONDED)


def test_an_open_shell_region_without_a_total_is_refused_by_the_schema():
    """The path a hand-written manifest takes, which never sees the planner."""

    manifest = build_manifest(
        _plan(),
        {
            **BONDED,
            "refinement.total_qm_multiplicity": 5,
        },
    )
    manifest["refinement"].pop("total_qm_multiplicity")

    with pytest.raises(ValidationError, match="open-shell QM region"):
        ManifestConfig.model_validate(manifest)


def test_the_selection_refuses_rather_than_guessing_if_it_is_reached():
    """The schema stops this first. If something builds a selection around
    the schema, guessing here would run the whole refinement at a spin state
    nobody chose, and report nothing."""

    from mdprep.refinement.selection import (
        RefinementSelectionError,
        _total_multiplicity,
    )

    manifest = ManifestConfig.model_validate(
        build_manifest(
            _plan(),
            {
                **BONDED,
                "refinement.total_qm_multiplicity": 5,
            },
        )
    )
    manifest.refinement.total_qm_multiplicity = None

    with pytest.raises(RefinementSelectionError, match="total_qm_multiplicity"):
        _total_multiplicity(manifest)
