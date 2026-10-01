"""The BesD reference case: an alpha-ketoglutarate-dependent halogenase.

This is the workflow the student-facing builder is aimed at, so the properties
it depends on are pinned here rather than left to be rediscovered: that the
iron's coordination shell is detectable, that chain selection isolates one copy
of the complex, and that the resulting manifest is the one mdprep accepts.
"""

from pathlib import Path

import pytest
import yaml

from mdprep.config.decisions import (
    COORDINATION_CUTOFF_ANGSTROM,
    metal_donor_atoms,
    plan_decisions,
)
from mdprep.config.models import ManifestConfig
from mdprep.structure.inspect import inspect_pdb_structure


EXAMPLE = Path("examples/16_besd_halogenase_gfn2_mcpb.yaml")
# 6NIE itself is a 1.4 MB four-chain deposition and is not vendored; the tests
# that need coordinates skip without it.
STRUCTURE = Path("/tmp/6NIE.pdb")

needs_structure = pytest.mark.skipif(
    not STRUCTURE.exists(),
    reason="6NIE is not available locally; fetch it from RCSB to run this test",
)


def load_example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


def test_the_besd_example_is_a_valid_manifest():
    ManifestConfig.model_validate(load_example())


def test_the_example_builds_one_chain_only():
    """MCPB inputs are single-chain and 6NIE holds four copies."""
    assert load_example()["structure"]["keep_chains"] == ["A"]


def test_the_iron_is_high_spin_iron_two():
    site = load_example()["metals"][0]
    ion = site["ions"][0]

    assert ion["element"] == "Fe"
    assert ion["charge"] == 2
    assert ion["multiplicity"] == 5  # high-spin d6


def test_the_model_charge_matches_the_declared_fragment():
    """Fe(2+) + AKG(2-) + Cl(1-) + 2 x His(0) = -1."""
    mcpb = load_example()["metals"][0]["mcpb"]

    assert mcpb["small_model_charge"] == -1
    assert mcpb["large_model_charge"] == -1
    assert mcpb["small_model_spin"] == mcpb["large_model_spin"] == 5


def test_the_hessian_and_the_refinement_both_use_gfn2():
    example = load_example()
    pyscf = example["metals"][0]["mcpb"]["pyscf"]

    assert pyscf["hessian_backend"] == "xtb"
    assert pyscf["xtb"]["model"] == "gfn2"
    assert pyscf["geometry_source"] == "qmmm_refinement"
    assert example["refinement"]["qm_method"] == "gfn2_xtb"


def test_the_coordinating_cofactor_is_not_given_independent_charges():
    """alpha-ketoglutarate bonds the iron, so AM1-BCC would be inconsistent."""
    ligands = {entry["id"]: entry for entry in load_example()["ligands"]}

    assert ligands["a_akg_304"]["charge_method"] == "mcpb_resp_pyscf"
    assert ligands["a_lys_301"]["charge_method"] == "am1bcc"  # substrate, not bonded


def test_every_declared_bond_names_a_donor_atom():
    bonds = load_example()["metals"][0]["mcpb"]["bonds"]

    donors = {(b["coordinator"]["resname"], b["coordinator"]["atom_name"]) for b in bonds}
    assert donors == {
        ("AKG", "O1"),
        ("AKG", "O5"),
        ("HIS", "NE2"),
        ("CL", "CL"),
    }
    assert len(bonds) == 5  # two histidines contribute the same atom name


@needs_structure
def test_detection_finds_exactly_the_declared_coordination_shell():
    """The donor search must reproduce what the example declares by hand."""
    summary = inspect_pdb_structure(STRUCTURE)
    iron = next(r for r in summary.metal_ions if r.id.chain_id == "A")

    donors = metal_donor_atoms(summary.structure.residues, iron)

    found = {(res.id.resname, atom.name) for res, atom, _ in donors}
    declared = {
        (b["coordinator"]["resname"], b["coordinator"]["atom_name"])
        for b in load_example()["metals"][0]["mcpb"]["bonds"]
    }
    assert found == declared
    assert len(donors) == 5


@needs_structure
def test_the_cutoff_excludes_the_carbons_behind_the_carboxylate():
    """A 3.0 A search picks up AKG C1/C2, which are not donors."""
    summary = inspect_pdb_structure(STRUCTURE)
    iron = next(r for r in summary.metal_ions if r.id.chain_id == "A")

    donors = metal_donor_atoms(summary.structure.residues, iron)

    assert all(atom.element.upper() != "C" for _, atom, _ in donors)
    assert max(distance for _, _, distance in donors) <= COORDINATION_CUTOFF_ANGSTROM


@needs_structure
def test_choosing_one_chain_prunes_the_other_copies():
    plan = plan_decisions(STRUCTURE)

    everything = plan.applicable({"structure.chains": "all"})
    chain_a = plan.applicable({"structure.chains": "A"})

    ligands_all = [d for d in everything if d.id.endswith(".include")]
    ligands_a = [d for d in chain_a if d.id.endswith(".include")]
    assert len(ligands_all) == 12  # four chains x three heterogens
    assert len(ligands_a) == 3
    assert all(".a_" in d.id for d in ligands_a)


@needs_structure
def test_the_chain_question_has_no_default_when_a_metal_is_present():
    """Building all four copies would make a bonded MCPB site impossible."""
    plan = plan_decisions(STRUCTURE)

    decision = plan.by_id("structure.chains")

    assert decision.default is None
    assert decision.requires_user_input is True


BESD_ANSWERS = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": False,
    "protonation.method": "propka_xtb_his",
    "ligand.a_lys_301.net_charge": 1,
    "ligand.a_akg_304.net_charge": -2,
    "ligand.a_cl_303.net_charge": -1,
    "metal.a_fe2_302_site.model": "bonded_mcpb",
    "metal.a_fe2_302_site.charge": 2,
    "metal.a_fe2_302_site.confirm_donors": True,
    "metal.a_fe2_302_site.multiplicity": 5,
    "metal.a_fe2_302_site.model_charge": -1,
}


@needs_structure
def generated() -> dict:
    from mdprep.config.decisions import build_manifest

    return build_manifest(plan_decisions(STRUCTURE), BESD_ANSWERS)


@needs_structure
def test_the_builder_generates_a_valid_besd_manifest():
    ManifestConfig.model_validate(generated())


@needs_structure
def test_the_generated_site_reproduces_the_hand_written_bonds():
    produced = {
        (b["coordinator"]["resname"], b["coordinator"]["resid"], b["coordinator"]["atom_name"])
        for b in generated()["metals"][0]["mcpb"]["bonds"]
    }
    declared = {
        (b["coordinator"]["resname"], b["coordinator"]["resid"], b["coordinator"]["atom_name"])
        for b in load_example()["metals"][0]["mcpb"]["bonds"]
    }

    assert produced == declared


@needs_structure
def test_the_generated_manifest_uses_gfn2_throughout():
    manifest = generated()

    assert manifest["metals"][0]["mcpb"]["pyscf"]["hessian_backend"] == "xtb"
    assert manifest["metals"][0]["mcpb"]["pyscf"]["xtb"]["model"] == "gfn2"
    assert manifest["refinement"]["qm_method"] == "gfn2_xtb"
    assert manifest["refinement"]["enabled"] is True


@needs_structure
def test_the_coordinating_cofactor_defaults_to_the_metal_site_fit():
    """alpha-ketoglutarate bonds the iron, so AM1-BCC would be inconsistent."""
    ligands = {entry["id"]: entry for entry in generated()["ligands"]}

    assert ligands["a_akg_304"]["charge_method"] == "mcpb_resp_pyscf"
    assert ligands["a_lys_301"]["charge_method"] == "am1bcc"


@needs_structure
def test_only_the_selected_chain_reaches_the_manifest():
    manifest = generated()

    assert manifest["structure"]["keep_chains"] == ["A"]
    assert all(entry["selector"]["chain"] == "A" for entry in manifest["ligands"])
    assert len(manifest["metals"]) == 1


@needs_structure
def test_the_spin_state_is_carried_into_every_model():
    site = generated()["metals"][0]

    assert site["ions"][0]["multiplicity"] == 5
    assert site["mcpb"]["small_model_spin"] == 5
    assert site["mcpb"]["large_model_spin"] == 5
    assert generated()["refinement"]["total_qm_multiplicity"] == 5


@needs_structure
def test_the_toolchain_for_this_run_includes_xtb_propka_and_pyscf():
    from mdprep.workflows.runtime import conda_packages_for

    packages = conda_packages_for(generated())

    assert "ambertools" in packages
    assert "propka" in packages          # propka_xtb_his
    assert "xtb" in packages             # histidine tautomers and the Hessian
    assert "pyscf" in packages           # bonded MCPB large-model ESP
