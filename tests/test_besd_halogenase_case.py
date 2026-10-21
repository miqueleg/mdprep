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
STRUCTURE = Path("examples/tutorials/besd_6nie/6NIE.pdb")

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


def test_the_metal_hessian_uses_gxtb_and_the_esp_uses_gfn2():
    """GFN2 gives this iron's Fe-O5 bond a negative force constant, so the
    Hessian takes the default g-xTB. The ESP has no such problem and stays on
    GFN2, which is where the cost would otherwise be."""

    from mdprep.config.models import McpbXtbHessianConfig

    example = load_example()
    pyscf = example["metals"][0]["mcpb"]["pyscf"]

    assert pyscf["hessian_backend"] == "xtb"
    # The example leaves the model unset, so it must be the default.
    assert "model" not in pyscf["xtb"]
    assert McpbXtbHessianConfig().model == "gxtb"
    assert pyscf["esp_backend"] == "xtb"
    assert pyscf["geometry_source"] == "qmmm_refinement"
    assert example["refinement"]["qm_method"] == "gfn2_xtb"


def test_the_coordinating_cofactor_is_not_given_independent_charges():
    """alpha-ketoglutarate bonds the iron, so AM1-BCC would be inconsistent."""
    ligands = {entry["id"]: entry for entry in load_example()["ligands"]}

    assert ligands["a_akg_304"]["charge_method"] == "mcpb_resp_pyscf"
    assert ligands["a_lig_301"]["charge_method"] == "am1bcc"  # substrate, not bonded


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
def test_choosing_one_chain_prunes_the_other_copies(besd_structure):
    plan = plan_decisions(besd_structure)

    everything = plan.applicable(
        {"structure.chains": "all", "structure.same_parametrization": False}
    )
    chain_a = plan.applicable({"structure.chains": "A"})

    ligands_all = [d for d in everything if d.id.endswith(".include")]
    ligands_a = [d for d in chain_a if d.id.endswith(".include")]
    assert len(ligands_all) == 8  # four chains x two parameterised heterogens
    assert len(ligands_a) == 2  # the chloride belongs to the metal site
    assert all(".a_" in d.id for d in ligands_a)


@needs_structure
def test_sharing_parameters_asks_only_what_differs_between_the_copies(besd_structure):
    """Four copies, asked about once -- except where a copy really differs.

    Chains B and C hold alpha-ketoglutarate on the iron through O2 where A
    and D use O1: the two oxygens of one carboxylate, named the other way
    round. A template is by atom name, so those two are not copies of A's
    site and are asked about on their own. B's second, surface iron has no
    counterpart at all."""

    plan = plan_decisions(besd_structure)
    asked = plan.applicable(
        {"structure.chains": "all", "structure.same_parametrization": True}
    )

    ligands = sorted(d.id for d in asked if d.id.endswith(".include"))
    sites = sorted(d.id for d in asked if d.id.endswith(".model"))
    assert ligands == [
        "ligand.a_akg_304.include",
        "ligand.a_lig_301.include",
        "ligand.b_akg_304.include",
        "ligand.c_akg_304.include",
    ]
    assert sites == [
        "metal.a_fe2_302_site.model",
        "metal.b_fe2_302_site.model",
        "metal.b_fe_305_site.model",
        "metal.c_fe2_302_site.model",
    ]
    shared = plan.by_id("structure.same_parametrization")
    assert "D:FE2302 would share A:FE2302's parameters" in shared.evidence
    assert any(
        line.startswith("Asked about on its own either way: B:FE2302")
        and "A:AKG304 O1" in line
        and "B:AKG304 O2" in line
        for line in shared.evidence
    )


@needs_structure
def test_the_chain_question_has_no_default_when_a_metal_is_present(besd_structure):
    """Building all four copies would make a bonded MCPB site impossible."""
    plan = plan_decisions(besd_structure)

    decision = plan.by_id("structure.chains")

    assert decision.default is None
    assert decision.requires_user_input is True


@needs_structure
def test_a_surface_iron_with_waters_no_longer_hides_the_structure_questions(besd_structure):
    """6NIE has an iron on the surface of chain B with three waters on it,
    and any water on any metal used to suppress both the crystal-water and
    the unknown-heterogen question -- for all 847 waters, and even after
    chain A, which has no such water, was chosen. The manifest then got
    keep_crystal_waters: true and remove_unknown_heterogens: false without
    either being asked, and the crystal-water decision is a workshop topic."""

    plan = plan_decisions(besd_structure)
    asked = {decision.id for decision in plan.applicable({"structure.chains": "A"})}

    assert {"structure.keep_crystal_waters", "structure.remove_unknown_heterogens"} <= asked
    waters = plan.by_id("structure.keep_crystal_waters")
    assert waters.evidence[0] == "847 water residues in the input"
    assert waters.evidence[1] == (
        "Bound to a metal, and kept with it whatever you answer: B:HOH464, B:HOH412, D:HOH410"
    )
    heterogens = plan.by_id("structure.remove_unknown_heterogens")
    assert heterogens.requires_user_input and heterogens.default is None
    assert "A:LIG301" in heterogens.evidence and "A:AKG304" in heterogens.evidence


BESD_ANSWERS = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": False,
    "structure.repair.add_missing_heavy_atoms": True,
    "protonation.method": "propka_xtb_his",
    "ligand.a_lig_301.net_charge": 1,
    # 6NIE resolves no hydrogens on the free lysine, and a net charge does not
    # say where they go, so the protonation is stated.
    "ligand.a_lig_301.smiles": "C(CCN)CC(C(=O)O)N",
    "ligand.a_akg_304.net_charge": -2,
    "ligand.a_cl_303.net_charge": -1,
    "metal.a_fe2_302_site.model": "bonded_mcpb",
    "metal.a_fe2_302_site.charge": 2,
    "metal.a_fe2_302_site.confirm_donors": True,
    # The region as a whole, which mdprep asks for rather than taking
    # from the iron: here the iron is the only open shell, so it agrees.
    "refinement.total_qm_multiplicity": 5,
    "metal.a_fe2_302_site.model_charge": -1,
}


def generated(structure) -> dict:
    from mdprep.config.decisions import build_manifest

    return build_manifest(plan_decisions(structure), BESD_ANSWERS)


@needs_structure
def test_the_builder_generates_a_valid_besd_manifest(besd_structure):
    ManifestConfig.model_validate(generated(besd_structure))


@needs_structure
def test_the_generated_site_reproduces_the_hand_written_bonds(besd_structure):
    produced = {
        (b["coordinator"]["resname"], b["coordinator"]["resid"], b["coordinator"]["atom_name"])
        for b in generated(besd_structure)["metals"][0]["mcpb"]["bonds"]
    }
    declared = {
        (b["coordinator"]["resname"], b["coordinator"]["resid"], b["coordinator"]["atom_name"])
        for b in load_example()["metals"][0]["mcpb"]["bonds"]
    }

    assert produced == declared


@needs_structure
def test_the_generated_manifest_uses_gxtb_for_the_metal_and_gfn2_elsewhere(besd_structure):
    """The Hessian is the one place GFN2 is not good enough: on this iron it
    gives a metal-ligand bond a negative force constant. The ESP and the
    refinement have no such trouble and stay on GFN2, which is where the cost
    would otherwise be."""

    pyscf = generated(besd_structure)["metals"][0]["mcpb"]["pyscf"]

    assert pyscf["hessian_backend"] == "xtb"
    assert pyscf["xtb"]["model"] == "gxtb"
    assert pyscf["esp_backend"] == "xtb"
    # No executable: naming "xtb" would hand a g-xTB run the GFN-only binary.
    assert "executable" not in pyscf["xtb"]
    assert generated(besd_structure)["refinement"]["qm_method"] == "gfn2_xtb"
    assert generated(besd_structure)["refinement"]["enabled"] is True


@needs_structure
def test_the_coordinating_cofactor_defaults_to_the_metal_site_fit(besd_structure):
    """alpha-ketoglutarate bonds the iron, so AM1-BCC would be inconsistent."""
    ligands = {entry["id"]: entry for entry in generated(besd_structure)["ligands"]}

    assert ligands["a_akg_304"]["charge_method"] == "mcpb_resp_pyscf"
    assert ligands["a_lig_301"]["charge_method"] == "am1bcc"


@needs_structure
def test_only_the_selected_chain_reaches_the_manifest(besd_structure):
    manifest = generated(besd_structure)

    assert manifest["structure"]["keep_chains"] == ["A"]
    assert all(entry["selector"]["chain"] == "A" for entry in manifest["ligands"])
    assert len(manifest["metals"]) == 1


@needs_structure
def test_the_spin_state_is_carried_into_every_model(besd_structure):
    site = generated(besd_structure)["metals"][0]

    assert site["ions"][0]["multiplicity"] == 5
    assert site["mcpb"]["small_model_spin"] == 5
    assert site["mcpb"]["large_model_spin"] == 5
    assert generated(besd_structure)["refinement"]["total_qm_multiplicity"] == 5


@needs_structure
def test_the_toolchain_for_this_run_includes_xtb_propka_and_pyscf(besd_structure):
    from mdprep.workflows.runtime import (
        conda_packages_for,
        gxtb_bin_directory,
        micromamba_setup_commands,
    )

    manifest = generated(besd_structure)
    packages = conda_packages_for(manifest)

    assert "ambertools" in packages
    assert "propka" in packages          # propka_xtb_his
    assert "pyscf" in packages           # bonded MCPB large-model ESP

    # xtb is needed for the histidine tautomers and the Hessian, but this run
    # asks for a g-xTB Hessian, and that build is an ordinary xtb with the
    # g-xTB method added. Installing conda's as well would be a second copy.
    assert "xtb" not in packages
    setup = "\n".join(micromamba_setup_commands(manifest))
    assert f"export PATH={gxtb_bin_directory()}:$PATH" in setup


# --------------------------------------------------------------------------
# A distance cutoff does not respect a coordination sphere
# --------------------------------------------------------------------------


def histidine_cluster(resid: int):
    """The xTB cluster around one chain-A histidine, before and after."""
    from mdprep.config.loader import load_manifest
    from mdprep.protonation.histidine_xtb import (
        build_histidine_cluster,
        complete_metal_spheres,
    )
    from mdprep.structure.classify import is_histidine
    from mdprep.structure.pdb import read_pdb

    manifest = load_manifest(EXAMPLE, resolve_paths=True)
    structure = read_pdb(STRUCTURE)
    histidine = next(
        r
        for r in structure.residues
        if is_histidine(r) and r.id.chain_id == "A" and r.id.resid == resid
    )
    cutoff = manifest.protonation.histidine.xtb.cutoff_angstrom
    base = build_histidine_cluster(structure, histidine, cutoff_angstrom=cutoff)
    full, added = complete_metal_spheres(structure, base, manifest)
    return base, full, added


def test_the_cutoff_alone_cuts_through_the_iron_coordination_sphere():
    """His134 is 5 A from the iron but further from the chloride bonded to it.

    The resulting fragment has an iron without one of its ligands, so its
    charge and electron count are artefacts of where the cut fell. This is the
    state the fix exists to repair, so it is asserted rather than assumed.
    """
    base, _, _ = histidine_cluster(134)

    names = {residue.id.resname for residue in base}
    assert "FE2" in names
    assert "CL" not in names


def test_completing_the_sphere_brings_the_bonded_chloride_in():
    _, full, added = histidine_cluster(134)

    assert "A:CL303" in added
    assert "CL" in {residue.id.resname for residue in full}


def test_a_cluster_already_holding_the_whole_sphere_is_left_alone():
    """His204 coordinates the iron and its cluster already has everything."""
    base, full, added = histidine_cluster(204)

    assert added == []
    assert len(full) == len(base)


def test_a_cluster_with_no_metal_is_untouched():
    base, full, added = histidine_cluster(42)

    assert "FE2" not in {residue.id.resname for residue in base}
    assert added == []
    assert full is base


def test_completion_keeps_the_cluster_in_structure_order():
    """A reproducible cluster is a reproducible calculation."""
    from mdprep.structure.pdb import read_pdb

    order = {
        (r.id.chain_id, r.id.resid): i
        for i, r in enumerate(read_pdb(STRUCTURE).residues)
    }
    _, full, _ = histidine_cluster(134)

    positions = [order[(r.id.chain_id, r.id.resid)] for r in full]
    assert positions == sorted(positions)
