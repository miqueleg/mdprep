"""Tests for detecting what a deposited structure is missing.

`tleap` cannot build a residue whose heavy atoms are absent, so a run that
ignores them fails late and expensively. Detection has to be reliable enough to
ask about up front.
"""

from pathlib import Path

import pytest

from mdprep.config.decisions import (
    coordinating_water_residues,
    metal_donor_atoms,
    plan_decisions,
)
from mdprep.structure.completeness import (
    chain_breaks,
    expected_heavy_atoms,
    incomplete_residues,
    missing_heavy_atoms,
)
from mdprep.structure.inspect import inspect_pdb_structure
from mdprep.structure.pdb import read_pdb


DATA = Path("tests/data")
SIX_NIE = Path("examples/tutorials/besd_6nie/6NIE.pdb")


def test_protonation_variants_share_their_parent_heavy_atoms():
    """HID/HIE/HIP differ only in hydrogens."""
    histidine = expected_heavy_atoms("HIS")

    assert expected_heavy_atoms("HID") == histidine
    assert expected_heavy_atoms("HIE") == histidine
    assert expected_heavy_atoms("HIP") == histidine
    assert expected_heavy_atoms("CYX") == expected_heavy_atoms("CYS")
    assert expected_heavy_atoms("ASH") == expected_heavy_atoms("ASP")


def test_glycine_has_no_side_chain_atoms():
    assert expected_heavy_atoms("GLY") == frozenset({"N", "CA", "C", "O"})


def test_a_non_standard_residue_is_not_assessed():
    assert expected_heavy_atoms("AKG") is None


def test_hydrogens_do_not_count_toward_completeness():
    """A structure with hydrogens must not look different from one without."""
    with_h = read_pdb(DATA / "protein_with_hydrogens.pdb")

    for residue in with_h.residues:
        missing = missing_heavy_atoms(residue)
        assert all(not name.startswith("H") for name in missing)


def test_a_complete_residue_reports_nothing_missing():
    structure = read_pdb(DATA / "protein_histidine_ring.pdb")
    histidines = [r for r in structure.residues if r.id.resname.startswith("HI")]

    assert histidines
    for residue in histidines:
        assert "CG" not in missing_heavy_atoms(residue)


@pytest.mark.skipif(not SIX_NIE.exists(), reason="6NIE is not available locally")
def test_detection_matches_what_the_depositors_recorded():
    """REMARK 470 lists the residues with missing atoms; we must agree."""
    import re

    structure = read_pdb(SIX_NIE)
    detected = incomplete_residues(structure.residues)

    recorded = [
        line
        for line in SIX_NIE.read_text().splitlines()
        if re.match(r"^REMARK 470\s+[A-Z]{3} [A-Z]\s+\d", line)
    ]
    assert len(detected) == len(recorded) == 33


@pytest.mark.skipif(not SIX_NIE.exists(), reason="6NIE is not available locally")
def test_a_structure_without_internal_gaps_reports_no_chain_breaks():
    """6NIE is missing only terminal expression-tag residues."""
    structure = read_pdb(SIX_NIE)

    assert chain_breaks(structure.residues) == []


def _alanine(serial: int, resid: int, x: float) -> list[str]:
    """Four backbone atoms plus CB, which is a complete alanine."""
    atoms = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]
    return [
        f"ATOM  {serial + i:5d} {name:^4} ALA A{resid:4d}    "
        f"{x + i * 0.1:8.3f}{0.0:8.3f}{0.0:8.3f}{1.00:6.2f}{0.00:6.2f}"
        f"          {element:>2}"
        for i, (name, element) in enumerate(atoms)
    ]


def test_a_numbering_gap_is_reported_as_a_chain_break(tmp_path):
    broken = tmp_path / "gap.pdb"
    # Residues 1 and 9 of one chain: residues 2-8 were not resolved.
    broken.write_text(
        "\n".join(_alanine(1, 1, 0.0) + _alanine(6, 9, 20.0) + ["END"]) + "\n",
        encoding="utf-8",
    )

    breaks = chain_breaks(read_pdb(broken).residues)

    assert len(breaks) == 1
    assert "7 residue(s) absent by numbering" in breaks[0].reason


def test_consecutive_residues_that_are_far_apart_are_a_chain_break(tmp_path):
    """Numbering can be continuous across a gap the coordinates still show."""
    stretched = tmp_path / "stretched.pdb"
    stretched.write_text(
        "\n".join(_alanine(1, 1, 0.0) + _alanine(6, 2, 30.0) + ["END"]) + "\n",
        encoding="utf-8",
    )

    breaks = chain_breaks(read_pdb(stretched).residues)

    assert len(breaks) == 1
    assert "C-N distance" in breaks[0].reason


def test_a_continuous_chain_reports_no_break(tmp_path):
    intact = tmp_path / "intact.pdb"
    # Residue 2's N placed 1.33 A from residue 1's C.
    lines = _alanine(1, 1, 0.0)
    second = _alanine(6, 2, 1.53)
    intact.write_text("\n".join(lines + second + ["END"]) + "\n", encoding="utf-8")

    assert chain_breaks(read_pdb(intact).residues) == []


def test_incomplete_residues_prompt_a_blocking_repair_question():
    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    decision = plan.by_id("structure.repair.add_missing_heavy_atoms")

    assert decision.default is None
    assert decision.requires_user_input is True
    assert decision.evidence


def test_answering_yes_configures_pdbfixer():
    from mdprep.config.decisions import build_manifest

    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    manifest = build_manifest(
        plan,
        {
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": True,
            "ligand.b_sub_501.net_charge": 0,
            "ligand.b_sub_501.smiles": "C",
            "ligand.c_cof_601.net_charge": 0,
            "ligand.c_cof_601.smiles": "C",
        },
    )

    assert manifest["structure"]["repair"]["backend"] == "pdbfixer"
    assert manifest["structure"]["repair"]["add_missing_heavy_atoms"] is True


def test_answering_no_leaves_repair_off():
    from mdprep.config.decisions import build_manifest

    plan = plan_decisions(DATA / "protein_two_ligands.pdb")

    manifest = build_manifest(
        plan,
        {
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": False,
            "ligand.b_sub_501.net_charge": 0,
            "ligand.b_sub_501.smiles": "C",
            "ligand.c_cof_601.net_charge": 0,
            "ligand.c_cof_601.smiles": "C",
        },
    )

    assert "repair" not in manifest["structure"]


# --------------------------------------------------------------------------
# Coordinating waters
# --------------------------------------------------------------------------


def test_a_first_shell_water_is_detected_as_a_donor():
    summary = inspect_pdb_structure(DATA / "protein_zinc_aqua.pdb")
    zinc = summary.metal_ions[0]

    donors = metal_donor_atoms(summary.structure.residues, zinc)

    waters = [res.id.display() for res, _, _ in donors if res.id.resname == "HOH"]
    assert waters == ["W:HOH601"]


def test_a_bulk_water_is_not_a_donor():
    """The second water sits at 4.5 A and is solvent, not part of the site."""
    summary = inspect_pdb_structure(DATA / "protein_zinc_aqua.pdb")

    coordinating = {r.id.display() for r in coordinating_water_residues(summary)}

    assert "W:HOH602" not in coordinating


def test_a_coordinating_water_does_not_hide_the_question_about_the_others():
    """The question used to vanish as soon as any water bound any metal. In
    6NIE that was a surface iron in chain B with three waters on it, and it
    hid the decision for all 847 waters -- even after chain A was chosen.
    Now it is asked, and says the bound water stays whatever the answer."""
    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")

    assert plan.findings["coordinating_waters"] == ["W:HOH601"]
    decision = plan.by_id("structure.keep_crystal_waters")
    assert any("W:HOH601" in line and "kept with it whatever" in line for line in decision.evidence)


def test_a_water_bound_to_a_bonded_metal_survives_dropping_the_others(tmp_path):
    """Discarding it would leave an MCPB bond pointing at a residue that is
    gone, which only surfaced at the MCPB step, after protonation and the
    ligand parameters."""
    from mdprep.config.decisions import build_manifest
    from mdprep.config.models import ManifestConfig
    from mdprep.structure.normalize import normalize_structure_stage

    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")
    data = build_manifest(
        plan,
        {
            "structure.keep_crystal_waters": True,
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": False,
            "metal.z_zn_500_site.model": "bonded_mcpb",
            "metal.z_zn_500_site.charge": 2,
            "metal.z_zn_500_site.confirm_donors": True,
            "metal.z_zn_500_site.multiplicity": 1,
            "metal.z_zn_500_site.model_charge": 2,
            "refinement.total_qm_multiplicity": 1,
        },
    )
    data["project"]["output_dir"] = str(tmp_path / "out")
    manifest = ManifestConfig.model_validate(data)
    # The builder will not write this combination with QM/MM refinement on
    # (next test), but a hand-written manifest using external_pdb geometry
    # can, and normalisation has to protect the bound water on its own.
    manifest = manifest.model_copy(
        update={"structure": manifest.structure.model_copy(update={"keep_crystal_waters": False})}
    )

    result = normalize_structure_stage(manifest, output_path=tmp_path / "n.pdb")

    kept = {(r.id.chain_id, r.id.resname, r.id.resid) for r in result.normalized_structure.residues}
    assert ("W", "HOH", 601) in kept
    # And only that one: the free waters went, as asked.
    assert sum(1 for key in kept if key[1] == "HOH") == 1


def test_waters_remain_optional_when_none_coordinate():
    plan = plan_decisions(DATA / "protein_with_waters.pdb")

    assert plan.findings["coordinating_waters"] == []
    assert any(d.id == "structure.keep_crystal_waters" for d in plan.decisions)


def test_a_coordinating_water_becomes_a_bond_in_the_generated_site():
    from mdprep.config.decisions import build_manifest
    from mdprep.config.models import ManifestConfig

    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")

    manifest = build_manifest(
        plan,
        {
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": True,
            "metal.z_zn_500_site.model": "bonded_mcpb",
            "metal.z_zn_500_site.charge": 2,
            "metal.z_zn_500_site.confirm_donors": True,
            "metal.z_zn_500_site.multiplicity": 1,
            "refinement.total_qm_multiplicity": 1,
            "metal.z_zn_500_site.model_charge": 2,
        },
    )

    ManifestConfig.model_validate(manifest)
    coordinators = {
        b["coordinator"]["resname"] for b in manifest["metals"][0]["mcpb"]["bonds"]
    }
    assert "HOH" in coordinators
    assert manifest["structure"]["keep_crystal_waters"] is True


def test_dropping_waters_with_a_bonded_site_is_caught_in_the_builder():
    """The QM/MM refinement a bonded site runs needs the active-site waters,
    and the schema refuses the combination -- but as a pydantic error at
    Finish. Say what to do instead."""
    import pytest

    from mdprep.config.decisions import DecisionError, build_manifest

    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")
    with pytest.raises(DecisionError, match="crystal waters have to be kept"):
        build_manifest(
            plan,
            {
                "structure.keep_crystal_waters": False,
                "structure.remove_unknown_heterogens": False,
                "structure.repair.add_missing_heavy_atoms": False,
                "metal.z_zn_500_site.model": "bonded_mcpb",
                "metal.z_zn_500_site.charge": 2,
                "metal.z_zn_500_site.confirm_donors": True,
                "metal.z_zn_500_site.multiplicity": 1,
                "metal.z_zn_500_site.model_charge": 2,
                "refinement.total_qm_multiplicity": 1,
            },
        )


def test_a_water_bound_to_a_nonbonded_metal_survives_dropping_the_others(tmp_path):
    """The crystal-water question promises a water on a metal stays with it
    whatever the answer. That held only for a bonded site; a water on a
    nonbonded zinc went with the others."""
    from mdprep.config.decisions import build_manifest
    from mdprep.config.models import ManifestConfig
    from mdprep.structure.normalize import normalize_structure_stage

    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")
    data = build_manifest(
        plan,
        {
            "structure.keep_crystal_waters": False,
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": False,
            "metal.z_zn_500_site.model": "nonbonded",
            "metal.z_zn_500_site.charge": 2,
        },
    )
    data["project"]["output_dir"] = str(tmp_path / "out")

    result = normalize_structure_stage(
        ManifestConfig.model_validate(data), output_path=tmp_path / "n.pdb"
    )

    report = result.to_report_dict()
    assert [(w["chain_id"], w["resid"]) for w in report["waters_kept"]] == [("W", 601)]
    assert [(w["chain_id"], w["resid"]) for w in report["waters_removed"]] == [("W", 602)]


def test_the_question_and_the_build_share_one_definition_of_bound():
    """Two copies of "within bonding distance" would let the question promise
    one set of waters and the build keep another."""
    from mdprep.config import decisions
    from mdprep.structure import coordination

    assert decisions.metal_donor_atoms is coordination.metal_donor_atoms
    assert decisions.COORDINATION_CUTOFF_ANGSTROM == coordination.COORDINATION_CUTOFF_ANGSTROM
