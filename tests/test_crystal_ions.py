"""Free monatomic ions resolved in the crystal.

Halogenase depositions are full of chlorides: SyrB2 (PDB 2FCT) has one bound
to the iron and another free in the structure. From 0.3.0 the free one was
offered as a ligand, and then refused because CL is a name tleap's ion
library already builds -- telling the user to rename a chloride to LIG.
tleap builds a crystal Cl-, Na+ or Br- the way it builds a counterion, so
the only decision is whether it stays.
"""

from pathlib import Path

import pytest

from mdprep.config.decisions import DecisionError, build_manifest, plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.structure.classify import bulk_ion_charge, is_likely_ligand_or_cofactor
from mdprep.structure.pdb import read_pdb


DATA = Path("tests/data")

# Zn at (5.6, 5.4, 1.1) in protein_zinc_aqua.pdb.
BOUND_CHLORIDE = "HETATM 9101 CL    CL Z 501       5.600   5.400  -1.200  1.00 20.00          CL\n"
FREE_CHLORIDE = "HETATM 9102 CL    CL X 700      20.000  20.000  20.000  1.00 20.00          CL\n"
FREE_SODIUM = "HETATM 9103 NA    NA X 701     -15.000   0.000   0.000  1.00 20.00          NA\n"


def structure_with(tmp_path, *extra_lines: str) -> Path:
    lines = (DATA / "protein_zinc_aqua.pdb").read_text(encoding="utf-8").splitlines(True)
    body = [line for line in lines if not line.startswith("END")]
    path = tmp_path / "site.pdb"
    path.write_text("".join(body) + "".join(extra_lines) + "END\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


def test_a_free_chloride_is_not_a_ligand_candidate(tmp_path):
    structure = read_pdb(structure_with(tmp_path, FREE_CHLORIDE, FREE_SODIUM))
    by_name = {residue.id.resname: residue for residue in structure.residues}

    assert not is_likely_ligand_or_cofactor(by_name["CL"])
    assert not is_likely_ligand_or_cofactor(by_name["NA"])
    assert bulk_ion_charge(by_name["CL"]) == -1
    assert bulk_ion_charge(by_name["NA"]) == 1


def test_a_metal_is_not_a_bulk_ion(tmp_path):
    structure = read_pdb(structure_with(tmp_path))
    zinc = next(residue for residue in structure.residues if residue.id.resname == "ZN")

    with pytest.raises(ValueError, match="not a monatomic bulk ion"):
        bulk_ion_charge(zinc)


# --------------------------------------------------------------------------
# The question
# --------------------------------------------------------------------------


def test_free_ions_get_one_question_and_no_ligand_questions(tmp_path):
    plan = plan_decisions(structure_with(tmp_path, FREE_CHLORIDE, FREE_SODIUM))

    decision = plan.by_id("structure.keep_crystal_ions")
    assert decision.default == "true"
    assert decision.evidence[:2] == ("X:CL700", "X:NA701")
    assert not [d.id for d in plan.decisions if d.id.startswith("ligand.x_")]
    # Nor is either of them an "unknown heterogen" to delete or keep.
    assert not any(d.id == "structure.remove_unknown_heterogens" for d in plan.decisions)


def test_a_halide_bound_to_a_metal_is_named_but_not_asked_about(tmp_path):
    """It is the metal site's coordinator. Bonded, the site keeps it; with a
    nonbonded site it follows the answer, so the question has to say so
    rather than drop it unseen."""

    plan = plan_decisions(structure_with(tmp_path, BOUND_CHLORIDE, FREE_CHLORIDE))

    decision = plan.by_id("structure.keep_crystal_ions")
    assert decision.evidence[0] == "X:CL700"
    assert any("Z:CL501" in line and "bonded site keeps it" in line for line in decision.evidence)
    assert [target.resid for target in decision.targets] == [700]


def test_only_a_bound_halide_asks_nothing(tmp_path):
    plan = plan_decisions(structure_with(tmp_path, BOUND_CHLORIDE))

    assert not any(d.id == "structure.keep_crystal_ions" for d in plan.decisions)
    assert plan.findings["crystal_ions"] == ["Z:CL501"]


def test_the_answer_reaches_the_manifest(tmp_path):
    plan = plan_decisions(structure_with(tmp_path, FREE_CHLORIDE))
    answers = {
        "structure.keep_crystal_ions": False,
        "structure.repair.add_missing_heavy_atoms": False,
        "metal.z_zn_500_site.model": "nonbonded",
        "metal.z_zn_500_site.charge": 2,
    }

    manifest = build_manifest(plan, answers)

    assert manifest["structure"]["keep_crystal_ions"] is False
    ManifestConfig.model_validate(manifest)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalized(tmp_path, structure: Path, answers: dict):
    from mdprep.structure.normalize import normalize_structure_stage

    plan = plan_decisions(structure)
    data = build_manifest(plan, answers)
    data["project"]["output_dir"] = str(tmp_path / "out")
    manifest = ManifestConfig.model_validate(data)
    result = normalize_structure_stage(manifest, output_path=tmp_path / "normalized.pdb")
    kept = {
        (residue.id.chain_id, residue.id.resname, residue.id.resid)
        for residue in result.normalized_structure.residues
    }
    return result, kept


NONBONDED = {
    "structure.repair.add_missing_heavy_atoms": False,
    "metal.z_zn_500_site.model": "nonbonded",
    "metal.z_zn_500_site.charge": 2,
}


def test_kept_crystal_ions_are_written_and_reported(tmp_path):
    from mdprep.reports.structure_report import _render_markdown

    result, kept = normalized(
        tmp_path,
        structure_with(tmp_path, FREE_CHLORIDE, FREE_SODIUM),
        {**NONBONDED, "structure.keep_crystal_ions": True},
    )

    assert {("X", "CL", 700), ("X", "NA", 701)} <= kept
    report = result.to_report_dict()
    assert [ion["resname"] for ion in report["crystal_ions_kept"]] == ["CL", "NA"]
    assert report["crystal_ions_removed"] == []
    assert "## Crystal Ions" in _render_markdown(report)


def test_removed_crystal_ions_are_recorded_not_dropped_silently(tmp_path):
    result, kept = normalized(
        tmp_path,
        structure_with(tmp_path, FREE_CHLORIDE, FREE_SODIUM),
        {**NONBONDED, "structure.keep_crystal_ions": False},
    )

    assert ("X", "CL", 700) not in kept
    removed = result.to_report_dict()["crystal_ions_removed"]
    assert [ion["resname"] for ion in removed] == ["CL", "NA"]


def test_a_bonded_site_keeps_its_halide_when_the_others_go(tmp_path):
    answers = {
        "structure.keep_crystal_ions": False,
        "structure.repair.add_missing_heavy_atoms": False,
        "metal.z_zn_500_site.model": "bonded_mcpb",
        "metal.z_zn_500_site.charge": 2,
        "metal.z_zn_500_site.confirm_donors": True,
        "metal.z_zn_500_site.model_charge": 1,
        "refinement.total_qm_multiplicity": 1,
    }

    _, kept = normalized(tmp_path, structure_with(tmp_path, BOUND_CHLORIDE, FREE_CHLORIDE), answers)

    assert ("Z", "CL", 501) in kept
    assert ("X", "CL", 700) not in kept


# --------------------------------------------------------------------------
# Charge accounting
# --------------------------------------------------------------------------


def test_a_crystal_ion_in_an_xtb_cluster_counts_its_charge(tmp_path):
    """A histidine cluster that takes in a free chloride has one more
    electron than one that does not. It used to stop as an unknown
    heterogen."""

    from mdprep.protonation.histidine_xtb import cluster_charge_breakdown

    path = structure_with(tmp_path, FREE_CHLORIDE)
    data = build_manifest(plan_decisions(path), {**NONBONDED, "structure.keep_crystal_ions": True})
    manifest = ManifestConfig.model_validate(data)
    residues = list(read_pdb(path).residues)

    terms = cluster_charge_breakdown(residues, manifest, {})

    ion = next(term for term in terms if term["source"] == "crystal_ion")
    assert ion == {"residue": "X:CL700", "state": "CL", "charge": -1, "source": "crystal_ion"}


def test_declining_an_ion_question_is_not_a_reason_to_refuse(tmp_path):
    """Removing crystal ions and keeping unknown heterogens are separate
    answers; neither needs the other."""

    plan = plan_decisions(structure_with(tmp_path, FREE_CHLORIDE))
    try:
        build_manifest(plan, {**NONBONDED, "structure.keep_crystal_ions": False})
    except DecisionError as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"refused: {exc}")
