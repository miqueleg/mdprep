"""SyrB2 (PDB 2FCT): a water on the iron, chlorides free and bound, gaps.

The deposition that found most of what this branch fixes. Measured against
the vendored file, so these need nothing but mdprep.
"""

from pathlib import Path

import pytest

from mdprep.config.decisions import build_manifest, plan_decisions, proposed_answers
from mdprep.config.loader import load_manifest
from mdprep.config.models import ManifestConfig
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb

DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
EXAMPLE = Path("examples/17_syrb2_bound_water_mcpb.yaml")


@pytest.fixture(scope="module")
def plan():
    return plan_decisions(DEPOSITION)


def test_the_water_on_the_iron_is_proposed_with_the_rest_of_the_shell(plan):
    donors = plan.by_id("metal.a_fe2_900_site.confirm_donors").evidence

    assert donors == (
        "A:AKG897 O2 at 2.03 A",
        "A:HIS116 NE2 at 2.18 A",
        "A:AKG897 O5 at 2.19 A",
        "A:HIS235 NE2 at 2.22 A",
        "A:HOH1072 O at 2.24 A",
        "A:CL901 CL at 2.47 A",
    )
    assert "A:HOH1072" in plan.findings["coordinating_waters"]


def test_free_chlorides_are_a_question_and_bound_ones_are_named(plan):
    """CL903 was refused as a reserved ligand name in 0.3.0."""

    decision = plan.by_id("structure.keep_crystal_ions")

    assert decision.evidence[:2] == ("A:CL903", "B:CL904")
    assert "A:CL901" in decision.evidence[2]
    assert not any(d.id.startswith("ligand.a_cl_") for d in plan.decisions)


def test_the_crystal_water_question_is_asked_despite_the_bound_water(plan):
    decision = plan.by_id("structure.keep_crystal_waters")

    assert any("A:HOH1072" in line for line in decision.evidence)


def test_the_answers_reproduce_the_example_manifest(plan):
    answers = {
        "structure.chains": "A",
        "structure.remove_unknown_heterogens": True,
        "structure.repair.add_missing_heavy_atoms": True,
        "ligand.a_dsu_896.include": False,
        "ligand.a_akg_897.net_charge": -2,
        "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
        "metal.a_fe2_900_site.confirm_donors": True,
        "refinement.total_qm_multiplicity": 5,
    }
    proposals = proposed_answers(plan, answers)
    # Fe +2, AKG -2, His 0, His 0, water 0, Cl -1.
    assert proposals["metal.a_fe2_900_site.model_charge"] == "-1"
    answers.update({key: int(value) for key, value in proposals.items()})

    manifest = ManifestConfig.model_validate(build_manifest(plan, answers))
    example = load_manifest(EXAMPLE)

    [site] = manifest.metals
    [example_site] = example.metals
    coordinators = {(bond.coordinator.resname, bond.coordinator.resid) for bond in site.mcpb.bonds}
    assert coordinators == {
        (bond.coordinator.resname, bond.coordinator.resid) for bond in example_site.mcpb.bonds
    }
    assert ("HOH", 1072) in coordinators
    assert manifest.structure.keep_crystal_ions and example.structure.keep_crystal_ions


def test_the_missing_loop_gets_a_ter_so_tleap_does_not_close_it(tmp_path):
    """Residues 57-60 of chain A are unresolved; 56 and 61 sit 6 A apart."""

    write_pdb(read_pdb(DEPOSITION), tmp_path / "out.pdb")

    lines = (tmp_path / "out.pdb").read_text(encoding="utf-8").splitlines()
    atom_or_ter = [line for line in lines if line.startswith(("ATOM", "HETATM", "TER"))]
    for index, line in enumerate(atom_or_ter):
        if line.startswith("ATOM") and line[21] == "A" and int(line[22:26]) == 61:
            assert atom_or_ter[index - 1].startswith("TER")
            assert int(atom_or_ter[index - 2][22:26]) == 56
            break
    else:
        pytest.fail("chain A residue 61 not found")
