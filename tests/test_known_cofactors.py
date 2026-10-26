"""Organic cofactors whose chemistry their names fix: flavins, SAM, NAD(P).

Flavin-dependent halogenases hold FAD, an iodotyrosine deiodinase FMN, a
fluorinase SAM. Typing a correct FAD SMILES in the middle of a form is not
something to ask of anyone, and the residue name settles everything but the
protonation, which at pH 7 is stated with the reason. Each SMILES is checked
here against the cofactor as a real deposition holds it.
"""

from pathlib import Path

import pytest

from mdprep.config.decisions import plan_decisions, proposal_explanations, proposed_answers
from mdprep.ligands.cofactors import KNOWN_ORGANIC_COFACTORS, known_organic_cofactor
from mdprep.structure.models import ResidueRecord
from mdprep.structure.pdb import read_pdb

DATA = Path(__file__).parent / "data" / "cofactors"

# One deposition each, heavy atoms only, as a crystal resolves them.
FIXTURES = {
    "FAD": "fad_2ar8.pdb",
    "FMN": "fmn_4ttb.pdb",
    "SAM": "sam_7rwg.pdb",
    "SAH": "sah_8pb4.pdb",
    "NAD": "nad_2jhf.pdb",
    "NAI": "nai_4plz.pdb",
    "NAP": "nap_4lbs.pdb",
    "NDP": "ndp_1kmv.pdb",
}


def _residue(name: str) -> ResidueRecord:
    (residue,) = read_pdb(DATA / FIXTURES[name]).residues
    return residue


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_a_deposited_cofactor_is_recognised_by_name_and_atoms(name):
    assert known_organic_cofactor(_residue(name)) is KNOWN_ORGANIC_COFACTORS[name]


def test_a_cofactor_missing_atoms_to_disorder_gets_nothing():
    residue = _residue("FAD")
    partial = ResidueRecord(
        id=residue.id,
        atoms=residue.atoms[:-3],
        record_names=residue.record_names,
        original_index=residue.original_index,
    )

    assert known_organic_cofactor(partial) is None


@pytest.mark.parametrize(
    "name, charge",
    [("FAD", -2), ("FMN", -2), ("SAM", 1), ("SAH", 0), ("NAD", -1), ("NAI", -2), ("NAP", -3), ("NDP", -4)],
)
def test_the_charge_is_the_one_at_ph_7(name, charge):
    assert KNOWN_ORGANIC_COFACTORS[name].net_charge == charge


def test_the_planner_proposes_the_smiles_and_the_charge(tmp_path):
    plan = plan_decisions(DATA / "fad_2ar8.pdb")
    (ligand_id,) = plan.findings["known_organic_cofactors"]
    answers = {f"ligand.{ligand_id}.include": "true"}

    proposals = proposed_answers(plan, answers)
    why = proposal_explanations(plan, answers)

    assert proposals[f"ligand.{ligand_id}.smiles"] == KNOWN_ORGANIC_COFACTORS["FAD"].smiles
    assert proposals[f"ligand.{ligand_id}.net_charge"] == "-2"
    assert "diphosphate fully ionised" in why[f"ligand.{ligand_id}.smiles"]
    # Proposed, not defaulted: the question still has no default of its own.
    assert plan.by_id(f"ligand.{ligand_id}.net_charge").default is None


def test_a_cofactor_that_came_with_hydrogens_has_its_smiles_proposed_all_the_same(tmp_path):
    """The input's hydrogens are replaced by the SMILES's, so it is asked for and proposed."""

    lines = (DATA / "sam_7rwg.pdb").read_text().splitlines()
    atom = next(line for line in lines if line.startswith(("ATOM", "HETATM")))
    hydrogen = atom[:12] + " H1 " + atom[16:30] + "  99.000  99.000  99.000" + atom[54:76] + " H"
    path = tmp_path / "sam.pdb"
    path.write_text("\n".join(lines[:-1] + [hydrogen, "END"]) + "\n", encoding="utf-8")

    plan = plan_decisions(path)
    proposals = proposed_answers(plan, {})

    (ligand_id,) = plan.findings["known_organic_cofactors"]
    assert proposals[f"ligand.{ligand_id}.smiles"] == KNOWN_ORGANIC_COFACTORS["SAM"].smiles


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_the_smiles_places_hydrogens_on_the_deposited_cofactor(name):
    """The proposal is only worth making if mdprep then accepts it.

    Each SMILES is matched onto the deposited heavy atoms and its bond orders
    checked against their lengths, which an amide, an amino-purine and a
    flavin used to fail: their conjugated single bonds were measured against
    saturated ones.
    """

    pytest.importorskip("rdkit")
    from mdprep.ligands.hydrogenate import add_ligand_hydrogens

    entry = KNOWN_ORGANIC_COFACTORS[name]

    result = add_ligand_hydrogens(
        _residue(name), entry.smiles, ligand_id=name.lower(), net_charge=entry.net_charge
    )

    assert result.added_hydrogens > 0
