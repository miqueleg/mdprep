"""Placing ligand hydrogens from a user-supplied SMILES.

A crystal structure carries none, and mdprep will not guess a tautomer, so a
deposited ligand cannot be parameterized at all without this. The SMILES is the
user stating the protonation they intend; mdprep only places the atoms.

Two rules are load-bearing and are tested from several directions:
a ligand that already has hydrogens is never touched, and adding any always
warns, because geometric placement of a wrong tautomer looks perfectly
reasonable.
"""

from pathlib import Path

import pytest

from mdprep.ligands.hydrogenate import (
    LigandHydrogenationError,
    add_ligand_hydrogens,
    count_hydrogens,
    hydrogen_names_for,
    is_already_protonated,
)
from mdprep.structure.models import AtomRecord, ResidueRecord
from mdprep.structure.pdb import read_pdb


rdkit = pytest.importorskip("rdkit", reason="RDKit is required to place hydrogens")

DATA = Path("tests/data")


def residue_named(path: Path, resname: str) -> ResidueRecord:
    return next(r for r in read_pdb(path).residues if r.id.resname == resname)


def atom(name: str, element: str, index: int) -> AtomRecord:
    return AtomRecord(
        serial=index,
        name=name,
        altloc=None,
        resname="LIG",
        chain_id="A",
        resid=1,
        icode=None,
        x=float(index),
        y=0.0,
        z=0.0,
        occupancy=1.0,
        bfactor=0.0,
        element=element,
        record_name="HETATM",
        original_line="",
    )


def residue_of(*atoms: AtomRecord) -> ResidueRecord:
    from mdprep.structure.models import ResidueId

    return ResidueRecord(
        id=ResidueId(chain_id="A", resname="LIG", resid=1, icode=None),
        atoms=list(atoms),
        record_names={"HETATM"},
        original_index=0,
    )


# --------------------------------------------------------------------------
# Never touch a ligand that is already protonated
# --------------------------------------------------------------------------



def _heavy_only_akg():
    """The alpha-ketoglutarate of 6NIE with its deposited hydrogens removed."""

    from dataclasses import replace

    from mdprep.structure.pdb import read_pdb

    structure = read_pdb("examples/tutorials/besd_6nie/6NIE.pdb")
    residue = next(
        item
        for item in structure.residues
        if item.id.resname == "AKG" and item.id.chain_id == "A"
    )
    return replace(
        residue,
        atoms=[
            atom
            for atom in residue.atoms
            if (atom.element or "").strip().upper() != "H"
        ],
    )


def test_a_protonated_ligand_is_left_exactly_as_it_was():
    before = residue_of(atom("C1", "C", 1), atom("H1", "H", 2))

    result = add_ligand_hydrogens(before, "C", ligand_id="lig")

    assert result.skipped is True
    assert result.added_hydrogens == 0
    assert result.residue is before
    assert [a.name for a in result.residue.atoms] == ["C1", "H1"]


def test_skipping_is_reported_rather_than_silent():
    before = residue_of(atom("C1", "C", 1), atom("H1", "H", 2))

    result = add_ligand_hydrogens(before, "C", ligand_id="lig")

    assert result.warnings
    assert "already carries" in result.warnings[0]


def test_a_single_hydrogen_is_enough_to_count_as_protonated():
    """Partial protonation is a different problem, caught by the formula check.

    Quietly finishing it off would be exactly the guess mdprep refuses to make.
    """
    partial = residue_of(
        atom("C1", "C", 1), atom("C2", "C", 2), atom("H1", "H", 3)
    )

    assert is_already_protonated(partial)
    assert add_ligand_hydrogens(partial, "CC", ligand_id="lig").skipped


def test_hydrogens_are_recognised_by_element_and_by_name():
    by_element = residue_of(atom("X1", "H", 1))
    unnamed_element = residue_of(
        AtomRecord(
            serial=1, name="HG1", altloc=None, resname="LIG", chain_id="A",
            resid=1, icode=None, x=0.0, y=0.0, z=0.0, occupancy=1.0,
            bfactor=0.0, element=None, record_name="HETATM", original_line="",
        )
    )

    assert count_hydrogens(by_element) == 1
    assert count_hydrogens(unnamed_element) == 1


# --------------------------------------------------------------------------
# Adding hydrogens
# --------------------------------------------------------------------------


SIX_NIE = Path("examples/tutorials/besd_6nie/6NIE.pdb")


@pytest.mark.skipif(not SIX_NIE.exists(), reason="6NIE is not available locally")
def test_hydrogens_are_placed_on_a_real_deposited_ligand():
    """The BesD lysine substrate, straight from the deposition."""
    structure = read_pdb(SIX_NIE)
    lysine = next(
        r
        for r in structure.residues
        if r.id.chain_id == "A" and r.id.resname == "LYS" and r.id.resid == 301
    )
    assert not is_already_protonated(lysine)

    result = add_ligand_hydrogens(
        lysine, "C(CCN)CC(C(=O)O)N", ligand_id="a_lys_301"
    )

    assert result.skipped is False
    assert result.added_hydrogens == 14
    # Names follow the heavy atoms they hang from, which is what makes the
    # result readable in a viewer.
    names = {a.name for a in result.residue.atoms if a.element == "H"}
    assert {"HA", "HB", "HG", "HD", "HE", "HZ"} <= names


def test_adding_hydrogens_always_warns():
    """Geometric placement of the wrong tautomer looks entirely reasonable."""
    methane_core = residue_of(atom("C1", "C", 1))

    result = add_ligand_hydrogens(methane_core, "C", ligand_id="lig")

    assert result.added_hydrogens == 4
    assert result.warnings
    warning = result.warnings[0]
    assert "LIGAND_HYDROGENS_ADDED" in warning
    assert "Inspect this ligand" in warning


def test_heavy_atoms_keep_their_names_order_and_coordinates():
    before = residue_of(atom("C1", "C", 1), atom("C2", "C", 2))

    result = add_ligand_hydrogens(before, "CC", ligand_id="lig")

    heavy = [a for a in result.residue.atoms if a.element != "H"]
    assert [a.name for a in heavy] == ["C1", "C2"]
    for original, produced in zip(before.atoms, heavy):
        assert (produced.x, produced.y, produced.z) == (
            original.x,
            original.y,
            original.z,
        )


def test_hydrogens_are_appended_after_the_heavy_atoms():
    before = residue_of(atom("C1", "C", 1), atom("C2", "C", 2))

    atoms = add_ligand_hydrogens(before, "CC", ligand_id="lig").residue.atoms

    elements = [a.element for a in atoms]
    assert elements[:2] == ["C", "C"]
    assert set(elements[2:]) == {"H"}


def test_new_hydrogen_names_are_unique_and_pdb_sized():
    before = residue_of(atom("C1", "C", 1), atom("C2", "C", 2))

    atoms = add_ligand_hydrogens(before, "CC", ligand_id="lig").residue.atoms

    names = [a.name for a in atoms]
    assert len(names) == len(set(names))
    assert all(len(name) <= 4 for name in names)


def test_the_smiles_determines_the_charge_and_therefore_the_hydrogens():
    """A charged SMILES must give a different count from its neutral form."""
    core = residue_of(atom("N1", "N", 1))

    neutral = add_ligand_hydrogens(core, "N", ligand_id="lig").added_hydrogens
    protonated = add_ligand_hydrogens(core, "[NH4+]", ligand_id="lig").added_hydrogens

    assert neutral == 3
    assert protonated == 4


# --------------------------------------------------------------------------
# Failures are explicit
# --------------------------------------------------------------------------


def test_a_smiles_with_the_wrong_atom_count_is_rejected():
    before = residue_of(atom("C1", "C", 1), atom("C2", "C", 2))

    with pytest.raises(LigandHydrogenationError, match="heavy atom"):
        add_ligand_hydrogens(before, "C", ligand_id="lig")


def test_an_invalid_smiles_is_rejected():
    before = residue_of(atom("C1", "C", 1))

    with pytest.raises(LigandHydrogenationError, match="not a valid SMILES"):
        add_ligand_hydrogens(before, "this is not a smiles", ligand_id="lig")


def test_a_smiles_implying_no_hydrogens_is_rejected():
    """Silently doing nothing would leave the ligand unusable downstream."""
    before = residue_of(atom("C1", "C", 1), atom("C2", "C", 2))

    with pytest.raises(LigandHydrogenationError, match="no hydrogens"):
        add_ligand_hydrogens(before, "[C-]#[C-]", ligand_id="lig")


def test_the_error_names_the_ligand():
    before = residue_of(atom("C1", "C", 1))

    with pytest.raises(LigandHydrogenationError, match="my_cofactor"):
        add_ligand_hydrogens(before, "CCCC", ligand_id="my_cofactor")


# --------------------------------------------------------------------------
# Naming helper
# --------------------------------------------------------------------------


def test_hydrogen_names_follow_their_parent():
    parent = atom("CB", "C", 1)

    names = hydrogen_names_for(parent, 2, set())

    assert names == ["HB", "HBA"]


def test_hydrogen_names_avoid_names_already_in_use():
    parent = atom("CB", "C", 1)

    names = hydrogen_names_for(parent, 1, {"HB"})

    assert names == ["HBA"]


def test_a_smiles_whose_charge_contradicts_the_manifest_is_refused():
    """The SMILES sets the protonation and net_charge sets the electrostatics,
    so the two describing different molecules is not something to warn about.

    A SMILES one proton out still matches the same heavy atoms and the same
    connectivity, so nothing else in the chain would notice.
    """

    residue = _heavy_only_akg()

    with pytest.raises(LigandHydrogenationError, match="formal charge of -1"):
        add_ligand_hydrogens(
            residue,
            "OC(=O)C(=O)CCC(=O)[O-]",  # the monoanion
            ligand_id="akg",
            net_charge=-2,
        )


def test_the_declared_charge_and_the_smiles_agreeing_is_accepted():
    residue = _heavy_only_akg()

    result = add_ligand_hydrogens(
        residue,
        "[O-]C(=O)C(=O)CCC(=O)[O-]",
        ligand_id="akg",
        net_charge=-2,
    )

    assert result.added_hydrogens == 4


def test_a_tautomer_cannot_be_checked_and_is_taken_on_trust():
    """The keto and enol forms of alpha-ketoglutarate are both C5H4O5(2-) over
    the same heavy-atom connectivity, and differ only in where two hydrogens
    sit and in two bond orders.

    Nothing mdprep can compute distinguishes them, which is precisely why the
    SMILES is asked for rather than inferred. This pins that the enol is
    accepted and places its hydrogens differently, so the limit is recorded
    rather than assumed away.
    """

    keto = add_ligand_hydrogens(
        _heavy_only_akg(),
        "[O-]C(=O)C(=O)CCC(=O)[O-]",
        ligand_id="akg",
        net_charge=-2,
    )
    enol = add_ligand_hydrogens(
        _heavy_only_akg(),
        "[O-]C(=O)C(O)=CCC(=O)[O-]",
        ligand_id="akg",
        net_charge=-2,
    )

    def hydrogen_parents(result):
        names = [
            atom.name.strip()
            for atom in result.residue.atoms
            if (atom.element or "").strip().upper() == "H"
        ]
        return sorted(names)

    assert keto.added_hydrogens == enol.added_hydrogens == 4
    # Same count, same charge, different atoms carrying them.
    assert hydrogen_parents(keto) != hydrogen_parents(enol)
