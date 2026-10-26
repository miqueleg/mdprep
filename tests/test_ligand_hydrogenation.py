"""Placing ligand hydrogens from a user-supplied SMILES.

A crystal structure carries none, and mdprep will not guess a tautomer, so a
deposited ligand cannot be parameterized at all without this. The SMILES is the
user stating the protonation they intend; mdprep only places the atoms.

Two rules are load-bearing and are tested from several directions:
a ligand that keeps the hydrogens it already has is never touched -- only
checked, atom by atom, against the SMILES -- and adding any always warns,
because geometric placement of a wrong tautomer looks perfectly reasonable.
"""

from pathlib import Path

import pytest

from mdprep.ligands.hydrogenate import (
    LigandHydrogenationError,
    add_ligand_hydrogens,
    count_hydrogens,
    hydrogen_names_for,
)
from mdprep.structure.models import AtomRecord, ResidueRecord
from mdprep.structure.pdb import read_pdb


rdkit = pytest.importorskip("rdkit", reason="RDKit is required to place hydrogens")

DATA = Path("tests/data")


def residue_named(path: Path, resname: str) -> ResidueRecord:
    return next(r for r in read_pdb(path).residues if r.id.resname == resname)


# A plausible single-bond separation. The hydrogenation now checks the bond
# orders a SMILES implies against the distances in the structure, so a fixture
# spacing its atoms 1 A apart would be read as a triple bond.
SINGLE_BOND_ANGSTROM = 1.52


def atom(name: str, element: str, index: int) -> AtomRecord:
    return AtomRecord(
        serial=index,
        name=name,
        altloc=None,
        resname="LIG",
        chain_id="A",
        resid=1,
        icode=None,
        x=float(index) * SINGLE_BOND_ANGSTROM,
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
# A ligand that arrives with hydrogens: kept if they are the SMILES's
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


def methane(hydrogens: int = 4) -> ResidueRecord:
    """A carbon with ``hydrogens`` of its four tetrahedral hydrogens."""

    from dataclasses import replace

    corners = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    carbon = replace(atom("C1", "C", 0), x=0.0)
    placed = [
        replace(
            atom(f"H{index + 1}", "H", index + 1),
            x=0.629 * cx,
            y=0.629 * cy,
            z=0.629 * cz,
        )
        for index, (cx, cy, cz) in enumerate(corners[:hydrogens])
    ]
    return residue_of(carbon, *placed)


def test_a_protonated_ligand_is_left_exactly_as_it_was():
    before = methane()

    result = add_ligand_hydrogens(before, "C", ligand_id="lig")

    assert result.skipped is True
    assert result.added_hydrogens == 0
    assert result.residue is before
    assert [a.name for a in result.residue.atoms] == ["C1", "H1", "H2", "H3", "H4"]


def test_keeping_them_is_reported_rather_than_silent():
    result = add_ligand_hydrogens(methane(), "C", ligand_id="lig")

    assert result.warnings
    assert "kept its 4 input hydrogen(s)" in result.warnings[0]


def test_hydrogens_that_are_not_the_smiles_are_replaced_and_reported():
    """The SMILES states the protonation; hydrogens saying otherwise are a different molecule.

    They used to be kept, which is how a docked substrate with only its
    polar hydrogens reached antechamber with an odd number of electrons.
    """
    result = add_ligand_hydrogens(methane(3), "C", ligand_id="lig")

    assert result.replaced_hydrogens == 3
    assert result.added_hydrogens == 4
    assert result.differences == ("C1 has 3 in the input and 4 in the SMILES",)
    assert [a.name for a in result.residue.atoms if a.element == "H"] == ["H1", "H1A", "H1B", "H1C"]
    assert "LIGAND_HYDROGENS_REPLACED" in result.warnings[0]


def test_keeping_them_on_purpose_makes_a_disagreement_an_error():
    with pytest.raises(LigandHydrogenationError, match="C1 has 3 in the input and 4 in the SMILES"):
        add_ligand_hydrogens(methane(3), "C", ligand_id="lig", keep_input_hydrogens=True)


def test_a_hydrogen_bonded_to_nothing_is_a_disagreement():
    stray = residue_of(atom("C1", "C", 0), atom("H1", "H", 2))

    result = add_ligand_hydrogens(stray, "C", ligand_id="lig")

    assert any("bonded to none of its atoms" in item for item in result.differences)
    assert count_hydrogens(result.residue) == 4
    with pytest.raises(LigandHydrogenationError, match="bonded to none of its atoms"):
        add_ligand_hydrogens(stray, "C", ligand_id="lig", keep_input_hydrogens=True)


def test_a_partially_hydrogenated_ligand_gets_all_of_the_smiles():
    """One hydrogen does not make a ligand protonated: the SMILES is still applied."""
    partial = residue_of(
        atom("C1", "C", 1), atom("C2", "C", 2), atom("H1", "H", 3)
    )

    result = add_ligand_hydrogens(partial, "CC", ligand_id="lig")

    assert result.replaced_hydrogens == 1
    assert count_hydrogens(result.residue) == 6
    assert [a.name for a in result.residue.atoms][:2] == ["C1", "C2"]


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
    assert count_hydrogens(lysine) == 0

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


def test_a_smiles_implying_no_hydrogens_places_none():
    """A hydrogen-free molecule is stated by its SMILES like any other.

    Every ligand with hydrogens now needs a SMILES, so one without any has to
    be expressible as one too.
    """
    from dataclasses import replace

    # Spaced as a triple bond, so the geometry agrees with the SMILES.
    first, second = atom("C1", "C", 1), atom("C2", "C", 2)
    before = residue_of(first, replace(second, x=first.x + 1.20))

    result = add_ligand_hydrogens(before, "[C-]#[C-]", ligand_id="lig")

    assert result.added_hydrogens == 0
    assert [a.name for a in result.residue.atoms] == ["C1", "C2"]
    assert "gives it none" in result.warnings[0]


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


def test_a_wrong_tautomer_is_caught_by_the_structure_geometry():
    """Formula, charge and connectivity cannot tell two tautomers apart, but
    the coordinates can: a ketone C=O refines near 1.22 A and an enol C-OH
    near 1.36.

    On 6NIE's alpha-ketoglutarate, C2-O5 is 1.207 A and C2-C3 is 1.500 A --
    the keto form. The enol SMILES shares its formula and its charge, so it
    passes every other check, and is refused on the geometry.
    """

    keto = add_ligand_hydrogens(
        _heavy_only_akg(),
        "[O-]C(=O)C(=O)CCC(=O)[O-]",
        ligand_id="akg",
        net_charge=-2,
    )
    assert keto.added_hydrogens == 4

    with pytest.raises(LigandHydrogenationError, match="do not match the geometry"):
        add_ligand_hydrogens(
            _heavy_only_akg(),
            "[O-]C(=O)C(O)=CCC(=O)[O-]",
            ligand_id="akg",
            net_charge=-2,
        )


def test_a_carboxylate_is_not_mistaken_for_a_contradiction():
    """A deprotonated carboxylate refines with two equal C-O bonds near
    1.25 A, which any SMILES has to write as one single and one double. The
    formal orders are notation there, not a claim about length."""

    from mdprep.ligands.hydrogenate import geometry_contradicts_bond_orders
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from mdprep.ligands.hydrogenate import _pdb_block

    residue = _heavy_only_akg()
    block = _pdb_block(list(residue.atoms))
    molecule = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    template = Chem.MolFromSmiles("[O-]C(=O)C(=O)CCC(=O)[O-]")
    molecule = AllChem.AssignBondOrdersFromTemplate(template, molecule)

    # Both carboxylates are present and neither is reported.
    assert geometry_contradicts_bond_orders(molecule, residue) == []


def test_a_ligand_from_a_prediction_with_a_wide_name_can_be_hydrogenated():
    """The failure this guards against was reported from a real run.

    Boltz named the ligand LIG2 and its chain AKG, neither of which fits the
    column a PDB record gives it. mdprep wrote the name out in full, every
    later field moved one column right, and RDKit refused the block with
    "Problem with residue number" -- several thousand atoms away from the
    ligand that actually caused it.
    """

    from mdprep.structure.reader import read_structure

    structure = read_structure(DATA / "boltz_wide_names.cif")
    residue = next(
        residue
        for residue in structure.residues
        if len(residue.atoms) == 4 and residue.id.resname not in {"HIS", "ASP"}
    )

    result = add_ligand_hydrogens(
        residue,
        smiles="OCC=O",
        ligand_id="wide_name_ligand",
    )

    assert count_hydrogens(result.residue) > 0


def test_a_porphyrin_is_not_read_bond_by_bond():
    """A SMILES draws two of a porphyrin's pyrroles as pyrrolides, with a
    single beta-beta bond, and two as pyrrolenines, with a double one. Which
    two is the writer's choice; the real ring has all four alike.

    2QFK's heme was refined with C2D-C3D at 1.52 A while its other three
    beta-beta bonds are 1.34-1.36 -- in both chains, so most likely the
    restraint dictionary of the time. The proposed heme SMILES happens to draw
    ring D as a pyrrolenine, and was refused as the wrong tautomer of heme b,
    of which there is none to pick.
    """
    from math import dist

    from mdprep.ligands.cofactors import KNOWN_METAL_FREE_COFACTORS

    heme = KNOWN_METAL_FREE_COFACTORS["HEM"]
    residue = residue_named(DATA / "cofactors" / "hem_2qfk.pdb", "HEM")
    xyz = {atom.name: (atom.x, atom.y, atom.z) for atom in residue.atoms}
    assert dist(xyz["C2D"], xyz["C3D"]) == pytest.approx(1.52, abs=0.01)

    result = add_ligand_hydrogens(
        residue, heme.smiles, ligand_id="a_hem_139", net_charge=heme.net_charge
    )

    # Four meso hydrogens, four methyls, two vinyls, two propionates.
    assert result.added_hydrogens == 30


def test_a_chlorin_is_still_read_bond_by_bond():
    """A chlorin's reduced pyrrole is a real single bond, not notation: its
    beta carbons carry four neighbours, and the porphyrin core does not match."""
    from rdkit import Chem

    from mdprep.ligands.hydrogenate import PORPHYRIN_CORE_SMARTS

    core = Chem.MolFromSmarts(PORPHYRIN_CORE_SMARTS)
    porphine = Chem.MolFromSmiles("c1cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3")
    chlorin = Chem.MolFromSmiles("C1Cc2cc3ccc(cc4ccc(cc5ccc(cc1n2)[nH]5)n4)[nH]3")

    assert porphine.HasSubstructMatch(core)
    assert not chlorin.HasSubstructMatch(core)
