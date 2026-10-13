"""Which residues the temporary xTB hydrogenation hands to OpenMM.

These exercise the decision on a fake topology, so they run without OpenMM.
The round trip through the real Modeller is in
``tests/test_external_structure_repair.py``.
"""

from dataclasses import dataclass, field

import pytest

from mdprep.protonation.temporary_hydrogenation import (
    TemporaryHydrogenationError,
    _hold_back_hydrogenated_residues,
    _release_held_back_residues,
)


@dataclass
class FakeElement:
    symbol: str


@dataclass
class FakeAtom:
    name: str
    index: int
    element: FakeElement


@dataclass
class FakeChain:
    id: str


@dataclass
class FakeResidue:
    name: str
    index: int
    id: str
    chain: FakeChain
    atom_list: list[FakeAtom]

    def atoms(self):
        return iter(self.atom_list)


@dataclass
class FakeTopology:
    residue_list: list[FakeResidue]
    bond_list: list[tuple[FakeAtom, FakeAtom]] = field(default_factory=list)

    def residues(self):
        return iter(self.residue_list)

    def bonds(self):
        return iter(self.bond_list)


def _residue(name, index, atoms, *, chain="A", resid="1"):
    serial = index * 100
    built = []
    for atom_name in atoms:
        symbol = "H" if atom_name.startswith("H") else atom_name[0]
        built.append(FakeAtom(name=atom_name, index=serial, element=FakeElement(symbol)))
        serial += 1
    return FakeResidue(
        name=name,
        index=index,
        id=resid,
        chain=FakeChain(id=chain),
        atom_list=built,
    )


def _bond(residue, parent_name, hydrogen_name):
    by_name = {atom.name: atom for atom in residue.atom_list}
    return (by_name[parent_name], by_name[hydrogen_name])


def test_a_residue_whose_hydrogens_openmm_cannot_see_is_held_back():
    # A free lysine hydrogenated from a SMILES: OpenMM's LYS template knows
    # H on N and HXT on OXT, and nothing about HAA, HB or HBA.
    lysine = _residue(
        "LYS",
        0,
        ["N", "CA", "CB", "OXT", "H", "HAA", "HB", "HBA", "HXT"],
        resid="301",
    )
    topology = FakeTopology(
        [lysine],
        [_bond(lysine, "N", "H"), _bond(lysine, "OXT", "HXT")],
    )

    held_back = _hold_back_hydrogenated_residues(topology, [None])

    assert set(held_back) == {0}
    assert lysine.name != "LYS"
    assert _release_held_back_residues(topology, held_back) == ("A:LYS301",)
    assert lysine.name == "LYS"


def test_a_residue_openmm_knows_every_hydrogen_of_is_hydrogenated_normally():
    # Standard names, so the topology bonds them and OpenMM only adds what is
    # missing instead of starting a second set.
    alanine = _residue("ALA", 0, ["N", "CA", "CB", "H", "HA"])
    topology = FakeTopology(
        [alanine],
        [_bond(alanine, "N", "H"), _bond(alanine, "CA", "HA")],
    )

    assert _hold_back_hydrogenated_residues(topology, [None]) == {}
    assert alanine.name == "ALA"


def test_a_dehydrogenated_residue_is_hydrogenated_normally():
    aspartate = _residue("ASP", 0, ["N", "CA", "CB", "OD1", "OD2"])
    topology = FakeTopology([aspartate])

    assert _hold_back_hydrogenated_residues(topology, [None]) == {}


def test_a_residue_with_a_decided_protonation_state_is_never_held_back():
    # Asked for a state, OpenMM drops the hydrogens it does not recognise
    # before building that state, so it cannot duplicate anything -- and
    # holding the residue back would leave the decided state unbuilt.
    glutamate = _residue("GLU", 0, ["N", "CA", "CB", "OE1", "H", "HB", "HBA"])
    topology = FakeTopology([glutamate], [_bond(glutamate, "N", "H")])

    assert _hold_back_hydrogenated_residues(topology, ["GLH"]) == {}
    assert glutamate.name == "GLU"


def test_a_held_back_residue_that_came_back_changed_is_an_error():
    lysine = _residue("LYS", 0, ["N", "CB", "H", "HB", "HBA"], resid="301")
    topology = FakeTopology([lysine], [_bond(lysine, "N", "H")])
    held_back = _hold_back_hydrogenated_residues(topology, [None])

    lysine.atom_list.append(FakeAtom(name="HB2", index=99, element=FakeElement("H")))

    with pytest.raises(TemporaryHydrogenationError) as error:
        _release_held_back_residues(topology, held_back)

    assert "A:LYS301" in str(error.value)
    assert lysine.name == "LYS"
