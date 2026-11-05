"""Crystal waters sitting on another atom, left out by the builders.

A water in a predicted model, or in a dimer assembled by hand, was placed for
other coordinates: a user's AlphaFold dimer carried 85 waters closer than
2.2 A to a protein atom or to each other, some 0.4 A from a side chain's,
and the refinement then could not place a lysine's hydrogens. A
high-resolution crystal has a few too, for the side-chain conformer the
build does not use. Each is left out, with the atom it overlaps, unless what
it overlaps is itself left out; a water on a metal never is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mdprep.config.decisions import build_manifest, plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.structure.models import PdbStructure, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb

DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
AKG_SMILES = "[O-]C(=O)CCC(=O)C(=O)[O-]"
ANSWERS = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": True,
    "structure.repair.add_missing_heavy_atoms": True,
    "ligand.a_dsu_896.include": False,
    "ligand.a_akg_897.net_charge": -2,
    "ligand.a_akg_897.smiles": AKG_SMILES,
    "ligand.a_akg_897.charge_method": "am1bcc",
    "metal.a_fe2_900_site.model": "nonbonded",
}
# A crystal water of chain A 2.14 A from Lys174's carbonyl oxygen: shorter
# than any hydrogen bond, and left out of 2FCT as deposited.
DEPOSITED = ("A", 1034)


@pytest.fixture(scope="module")
def deposition() -> PdbStructure:
    return read_pdb(DEPOSITION)


def with_waters_at(structure: PdbStructure, places: dict[tuple[str, int], tuple[float, float, float]]):
    atoms = [
        replace(atom, x=places[key][0], y=places[key][1], z=places[key][2], original_line="")
        if (key := (atom.chain_id, atom.resid)) in places and atom.resname == "HOH"
        else atom
        for atom in structure.atoms
    ]
    return PdbStructure(path=structure.path, atoms=atoms, residues=build_residues(atoms), model_count=1)


def atom_of(structure: PdbStructure, chain: str, resid: int, name: str):
    return next(a for a in structure.atoms if (a.chain_id, a.resid, a.name) == (chain, resid, name))


def removed_waters(structure: PdbStructure, tmp_path: Path, answers=ANSWERS) -> dict[int, str]:
    path = tmp_path / "waters.pdb"
    write_pdb(structure, path)
    data = build_manifest(plan_decisions(path), answers)
    ManifestConfig.model_validate(data)
    return {
        item["selector"]["resid"]: item["reason"]
        for item in data["structure"].get("remove_residues", [])
        if item["selector"]["resname"] == "HOH"
    }


def test_a_water_inside_a_side_chain_is_left_out(tmp_path, deposition):
    nz = atom_of(deposition, "A", 174, "NZ")
    structure = with_waters_at(deposition, {("A", 904): (nz.x + 0.4, nz.y, nz.z)})

    removed = removed_waters(structure, tmp_path)

    assert removed[904].startswith("crystal water overlapping A:LYS174@NZ (0.40 A away)")
    assert removed[DEPOSITED[1]].startswith("crystal water overlapping A:LYS174@O (2.14 A away)")


def test_of_two_waters_on_one_spot_the_second_goes(tmp_path, deposition):
    first = atom_of(deposition, "A", 905, "O")
    structure = with_waters_at(deposition, {("A", 906): (first.x + 0.3, first.y, first.z)})

    removed = removed_waters(structure, tmp_path)

    assert 905 not in removed
    assert removed[906].startswith("crystal water overlapping A:HOH905@O (0.30 A away)")


def test_a_water_on_the_iron_is_never_left_out(tmp_path, deposition):
    """A:HOH1072 is the iron's water. Put a protein atom's worth of crowding
    beside it and it is still part of the site."""

    water = atom_of(deposition, "A", 1072, "O")
    iron = atom_of(deposition, "A", 900, "FE")
    # One angstrom further out along the iron's bond: crowding the water,
    # not bound to the iron itself.
    away = [w - f for w, f in zip((water.x, water.y, water.z), (iron.x, iron.y, iron.z))]
    norm = sum(v * v for v in away) ** 0.5
    place = tuple(w + v / norm for w, v in zip((water.x, water.y, water.z), away))
    structure = with_waters_at(deposition, {("A", 907): place})

    removed = removed_waters(structure, tmp_path)

    assert 1072 not in removed
    assert removed[907].startswith("crystal water overlapping A:HOH1072@O (1.00 A away)")


def test_a_water_overlapping_only_what_is_left_out_stays(tmp_path, deposition):
    """The detergent is declined and removed, so a water in its place is in
    nobody's way."""

    detergent = atom_of(deposition, "A", 896, "O1N")
    structure = with_waters_at(
        deposition, {("A", 908): (detergent.x + 0.5, detergent.y, detergent.z)}
    )

    assert 908 not in removed_waters(structure, tmp_path)


def test_without_the_crystal_waters_nothing_is_said(tmp_path, deposition):
    removed = removed_waters(
        deposition, tmp_path, {**ANSWERS, "structure.keep_crystal_waters": False}
    )

    assert removed == {}
