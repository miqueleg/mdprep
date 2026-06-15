import pytest
from dataclasses import replace

from mdprep.protonation.histidine_geometry import (
    HistidineGeometryError,
    build_tautomer_xyz_atoms,
    place_histidine_tautomer_hydrogen,
)
from mdprep.structure.pdb import read_pdb


def histidine():
    structure = read_pdb("tests/data/protein_histidine_ring.pdb")
    return next(residue for residue in structure.residues if residue.id.resname == "HIS")


def distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def test_places_hid_hydrogen_on_nd1_at_expected_distance():
    residue = histidine()
    hydrogen = place_histidine_tautomer_hydrogen(residue, tautomer="HID")
    nd1 = next(atom for atom in residue.atoms if atom.name == "ND1")

    assert hydrogen.name == "HD1"
    assert distance(hydrogen, nd1) == pytest.approx(1.01)


def test_places_hie_hydrogen_on_ne2_at_expected_distance():
    residue = histidine()
    hydrogen = place_histidine_tautomer_hydrogen(residue, tautomer="HIE")
    ne2 = next(atom for atom in residue.atoms if atom.name == "NE2")

    assert hydrogen.name == "HE2"
    assert distance(hydrogen, ne2) == pytest.approx(1.01)


def test_degenerate_geometry_fails_clearly():
    residue = histidine()
    cg = next(atom for atom in residue.atoms if atom.name == "CG")
    ce1 = next(atom for atom in residue.atoms if atom.name == "CE1")
    nd1_index = next(index for index, atom in enumerate(residue.atoms) if atom.name == "ND1")
    residue.atoms[nd1_index] = replace(
        residue.atoms[nd1_index],
        x=(cg.x + ce1.x) / 2,
        y=(cg.y + ce1.y) / 2,
        z=(cg.z + ce1.z) / 2,
    )

    with pytest.raises(HistidineGeometryError):
        place_histidine_tautomer_hydrogen(residue, tautomer="HID")


def test_generated_xyz_has_one_added_hydrogen_and_preserves_heavy_coordinates():
    residue = histidine()
    atoms = build_tautomer_xyz_atoms([residue], residue, tautomer="HID")
    heavy = [atom for atom in residue.atoms if atom.element != "H"]

    assert len(atoms) == len(heavy) + 1
    assert atoms[-1].element == "H"
    assert [(atom.x, atom.y, atom.z) for atom in atoms[:-1]] == [
        (atom.x, atom.y, atom.z) for atom in heavy
    ]
