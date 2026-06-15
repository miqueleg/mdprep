from pathlib import Path

from mdprep.structure.disulfides import detect_possible_disulfides
from mdprep.structure.pdb import read_pdb


DATA = Path("tests/data")


def test_sg_sg_pair_within_cutoff_is_detected():
    structure = read_pdb(DATA / "protein_disulfide.pdb")
    candidates = detect_possible_disulfides(structure.residues, cutoff_angstrom=2.2)

    assert len(candidates) == 1
    assert candidates[0].a.resid == 10
    assert candidates[0].b.resid == 20
    assert candidates[0].distance_angstrom == 2.05


def test_sg_sg_pair_outside_cutoff_is_not_detected():
    structure = read_pdb(DATA / "protein_disulfide.pdb")
    candidates = detect_possible_disulfides(structure.residues, cutoff_angstrom=1.5)

    assert candidates == []

