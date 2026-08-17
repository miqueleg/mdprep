from pathlib import Path

from mdprep.structure.classify import (
    is_histidine,
    is_likely_ligand_or_cofactor,
    is_standard_protein_residue,
    is_titratable_residue,
    is_water_residue,
)
from mdprep.structure.pdb import read_pdb


DATA = Path("tests/data")


def test_standard_residues_are_not_likely_ligands_and_waters_are_waters():
    structure = read_pdb(DATA / "protein_with_waters.pdb")
    ala = structure.residues[0]
    water = next(residue for residue in structure.residues if residue.id.resname == "HOH")

    assert is_standard_protein_residue(ala)
    assert not is_likely_ligand_or_cofactor(ala)
    assert is_water_residue(water)
    assert not is_likely_ligand_or_cofactor(water)


def test_hetatm_non_water_residues_are_likely_ligands():
    structure = read_pdb(DATA / "protein_two_ligands.pdb")
    ligands = [residue for residue in structure.residues if is_likely_ligand_or_cofactor(residue)]

    assert [residue.id.resname for residue in ligands] == ["SUB", "COF"]


def test_hetatm_proline_is_a_ligand_not_a_polymer_residue(tmp_path):
    path = tmp_path / "free_proline.pdb"
    path.write_text(
        "HETATM    1  N   PRO A 502       0.000   0.000   0.000  1.00  0.00           N  \n"
        "HETATM    2  CA  PRO A 502       1.450   0.000   0.000  1.00  0.00           C  \n"
        "END\n",
        encoding="utf-8",
    )
    residue = read_pdb(path).residues[0]

    assert not is_standard_protein_residue(residue)
    assert is_likely_ligand_or_cofactor(residue)
    assert not is_titratable_residue(residue)


def test_atom_record_nonstandard_residue_is_likely_ligand():
    structure = read_pdb(DATA / "protein_atom_record_ligand_blank_chain.pdb")
    ligands = [residue for residue in structure.residues if is_likely_ligand_or_cofactor(residue)]

    assert [residue.id.resname for residue in ligands] == ["5NB"]
    assert ligands[0].record_names == {"ATOM"}


def test_histidines_and_titratable_residues_are_detected():
    structure = read_pdb(DATA / "protein_with_waters.pdb")
    histidines = [residue for residue in structure.residues if is_histidine(residue)]
    titratable = [residue for residue in structure.residues if is_titratable_residue(residue)]

    assert [residue.id.resname for residue in histidines] == ["HIS"]
    assert {residue.id.resname for residue in titratable} == {"HIS", "ASP"}
