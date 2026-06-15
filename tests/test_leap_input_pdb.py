from dataclasses import replace

from mdprep.leap.residues import prepare_leap_input_pdb
from mdprep.structure.models import PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb


def test_prepare_leap_input_renames_water_and_preserves_ligand_atoms(tmp_path):
    structure = read_pdb("tests/data/protein_with_waters.pdb")
    atoms = []
    for atom in structure.atoms:
        if atom.resname == "HOH" and atom.name == "O":
            atoms.append(replace(atom, name="OW"))
        else:
            atoms.append(atom)
    structure = PdbStructure(
        path=structure.path,
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=[],
    )

    result = prepare_leap_input_pdb(structure, tmp_path / "system.leap_input.pdb")

    parsed = read_pdb(result.path)
    water = next(residue for residue in parsed.residues if residue.id.resname == "WAT")
    assert water.atom_names()[0] == "O"
    assert parsed.atoms[0].name == structure.atoms[0].name
    assert result.water_renames


def _build_residues(atoms):
    grouped = {}
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    residues = []
    for index, ((chain, resname, resid, icode), residue_atoms) in enumerate(grouped.items()):
        residues.append(
            ResidueRecord(
                id=ResidueId(chain_id=chain, resname=resname, resid=resid, icode=icode),
                atoms=residue_atoms,
                record_names={atom.record_name for atom in residue_atoms},
                original_index=index,
            )
        )
    return residues
