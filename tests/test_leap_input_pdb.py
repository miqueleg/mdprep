from dataclasses import replace
from types import SimpleNamespace

import pytest

from mdprep.leap.residues import (
    LeapResidueError,
    prepare_leap_input_pdb,
    validate_tleap_ligand_coordinates,
)
from mdprep.structure.models import PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


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


def test_prepare_leap_input_anchors_ligand_to_extracted_parameterization_pdb(tmp_path):
    original = read_pdb("tests/data/protein_two_ligands.pdb")
    reference = next(residue for residue in original.residues if residue.id.resname == "SUB")
    extracted_pdb = tmp_path / "sub_501.pdb"
    write_pdb(
        PdbStructure(
            path=extracted_pdb,
            atoms=list(reference.atoms),
            residues=[reference],
            model_count=1,
        ),
        extracted_pdb,
    )
    shifted_atoms = [
        replace(atom, x=atom.x + 100.0, y=atom.y + 100.0)
        if atom.resname == "SUB"
        else atom
        for atom in original.atoms
    ]
    shifted = PdbStructure(
        path=original.path,
        atoms=shifted_atoms,
        residues=_build_residues(shifted_atoms),
        model_count=original.model_count,
        used_model=original.used_model,
        warnings=[],
    )
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]
    manifest = make_manifest(data)
    ligand_result = SimpleNamespace(
        ligands=[SimpleNamespace(ligand_id="sub_501", extracted_pdb_path=extracted_pdb)]
    )

    result = prepare_leap_input_pdb(
        shifted,
        tmp_path / "system.leap_input.pdb",
        manifest=manifest,
        ligand_result=ligand_result,
    )

    anchored = next(residue for residue in read_pdb(result.path).residues if residue.id.resname == "SUB")
    assert [(atom.x, atom.y, atom.z) for atom in anchored.atoms] == [
        (atom.x, atom.y, atom.z) for atom in reference.atoms
    ]
    assert result.ligand_coordinate_anchors[0]["max_coordinate_delta_applied_angstrom"] > 100.0


def test_prepare_leap_input_applies_extracted_unique_ligand_atom_names(tmp_path):
    original = read_pdb("tests/data/protein_two_ligands.pdb")
    target = next(residue for residue in original.residues if residue.id.resname == "SUB")
    duplicate_atoms = [
        replace(atom, name="C", element=target.atoms[index].element)
        for index, atom in enumerate(target.atoms)
    ]
    unique_atoms = [
        replace(duplicate_atoms[0], name="C1"),
        replace(duplicate_atoms[1], name="O1"),
    ]
    extracted_pdb = tmp_path / "sub_unique.pdb"
    write_pdb(
        PdbStructure(
            path=extracted_pdb,
            atoms=unique_atoms,
            residues=[
                ResidueRecord(
                    id=target.id,
                    atoms=unique_atoms,
                    record_names={"HETATM"},
                    original_index=0,
                )
            ],
            model_count=1,
        ),
        extracted_pdb,
    )
    structure_atoms = [
        duplicate_atoms.pop(0) if atom.resname == "SUB" else atom
        for atom in original.atoms
    ]
    structure = PdbStructure(
        path=original.path,
        atoms=structure_atoms,
        residues=_build_residues(structure_atoms),
        model_count=original.model_count,
        used_model=original.used_model,
        warnings=[],
    )
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]
    manifest = make_manifest(data)
    ligand_result = SimpleNamespace(
        ligands=[SimpleNamespace(ligand_id="sub_501", extracted_pdb_path=extracted_pdb)]
    )

    result = prepare_leap_input_pdb(
        structure,
        tmp_path / "system.leap_input.pdb",
        manifest=manifest,
        ligand_result=ligand_result,
    )

    anchored = next(residue for residue in read_pdb(result.path).residues if residue.id.resname == "SUB")
    assert anchored.atom_names() == ["C1", "O1"]


def test_tleap_ligand_coordinate_validation_fails_if_dry_build_moves_ligand(tmp_path):
    reference = read_pdb("tests/data/protein_two_ligands.pdb")
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]
    manifest = make_manifest(data)
    moved_atoms = [
        replace(atom, x=atom.x + 1.0)
        if atom.resname == "SUB"
        else atom
        for atom in reference.atoms
    ]
    moved_pdb = tmp_path / "dry.pdb"
    write_pdb(
        PdbStructure(
            path=moved_pdb,
            atoms=moved_atoms,
            residues=_build_residues(moved_atoms),
            model_count=1,
        ),
        moved_pdb,
    )

    with pytest.raises(LeapResidueError) as excinfo:
        validate_tleap_ligand_coordinates(
            manifest=manifest,
            reference_structure=reference,
            output_pdb=moved_pdb,
            stage="dry",
        )

    assert "moved during dry tleap build" in str(excinfo.value)




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
