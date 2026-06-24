import pytest
from dataclasses import replace

from mdprep.leap.residues import LeapResidueError, disulfide_bond_commands, residue_index_map
from mdprep.protonation.apply import apply_protonation_stage
from mdprep.structure.models import PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import make_manifest, manifest_data


def test_residue_index_map_and_disulfide_bond_command():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    manifest = make_manifest(data)
    structure = read_pdb("tests/data/protein_disulfide.pdb")
    protonation = apply_protonation_stage(
        structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    mapping = residue_index_map(protonation.structure)
    assert mapping[("A", 10, None)] == 1
    commands = disulfide_bond_commands(
        structure=protonation.structure,
        protonation_result=protonation,
    )

    assert len(commands) == 1
    assert commands[0].command == "bond system.10.SG system.20.SG"


def test_duplicate_disulfide_residue_number_fails_before_tleap():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["disulfides"]["force"] = [
        {
            "a": {"chain": "A", "resname": "CYS", "resid": 10, "icode": None},
            "b": {"chain": "A", "resname": "CYS", "resid": 20, "icode": None},
            "reason": "test",
        }
    ]
    data["disulfides"]["auto_detect"] = False
    manifest = make_manifest(data)
    structure = read_pdb("tests/data/protein_disulfide.pdb")
    extra = _copy_residue_with_chain(structure.residues[0], chain_id="B")
    ambiguous_structure = PdbStructure(
        path=structure.path,
        atoms=structure.atoms + extra.atoms,
        residues=structure.residues + [extra],
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=structure.warnings,
    )
    protonation = apply_protonation_stage(
        ambiguous_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    with pytest.raises(LeapResidueError, match="not unique"):
        disulfide_bond_commands(structure=protonation.structure, protonation_result=protonation)


def test_cyx_without_pair_fails():
    data = manifest_data("tests/data/protein_cyx_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    data["disulfides"]["auto_detect"] = False
    manifest = make_manifest(data)
    structure = read_pdb("tests/data/protein_cyx_disulfide.pdb")
    protonation = apply_protonation_stage(
        structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    with pytest.raises(LeapResidueError):
        disulfide_bond_commands(structure=protonation.structure, protonation_result=protonation)


def _copy_residue_with_chain(residue, *, chain_id: str):
    atoms = [replace(atom, chain_id=chain_id) for atom in residue.atoms]
    return ResidueRecord(
        id=ResidueId(chain_id=chain_id, resname=residue.id.resname, resid=residue.id.resid, icode=residue.id.icode),
        atoms=atoms,
        record_names=residue.record_names,
        original_index=residue.original_index + 100,
    )
