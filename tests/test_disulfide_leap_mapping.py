import pytest
from dataclasses import replace

from mdprep.leap.residues import (
    LeapResidueError,
    append_disulfide_conect_records,
    disulfide_bond_commands,
    residue_index_map,
)
from mdprep.protonation.apply import apply_protonation_stage
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
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
    assert commands[0].index_a == 1
    assert commands[0].index_b == 2
    assert commands[0].atom_serial_a == 2
    assert commands[0].atom_serial_b == 4
    assert commands[0].command == "CONECT 2 4"
    assert commands[0].pdb_conect_records == ("CONECT    2    4", "CONECT    4    2")


def test_duplicate_disulfide_residue_number_is_safe_with_conect_records():
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

    commands = disulfide_bond_commands(structure=protonation.structure, protonation_result=protonation)

    assert len(commands) == 1
    assert commands[0].pdb_conect_records == ("CONECT    2    4", "CONECT    4    2")


def test_unrelated_duplicate_residue_identity_does_not_block_disulfide_conect():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["protonation"]["method"] = "manual_only"
    manifest = make_manifest(data)
    structure = read_pdb("tests/data/protein_disulfide.pdb")
    water = _water_residue_same_identity_as(structure.residues[2])
    duplicate_structure = PdbStructure(
        path=structure.path,
        atoms=structure.atoms + water.atoms,
        residues=structure.residues + [water],
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=structure.warnings,
    )
    protonation = apply_protonation_stage(
        duplicate_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    commands = disulfide_bond_commands(structure=protonation.structure, protonation_result=protonation)

    assert len(commands) == 1
    assert commands[0].atom_serial_a == 2
    assert commands[0].atom_serial_b == 4


def test_disulfide_conect_records_are_inserted_before_end(tmp_path):
    pdb_path = tmp_path / "leap_input.pdb"
    pdb_path.write_text(
        "ATOM      1  SG  CYX A  10       0.000   0.000   0.000  1.00 20.00           S\n"
        "ATOM      2  SG  CYX A  20       2.050   0.000   0.000  1.00 20.00           S\n"
        "END\n",
        encoding="utf-8",
    )
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
    commands = disulfide_bond_commands(structure=protonation.structure, protonation_result=protonation)

    inserted = append_disulfide_conect_records(pdb_path, commands)

    assert inserted == ["CONECT    2    4", "CONECT    4    2"]
    lines = pdb_path.read_text(encoding="utf-8").splitlines()
    assert lines[-3:] == ["CONECT    2    4", "CONECT    4    2", "END"]


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


def _water_residue_same_identity_as(residue):
    atom = AtomRecord(
        serial=99,
        name="O",
        altloc=None,
        resname="HOH",
        chain_id=residue.id.chain_id,
        resid=residue.id.resid,
        icode=residue.id.icode,
        x=10.0,
        y=10.0,
        z=10.0,
        occupancy=1.0,
        bfactor=20.0,
        element="O",
        record_name="HETATM",
        original_line="",
    )
    return ResidueRecord(
        id=ResidueId(
            chain_id=residue.id.chain_id,
            resname="HOH",
            resid=residue.id.resid,
            icode=residue.id.icode,
        ),
        atoms=[atom],
        record_names={"HETATM"},
        original_index=residue.original_index + 200,
    )
