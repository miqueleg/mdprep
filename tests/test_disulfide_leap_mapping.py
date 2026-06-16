import pytest

from mdprep.leap.residues import LeapResidueError, disulfide_bond_commands, residue_index_map
from mdprep.protonation.apply import apply_protonation_stage
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
    assert commands[0].command == "bond system.1.SG system.2.SG"


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
