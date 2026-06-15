from pathlib import Path

import pytest

from mdprep.leap.builder import TLeapBuildError, TLeapOutputs, build_tleap_script, solvation_command
from mdprep.leap.forcefields import forcefield_sources
from mdprep.leap.residues import DisulfideBondCommand, LigandParameterFiles
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_files(tmp_path):
    mol2 = tmp_path / "lig.final.mol2"
    frcmod = tmp_path / "lig.frcmod"
    mol2.write_text("", encoding="utf-8")
    frcmod.write_text("", encoding="utf-8")
    return [
        LigandParameterFiles(
            ligand_id="lig1",
            variable_name="LIG1",
            residue_name="SUB",
            final_mol2_path=mol2,
            final_frcmod_path=frcmod,
            atom_names=["C1"],
        )
    ]


def test_dry_tleap_script_contains_required_commands(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("lig1", "B", "SUB", 501)]
    manifest = make_manifest(data)
    sources = forcefield_sources(
        protein_forcefield=manifest.protein.forcefield,
        water_model=manifest.protein.water_model,
        ligands=manifest.ligands,
    )
    outputs = TLeapOutputs(
        prmtop=tmp_path / "system.dry.prmtop",
        inpcrd=tmp_path / "system.dry.inpcrd",
        pdb=tmp_path / "system.dry.pdb",
    )
    bond = DisulfideBondCommand({}, {}, 1, 2, "bond system.1.SG system.2.SG")

    script = build_tleap_script(
        sources=sources,
        ligands=ligand_files(tmp_path),
        input_pdb=tmp_path / "system.pdb",
        disulfide_bonds=[bond],
        outputs=outputs,
    )

    assert "source leaprc.protein.ff19SB" in script
    assert "source leaprc.gaff2" in script
    assert "LIG1 = loadmol2" in script
    assert "loadamberparams" in script
    assert "system = loadpdb" in script
    assert "check system" in script
    assert "charge system" in script
    assert "savepdb system system.dry.pdb" in script
    assert "saveamberparm system system.dry.prmtop system.dry.inpcrd" in script
    assert script.rstrip().endswith("quit")


def test_solvation_commands():
    assert solvation_command(box="truncated_octahedron", water_box="OPCBOX", buffer_angstrom=10.0).startswith(
        "solvateOct"
    )
    assert solvation_command(box="rectangular", water_box="TIP3PBOX", buffer_angstrom=8.0).startswith(
        "solvateBox"
    )
    with pytest.raises(TLeapBuildError):
        solvation_command(box="bad", water_box="TIP3PBOX", buffer_angstrom=8.0)
