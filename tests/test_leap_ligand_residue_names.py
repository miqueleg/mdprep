"""A declared ligand must reach tleap under a residue name tleap has not taken.

6NIE's BesD substrate is a free lysine recorded as HETATM, declared in the
manifest as a ligand with its own SMILES and GAFF parameters. Its residue name
is LYS, which every Amber protein force field already defines, so tleap built
it from the protein template -- as NLYS, because it opens a chain, which no
``LYS = <ligand unit>`` binding can intercept -- and stopped on the first
ligand-named hydrogen it could not type.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mdprep.ambertools.mol2 import read_mol2, write_mol2_under_residue_name
from mdprep.leap.builder import TLeapOutputs, build_tleap_script
from mdprep.leap.forcefields import forcefield_sources
from mdprep.leap.residues import (
    LEAP_RESERVED_RESIDUE_NAMES,
    plan_ligand_residue_renames,
    prepare_leap_input_pdb,
    validate_ligand_parameter_files,
)
from mdprep.structure.models import PdbStructure
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


FIXTURE = "tests/data/protein_two_ligands.pdb"


def _structure_with_free_amino_acid():
    """The fixture's SUB ligand, renamed to a residue Amber already defines."""

    original = read_pdb(FIXTURE)
    atoms = [
        replace(atom, resname="LYS") if atom.resname == "SUB" else atom
        for atom in original.atoms
    ]
    return PdbStructure(
        path=original.path,
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=original.model_count,
        used_model=original.used_model,
        warnings=[],
    )


def _manifest_with_free_amino_acid():
    data = manifest_data(FIXTURE)
    entry = ligand_entry("sub_501", "B", "LYS", 501)
    entry["expected_formula"] = "CO"
    data["ligands"] = [entry, ligand_entry("cof_601", "C", "COF", 601)]
    return make_manifest(data)


def _write_mol2(path, residue_name, atoms):
    atom_lines = [
        f"{index:7d} {atom.name:<8} {atom.x:10.4f} {atom.y:10.4f} {atom.z:10.4f} "
        f"{'c3' if atom.element == 'C' else 'o':<8} {1:4d} {residue_name:<8} "
        f"{0.25 if index == 1 else -0.25:10.6f}"
        for index, atom in enumerate(atoms, start=1)
    ]
    path.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                residue_name,
                f" {len(atoms)} 1 1 0 0",
                "SMALL",
                "USER_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                *atom_lines,
                "@<TRIPOS>BOND",
                "     1    1    2 1",
                "@<TRIPOS>SUBSTRUCTURE",
                f"     1 {residue_name}         1 TEMP              0 ****  ****    0 ROOT",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _ligand_result(tmp_path, structure):
    ligand = next(
        residue for residue in structure.residues if residue.id.resid == 501
    )
    mol2 = tmp_path / "sub_501.final.mol2"
    frcmod = tmp_path / "sub_501.frcmod"
    _write_mol2(mol2, ligand.id.resname, ligand.atoms)
    frcmod.write_text("MASS\n\n", encoding="utf-8")
    cofactor = next(
        residue for residue in structure.residues if residue.id.resid == 601
    )
    cof_mol2 = tmp_path / "cof_601.final.mol2"
    cof_frcmod = tmp_path / "cof_601.frcmod"
    _write_mol2(cof_mol2, cofactor.id.resname, cofactor.atoms)
    cof_frcmod.write_text("MASS\n\n", encoding="utf-8")
    return SimpleNamespace(
        ligands=[
            SimpleNamespace(
                ligand_id="sub_501",
                final_mol2_path=mol2,
                final_frcmod_path=frcmod,
                extracted_pdb_path=None,
            ),
            SimpleNamespace(
                ligand_id="cof_601",
                final_mol2_path=cof_mol2,
                final_frcmod_path=cof_frcmod,
                extracted_pdb_path=None,
            ),
        ]
    )


def test_a_ligand_named_after_an_amino_acid_is_renamed_and_the_others_are_not():
    structure = _structure_with_free_amino_acid()

    renames = plan_ligand_residue_renames(
        manifest=_manifest_with_free_amino_acid(),
        structure=structure,
    )

    assert [rename.ligand_id for rename in renames] == ["sub_501"]
    rename = renames[0]
    assert rename.original_resname == "LYS"
    assert rename.tleap_resname not in LEAP_RESERVED_RESIDUE_NAMES
    assert rename.tleap_resname not in {
        residue.id.resname for residue in structure.residues
    }


def test_the_renamed_ligand_reaches_the_leap_input_pdb_unchanged_but_for_its_name(
    tmp_path,
):
    structure = _structure_with_free_amino_acid()
    manifest = _manifest_with_free_amino_acid()
    renames = plan_ligand_residue_renames(manifest=manifest, structure=structure)
    ligand = next(residue for residue in structure.residues if residue.id.resid == 501)

    result = prepare_leap_input_pdb(
        structure,
        tmp_path / "system.leap_input.pdb",
        ligand_residue_renames=renames,
    )

    written = read_pdb(result.path)
    renamed = next(residue for residue in written.residues if residue.id.resid == 501)
    assert renamed.id.resname == renames[0].tleap_resname
    assert renamed.atom_names() == ligand.atom_names()
    assert [(atom.x, atom.y, atom.z) for atom in renamed.atoms] == [
        (atom.x, atom.y, atom.z) for atom in ligand.atoms
    ]
    # The protein's own residues keep their names, which is the whole point of
    # not binding LYS to the ligand unit.
    assert [residue.id.resname for residue in written.residues if residue.id.chain_id == "A"] == ["ALA"]
    assert result.ligand_residue_renames[0]["original_resname"] == "LYS"


def test_the_mol2_handed_to_tleap_carries_the_new_name_and_the_fitted_charges(
    tmp_path,
):
    structure = _structure_with_free_amino_acid()
    manifest = _manifest_with_free_amino_acid()
    renames = plan_ligand_residue_renames(manifest=manifest, structure=structure)
    ligand_result = _ligand_result(tmp_path, structure)
    fitted = read_mol2(ligand_result.ligands[0].final_mol2_path)

    files = validate_ligand_parameter_files(
        manifest=manifest,
        structure=structure,
        ligand_result=ligand_result,
        residue_renames=renames,
    )

    substrate = next(item for item in files if item.ligand_id == "sub_501")
    assert substrate.residue_name == renames[0].tleap_resname
    handed = read_mol2(substrate.final_mol2_path)
    assert {atom.subst_name for atom in handed.atoms} == {renames[0].tleap_resname}
    assert [atom.name for atom in handed.atoms] == [atom.name for atom in fitted.atoms]
    assert [atom.charge for atom in handed.atoms] == [
        atom.charge for atom in fitted.atoms
    ]
    assert [(atom.x, atom.y, atom.z) for atom in handed.atoms] == [
        (atom.x, atom.y, atom.z) for atom in fitted.atoms
    ]
    # An untouched ligand is handed its own fitted mol2, not a copy.
    cofactor = next(item for item in files if item.ligand_id == "cof_601")
    assert cofactor.final_mol2_path == ligand_result.ligands[1].final_mol2_path


def test_the_tleap_script_never_rebinds_a_force_field_residue_name(tmp_path):
    structure = _structure_with_free_amino_acid()
    manifest = _manifest_with_free_amino_acid()
    renames = plan_ligand_residue_renames(manifest=manifest, structure=structure)
    files = validate_ligand_parameter_files(
        manifest=manifest,
        structure=structure,
        ligand_result=_ligand_result(tmp_path, structure),
        residue_renames=renames,
    )

    script = build_tleap_script(
        sources=forcefield_sources(
            protein_forcefield=manifest.protein.forcefield,
            water_model=manifest.protein.water_model,
            ligands=manifest.ligands,
        ),
        ligands=files,
        input_pdb=tmp_path / "system.leap_input.pdb",
        disulfide_bonds=[],
        outputs=TLeapOutputs(
            prmtop=tmp_path / "system.prmtop",
            inpcrd=tmp_path / "system.inpcrd",
            pdb=tmp_path / "system.pdb",
        ),
        work_dir=tmp_path,
    )

    assignments = {
        line.split("=")[0].strip()
        for line in script.splitlines()
        if "=" in line and "loadmol2" not in line
    }
    assert not assignments & LEAP_RESERVED_RESIDUE_NAMES
    assert renames[0].tleap_resname in assignments


def test_a_ligand_name_that_is_free_is_left_alone(tmp_path):
    structure = read_pdb(FIXTURE)
    data = manifest_data(FIXTURE)
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]

    renames = plan_ligand_residue_renames(
        manifest=make_manifest(data),
        structure=structure,
    )

    assert renames == []
    result = prepare_leap_input_pdb(
        structure,
        tmp_path / "system.leap_input.pdb",
        ligand_residue_renames=renames,
    )
    assert result.ligand_residue_renames == []
    assert "SUB" in {residue.id.resname for residue in read_pdb(result.path).residues}


def test_a_renamed_mol2_copy_changes_nothing_but_the_residue_name(tmp_path):
    structure = _structure_with_free_amino_acid()
    source = tmp_path / "source.mol2"
    ligand = next(residue for residue in structure.residues if residue.id.resid == 501)
    _write_mol2(source, "LYS", ligand.atoms)
    before = read_mol2(source)

    write_mol2_under_residue_name(source, "LY1", tmp_path / "renamed.mol2")

    after = read_mol2(tmp_path / "renamed.mol2")
    assert {atom.subst_name for atom in after.atoms} == {"LY1"}
    assert [
        (atom.atom_id, atom.name, atom.x, atom.y, atom.z, atom.atom_type, atom.charge)
        for atom in after.atoms
    ] == [
        (atom.atom_id, atom.name, atom.x, atom.y, atom.z, atom.atom_type, atom.charge)
        for atom in before.atoms
    ]
    assert "LYS" not in (tmp_path / "renamed.mol2").read_text().split("@<TRIPOS>BOND")[1]


def _build_residues(atoms):
    from mdprep.leap.residues import _build_residues as build

    return build(atoms)
