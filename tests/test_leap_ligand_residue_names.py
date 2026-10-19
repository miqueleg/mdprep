"""A declared ligand whose residue name a force field owns is refused.

6NIE's BesD substrate is a free lysine recorded as HETATM, declared in the
manifest as a ligand with its own SMILES and GAFF parameters. Its residue name
is LYS, which every Amber protein force field already defines, so tleap built
it from the protein template -- as NLYS, because it opens a chain, which no
``LYS = <ligand unit>`` binding can intercept -- and stopped on the first
ligand-named hydrogen it could not type.

mdprep used to rename the ligand for the build: LYS became LY1. That kept
tleap happy and moved the problem, because the prepared system then described
a residue under a name the input never used, and every later reader had to
know about the substitution -- the MCPB site builder, the QM/MM point-charge
reader, the refinement selection. Those are separate statements of one fact,
and they drifted.

So mdprep refuses instead, and says to rename the residue in the input. The
refusal is raised twice over, deliberately: once by the decision planner, so
the GUI rejects the file before asking a single question, and once in the
preparation workflow, for a manifest written by hand.
"""

from dataclasses import replace

import pytest

from mdprep.ambertools.mol2 import read_mol2, write_mol2_under_residue_name
from mdprep.config.decisions import DecisionError, plan_decisions
from mdprep.leap.residues import (
    LEAP_RESERVED_RESIDUE_NAMES,
    LeapResidueError,
    check_ligand_residue_names,
    prepare_leap_input_pdb,
)
from mdprep.structure.models import PdbStructure
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


FIXTURE = "tests/data/protein_two_ligands.pdb"


def _structure_named(resname):
    """The fixture's SUB ligand, renamed to whatever is under test."""

    original = read_pdb(FIXTURE)
    atoms = [
        replace(atom, resname=resname) if atom.resname == "SUB" else atom
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


def _manifest_named(resname):
    data = manifest_data(FIXTURE)
    entry = ligand_entry("sub_501", "B", resname, 501)
    entry["expected_formula"] = "CO"
    data["ligands"] = [entry, ligand_entry("cof_601", "C", "COF", 601)]
    return make_manifest(data)


def test_a_ligand_named_after_an_amino_acid_is_refused_by_name():
    with pytest.raises(LeapResidueError) as excinfo:
        check_ligand_residue_names(
            manifest=_manifest_named("LYS"), structure=_structure_named("LYS")
        )

    message = str(excinfo.value)
    # Which residue, why it cannot be built, and what to do about it.
    assert "LYS" in message
    assert "an amino acid" in message
    assert "Rename" in message


def test_the_refusal_does_not_name_a_ligand_whose_name_is_free():
    """The cofactor in the same manifest keeps its own name, so a message
    listing it would send the user looking for a problem that is not there."""

    message = str(
        pytest.raises(
            LeapResidueError,
            check_ligand_residue_names,
            manifest=_manifest_named("LYS"),
            structure=_structure_named("LYS"),
        ).value
    )

    assert "COF" not in message
    assert "cof_601" not in message


@pytest.mark.parametrize(
    "resname, reason",
    [
        ("LYS", "an amino acid"),
        ("NGLY", "an amino acid"),
        ("HOH", "water"),
        ("DA", "a nucleic acid residue"),
        ("ZN2", "a monatomic ion"),
    ],
)
def test_the_refusal_says_which_template_claims_the_name(resname, reason):
    """A ligand colliding with an amino acid is a different mistake from a
    strand of DNA read as a pile of ligands, and the advice differs."""

    with pytest.raises(LeapResidueError) as excinfo:
        check_ligand_residue_names(
            manifest=_manifest_named(resname), structure=_structure_named(resname)
        )

    assert reason in str(excinfo.value)


def test_a_nucleic_acid_name_says_renaming_will_not_help():
    """Renaming DA to LIG would get it past tleap's name lookup and then fail
    for the real reason, which is that mdprep has no nucleic acid force
    field. Telling the user to rename it would waste their afternoon."""

    with pytest.raises(LeapResidueError) as excinfo:
        check_ligand_residue_names(
            manifest=_manifest_named("DA"), structure=_structure_named("DA")
        )

    assert "no nucleic acid force field" in str(excinfo.value)


def test_the_gui_rejects_the_structure_before_asking_anything(tmp_path):
    """plan_decisions is what the page runs on an uploaded file. Raising here
    is what makes the GUI refuse: the load path restores the previous
    structure and shows the message, so no question is ever asked about a
    ligand that cannot be built."""

    path = tmp_path / "free_lysine.pdb"
    write_pdb(_structure_named("LYS"), path)

    with pytest.raises(DecisionError) as excinfo:
        plan_decisions(path)

    assert "LYS" in str(excinfo.value)
    assert "Rename" in str(excinfo.value)


def test_mdprep_and_the_gui_refuse_in_the_same_words(tmp_path):
    """Two refusals, one wording. A student who hits this in the browser and
    then on the command line must not be told two different things."""

    path = tmp_path / "free_lysine.pdb"
    structure = _structure_named("LYS")
    write_pdb(structure, path)

    planner = str(pytest.raises(DecisionError, plan_decisions, path).value)
    workflow = str(
        pytest.raises(
            LeapResidueError,
            check_ligand_residue_names,
            manifest=_manifest_named("LYS"),
            structure=structure,
        ).value
    )

    # The planner identifies the residue by position and the manifest check by
    # ligand id, so only the advice itself has to match -- but it has to.
    advice = "Rename it in the input structure"
    assert advice in planner and advice in workflow
    assert planner.split(". ", 1)[1] == workflow.split(". ", 1)[1]


def test_a_ligand_name_that_is_free_is_left_alone(tmp_path):
    structure = read_pdb(FIXTURE)
    data = manifest_data(FIXTURE)
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]

    check_ligand_residue_names(manifest=make_manifest(data), structure=structure)

    result = prepare_leap_input_pdb(structure, tmp_path / "system.leap_input.pdb")
    assert result.ligand_residue_renames == []
    assert "SUB" in {residue.id.resname for residue in read_pdb(result.path).residues}


def test_no_reserved_name_is_ever_bound_in_a_tleap_script():
    """The guard behind the refusal: these are the names a ligand may not
    use, and the set is the one the force fields define."""

    assert {"LYS", "NLYS", "HOH", "WAT", "ZN", "ZN2", "DA", "CL"} <= (
        LEAP_RESERVED_RESIDUE_NAMES
    )
    assert "SUB" not in LEAP_RESERVED_RESIDUE_NAMES
    assert "LIG" not in LEAP_RESERVED_RESIDUE_NAMES


def test_a_renamed_mol2_copy_changes_nothing_but_the_residue_name(tmp_path):
    structure = _structure_named("LYS")
    source = tmp_path / "source.mol2"
    ligand = next(residue for residue in structure.residues if residue.id.resid == 501)
    atom_lines = [
        f"{index:7d} {atom.name:<8} {atom.x:10.4f} {atom.y:10.4f} {atom.z:10.4f} "
        f"{'c3' if atom.element == 'C' else 'o':<8} {1:4d} {'LYS':<8} "
        f"{0.25 if index == 1 else -0.25:10.6f}"
        for index, atom in enumerate(ligand.atoms, start=1)
    ]
    source.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "LYS",
                f" {len(ligand.atoms)} 1 1 0 0",
                "SMALL",
                "USER_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                *atom_lines,
                "@<TRIPOS>BOND",
                "     1    1    2 1",
                "@<TRIPOS>SUBSTRUCTURE",
                "     1 LYS         1 TEMP              0 ****  ****    0 ROOT",
                "",
            ]
        ),
        encoding="utf-8",
    )
    before = read_mol2(source)

    write_mol2_under_residue_name(source, "LIG", tmp_path / "renamed.mol2")

    after = read_mol2(tmp_path / "renamed.mol2")
    assert {atom.subst_name for atom in after.atoms} == {"LIG"}
    assert [
        (atom.atom_id, atom.name, atom.x, atom.y, atom.z, atom.atom_type, atom.charge)
        for atom in after.atoms
    ] == [
        (atom.atom_id, atom.name, atom.x, atom.y, atom.z, atom.atom_type, atom.charge)
        for atom in before.atoms
    ]


def _build_residues(atoms):
    from mdprep.leap.residues import _build_residues as build

    return build(atoms)


def test_the_preparation_workflow_refuses_before_it_does_any_work(tmp_path):
    """The three builders that would notice a reserved name are all on the
    far side of protonation, ligand parameterisation and the metal site. A
    user whose ligand is called LYS must not be told so after an hour of
    quantum chemistry, so prepare_system checks the normalized structure
    first and stops there."""

    import yaml

    from mdprep.workflows.prepare import PrepareWorkflowError, prepare_system

    structure_path = tmp_path / "free_lysine.pdb"
    write_pdb(_structure_named("LYS"), structure_path)

    data = manifest_data(str(structure_path))
    entry = ligand_entry("sub_501", "B", "LYS", 501)
    entry["expected_formula"] = "CO"
    data["ligands"] = [entry]
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    manifest_path = tmp_path / "system.yaml"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(PrepareWorkflowError) as excinfo:
        prepare_system(manifest_path)

    assert "LYS" in str(excinfo.value)
    assert "Rename" in str(excinfo.value)
    # Nothing past the structure stage ran.
    output = tmp_path / "prepared"
    assert not (output / "protonation").exists()
    assert not (output / "ligands").exists()
