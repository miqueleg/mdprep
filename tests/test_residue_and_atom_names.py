"""Names a structure arrives with that cannot be built as they are.

A molecule editor names atoms by element -- Avogadro writes every carbon as C
-- and writes everything it holds as one residue. A hand-assembled dimer
names the second copy's substrate THC2. A file can call two different
molecules by one name. tleap builds every residue of one name from the one
template of that name, and matches template atoms by name, so each of these
used to stop the build with an instruction to edit the input.

mdprep now renames: atoms as the structure is read, residues in the manifest
it writes, each rename said where it happens and listed in the structure
report.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mdprep.config.decisions import build_manifest, plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.structure.models import PdbStructure, build_residues
from mdprep.structure.normalize import StructureNormalizationError, normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.rename import ResidueRenameError, apply_residue_renames
from mdprep.structure.repair import _write_repair_input
from mdprep.structure.writer import write_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data

DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
AKG_SMILES = "[O-]C(=O)CCC(=O)C(=O)[O-]"

# Ethanol drawn in an editor, every atom named by its element, and a water
# drawn beside it: one residue, two molecules.
EDITOR = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
HETATM    4  C   UNL B   1       8.000   0.000   0.000  1.00  0.00           C
HETATM    5  C   UNL B   1       9.520   0.000   0.000  1.00  0.00           C
HETATM    6  O   UNL B   1      10.100   1.200   0.000  1.00  0.00           O
HETATM    7  H   UNL B   1       7.600   1.000   0.000  1.00  0.00           H
HETATM    8  H   UNL B   1       7.600  -0.500   0.900  1.00  0.00           H
HETATM    9  O   UNL B   1      20.000  10.000  10.000  1.00  0.00           O
HETATM   10  H   UNL B   1      20.960  10.000  10.000  1.00  0.00           H
HETATM   11  H   UNL B   1      19.760  10.930  10.000  1.00  0.00           H
END
"""


def written(text: str, path: Path) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def with_atoms(structure: PdbStructure, atoms) -> PdbStructure:
    atoms = list(atoms)
    return PdbStructure(
        path=structure.path, atoms=atoms, residues=build_residues(atoms), model_count=1
    )


# --------------------------------------------------------------------------
# Atoms an editor named alike
# --------------------------------------------------------------------------


def test_an_editors_atoms_are_named_apart_as_the_structure_is_read(tmp_path):
    structure = read_pdb(written(EDITOR, tmp_path / "editor.pdb"))

    names = {residue.id.display(): residue.atom_names() for residue in structure.residues}
    assert names["B:UNL1"] == ["C1", "C2", "O", "H1", "H2"]
    # The protein's names are its template's, and untouched.
    assert names["A:ALA1"] == ["N", "CA", "C"]
    assert any(
        "B:UNL1 has atoms sharing a name (C x2, H x2)" in warning
        and "C->C1, C->C2, H->H1, H->H2" in warning
        for warning in structure.warnings
    )


def test_molecules_written_as_one_residue_become_residues_of_their_own(tmp_path):
    structure = read_pdb(written(EDITOR, tmp_path / "editor.pdb"))

    by_name = {residue.id.display(): residue for residue in structure.residues}
    assert set(by_name) == {"A:ALA1", "B:UNL1", "B:UNL2"}
    # The water keeps its own hydrogens and its own names.
    assert by_name["B:UNL2"].atom_names() == ["O", "H1", "H2"]
    assert any(
        "B:UNL1 holds 2 separate molecules under one residue number" in warning
        and "B:UNL1, B:UNL2" in warning
        for warning in structure.warnings
    )


def test_one_atom_written_twice_is_not_made_into_two(tmp_path):
    """Two like-named atoms on one spot are one atom written twice -- an
    unlabelled alternate at full occupancy -- and naming them apart would
    build both."""

    text = EDITOR.replace(
        "HETATM    5  C   UNL B   1       9.520   0.000   0.000",
        "HETATM    5  C   UNL B   1       8.100   0.000   0.000",
    )
    structure = read_pdb(written(text, tmp_path / "doubled.pdb"))

    ligand = next(r for r in structure.residues if r.id.display() == "B:UNL1")
    assert ligand.atom_names().count("C") == 2


def test_pdbfixer_is_given_the_names_mdprep_reads(tmp_path):
    """PDBFixer keeps one of several atoms sharing a name. The file it reads
    carries the names mdprep gave them, not the editor's."""

    prepared = tmp_path / "repair_input.pdb"

    _write_repair_input(
        written(EDITOR, tmp_path / "editor.pdb"),
        prepared,
        chains=[],
        altloc_policy="highest_occupancy",
    )

    repaired = read_pdb(prepared)
    names = {residue.id.display(): residue.atom_names() for residue in repaired.residues}
    assert names["B:UNL1"] == ["C1", "C2", "O", "H1", "H2"]
    assert names["B:UNL2"] == ["O", "H1", "H2"]


# --------------------------------------------------------------------------
# Residues named apart, and alike
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deposition() -> PdbStructure:
    return read_pdb(DEPOSITION)


def with_a_loose_cofactor(deposition: PdbStructure, resname: str, swap: bool = False):
    """2FCT with a third alpha-ketoglutarate in chain A, 30 A from the protein.

    ``swap`` exchanges its two carboxylate carbons' places: the same names,
    bonded to other neighbours.
    """

    source = [a for a in deposition.atoms if (a.chain_id, a.resid) == ("A", 897)]
    places = {a.name: (a.x, a.y, a.z) for a in source}
    if swap:
        places["C1"], places["C5"] = places["C5"], places["C1"]
    loose = [
        replace(
            a,
            resname=resname,
            resid=950,
            serial=None,
            x=places[a.name][0] + 30.0,
            y=places[a.name][1],
            z=places[a.name][2],
            original_line="",
        )
        for a in source
    ]
    return with_atoms(deposition, [*deposition.atoms, *loose])


ANSWERS = {
    "structure.chains": "all",
    "structure.remove_unknown_heterogens": True,
    "structure.repair.add_missing_heavy_atoms": True,
    "ligand.a_dsu_896.include": False,
    "ligand.a_akg_897.net_charge": -2,
    "ligand.a_akg_897.smiles": AKG_SMILES,
    "ligand.a_akg_897.charge_method": "am1bcc",
    "metal.a_fe2_900_site.model": "nonbonded",
}


def planned(structure: PdbStructure, tmp_path: Path):
    path = tmp_path / "loose.pdb"
    write_pdb(structure, path)
    return plan_decisions(path)


def test_one_molecule_under_two_names_is_built_once_under_one(tmp_path, deposition):
    plan = planned(with_a_loose_cofactor(deposition, "AKX"), tmp_path)
    answers = {
        **ANSWERS,
        "ligand.a_akx_950.net_charge": -2,
        "ligand.a_akx_950.smiles": AKG_SMILES,
        "ligand.a_akx_950.charge_method": "am1bcc",
    }

    data = build_manifest(plan, answers)

    loose = next(block for block in data["ligands"] if block["id"] == "a_akx_950")
    assert loose["parameters_from"] == "a_akg_897"
    assert loose["selector"]["resname"] == "AKG"
    (rename,) = data["structure"]["rename_residues"]
    assert rename["selector"] == {"chain": "A", "resname": "AKX", "resid": 950}
    assert rename["resname"] == "AKG"
    assert "the same molecule as A:AKG897, answered alike" in rename["reason"]
    ManifestConfig.model_validate(data)


def test_one_name_on_two_bondings_is_two_templates(tmp_path, deposition):
    """The same atom names bonded to other neighbours are not one template,
    whatever the SMILES: built as one, the second molecule's atoms would be
    bonded as the first's. It gets a name of its own."""

    plan = planned(with_a_loose_cofactor(deposition, "AKG", swap=True), tmp_path)
    answers = {
        **ANSWERS,
        "ligand.a_akg_950.net_charge": -2,
        "ligand.a_akg_950.smiles": AKG_SMILES,
        "ligand.a_akg_950.charge_method": "am1bcc",
    }

    data = build_manifest(plan, answers)

    loose = next(block for block in data["ligands"] if block["id"] == "a_akg_950")
    assert "parameters_from" not in loose
    assert loose["selector"]["resname"] == "LIG"
    (rename,) = data["structure"]["rename_residues"]
    assert rename["selector"]["resid"] == 950 and rename["resname"] == "LIG"
    ManifestConfig.model_validate(data)


# --------------------------------------------------------------------------
# What preparation does with structure.rename_residues
# --------------------------------------------------------------------------

TWO_COPIES = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
HETATM    4  C1  SUB B 501       5.000   5.000   5.000  1.00 20.00           C
HETATM    5  O1  SUB B 501       6.000   5.000   5.000  1.00 20.00           O
HETATM    6  C1  SU2 D 501      15.000   5.000   5.000  1.00 20.00           C
HETATM    7  O1  SU2 D 501      16.000   5.000   5.000  1.00 20.00           O
HETATM    8  N1  COF C 601       8.000   8.000   8.000  1.00 20.00           N
HETATM    9  C1  COF C 601       9.000   8.000   8.000  1.00 20.00           C
END
"""


def renaming_manifest(path: Path, renames: list[dict]) -> dict:
    data = manifest_data(str(path))
    data["structure"]["remove_unknown_heterogens"] = False
    data["structure"]["rename_residues"] = renames
    data["ligands"] = [
        ligand_entry("b_sub", "B", "SUB", 501),
        {**ligand_entry("d_sub", "D", "SUB", 501), "parameters_from": "b_sub"},
        ligand_entry("c_cof", "C", "COF", 601),
    ]
    return data


SU2_IS_SUB = {
    "selector": {"chain": "D", "resname": "SU2", "resid": 501},
    "resname": "SUB",
    "reason": "the counterpart of B:SUB501 in another copy",
}


def test_preparation_renames_first_and_reports_it(tmp_path):
    data = renaming_manifest(written(TWO_COPIES, tmp_path / "copies.pdb"), [SU2_IS_SUB])

    result = normalize_structure_stage(make_manifest(data))

    # The ligand selectors name the copy as the build does, and resolve.
    assert [ligand.ligand_id for ligand in result.configured_ligands_kept] == [
        "b_sub",
        "d_sub",
        "c_cof",
    ]
    assert [r.id.display() for r in result.normalized_structure.residues] == [
        "A:ALA1",
        "B:SUB501",
        "D:SUB501",
        "C:COF601",
    ]
    (record,) = result.to_report_dict()["residue_renames"]
    assert record["display"] == "D:SU2501"
    assert record["new_resname"] == "SUB"
    assert record["reason"] == "the counterpart of B:SUB501 in another copy"


def test_a_rename_onto_another_molecule_is_refused(tmp_path):
    structure = read_pdb(written(TWO_COPIES, tmp_path / "copies.pdb"))
    onto_cofactor = ManifestConfig.model_validate(
        renaming_manifest(
            tmp_path / "copies.pdb",
            [{**SU2_IS_SUB, "resname": "COF", "reason": "a mistake"}],
        )
    ).structure.rename_residues

    with pytest.raises(ResidueRenameError, match="D:COF501 and C:COF601 under one name|not the same atoms"):
        apply_residue_renames(structure, onto_cofactor)


def test_a_rename_of_a_residue_the_input_lacks_stops_preparation(tmp_path):
    data = renaming_manifest(
        written(TWO_COPIES, tmp_path / "copies.pdb"),
        [{**SU2_IS_SUB, "selector": {"chain": "D", "resname": "SU2", "resid": 777}}],
    )

    with pytest.raises(StructureNormalizationError, match="not in the input: D:SU2777"):
        normalize_structure_stage(make_manifest(data))


@pytest.mark.parametrize(
    "rename, message",
    [
        ({**SU2_IS_SUB, "resname": "SU2"}, "to the name it already has"),
        ({**SU2_IS_SUB, "resname": "SUBS"}, "String should match pattern"),
        ({**SU2_IS_SUB, "reason": ""}, "at least 1 character"),
    ],
)
def test_a_rename_says_what_and_why(tmp_path, rename, message):
    data = renaming_manifest(tmp_path / "copies.pdb", [rename])

    with pytest.raises(ValueError, match=message):
        ManifestConfig.model_validate(data)


def test_a_residue_renamed_twice_is_refused(tmp_path):
    data = renaming_manifest(tmp_path / "copies.pdb", [SU2_IS_SUB, SU2_IS_SUB])

    with pytest.raises(ValueError, match="names a residue more than once: D:SU2501"):
        ManifestConfig.model_validate(data)


def test_a_ligand_cannot_be_renamed_to_a_force_field_name(tmp_path):
    structure = read_pdb(written(TWO_COPIES, tmp_path / "copies.pdb"))
    to_lysine = ManifestConfig.model_validate(
        renaming_manifest(
            tmp_path / "copies.pdb",
            [{**SU2_IS_SUB, "resname": "LYS"}],
        )
    ).structure.rename_residues

    with pytest.raises(ResidueRenameError, match="force fields already define that name"):
        apply_residue_renames(structure, to_lysine)


def test_a_cofactor_in_pieces_is_not_split_for_its_hydrogens_names(tmp_path):
    """A disordered gap leaves a cofactor in two pieces, and a converter
    names every hydrogen H. No heavy atom of one piece is named like one of
    the other, so it is one residue still, its hydrogens named apart."""

    text = """\
HETATM    1  C1  COF B   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  C2  COF B   1       1.520   0.000   0.000  1.00  0.00           C
HETATM    3  H   COF B   1      -0.400   1.000   0.000  1.00  0.00           H
HETATM    4  C5  COF B   1       6.000   0.000   0.000  1.00  0.00           C
HETATM    5  C6  COF B   1       7.520   0.000   0.000  1.00  0.00           C
HETATM    6  H   COF B   1       7.920   1.000   0.000  1.00  0.00           H
END
"""
    structure = read_pdb(written(text, tmp_path / "gapped.pdb"))

    (residue,) = structure.residues
    assert residue.atom_names() == ["C1", "C2", "H1", "C5", "C6", "H2"]
    assert not any("separate molecules" in warning for warning in structure.warnings)
