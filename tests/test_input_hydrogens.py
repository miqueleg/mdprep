"""Structures that arrive with hydrogens.

mdprep places every hydrogen of a system itself, so the ones a file arrives
with are removed as it is read, before anything looks at the structure. Each
program people clean structures with adds them differently, and each broke a
different stage before this: PyMOL and format converters turn a chloride
into HCl, reduce numbers every hydrogen it adds 0 and keeps the CONECT
records, PDBFixer protonates alpha-ketoglutarate as the neutral acid, and a
riding-hydrogen deposition gives a lysine a third hydrogen the temporary
hydrogenation would not give back. The fixtures in tests/data/input_hydrogens
are those programs' own output for one iron site; see its README.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from mdprep.config.decisions import plan_decisions
from mdprep.config.models import LigandConfig, ManifestConfig
from mdprep.structure.hydrogens import (
    is_hydrogen,
    ligand_keeps_input_hydrogens,
    strip_input_hydrogens,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord, build_residues
from mdprep.structure.normalize import StructureNormalizationError, normalize_structure_stage
from mdprep.structure.pdb import infer_element, read_pdb
from mdprep.structure.repair import _write_repair_input
from tests.test_structure_normalize import manifest_data


DATA = Path("tests/data/input_hydrogens")
CLEAN = DATA / "syrb2_site.pdb"
TOOLS = ("pymol", "pymol_polar", "converter", "reduce", "pdbfixer")
AKG_DIANION = "O=C([O-])CCC(=O)C(=O)[O-]"


def _line(serial: int, name: str, resname: str, resid: int, x: float, element: str = "") -> str:
    record = "ATOM  " if resname in {"ALA", "SER", "LYS"} else "HETATM"
    field = name if len(name) == 4 else f" {name:<3}"
    return (
        f"{record}{serial:5d} {field} {resname:>3} A{resid:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2}"
    )


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines + ["END"]) + "\n", encoding="utf-8")
    return path


def _manifest(path: Path | str, ligands: list[dict] | None = None, **structure) -> ManifestConfig:
    data = manifest_data(str(path))
    data["protonation"] = {"ph": 7.0, "method": "manual_only", "overrides": []}
    data["structure"].update(structure)
    data["ligands"] = ligands or []
    return ManifestConfig.model_validate(data)


def _akg(**extra) -> dict:
    return {
        "id": "akg",
        "selector": {"chain": "A", "resname": "AKG", "resid": 897},
        "net_charge": -2,
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
        "smiles": AKG_DIANION,
        **extra,
    }


def _heavy_atoms(structure: PdbStructure) -> list[tuple]:
    # Sorted: PyMOL and the converter both reorder atoms within a residue.
    return sorted(
        (atom.chain_id, atom.resname, atom.resid, atom.name, atom.x, atom.y, atom.z)
        for atom in structure.atoms
        if not is_hydrogen(atom)
    )


# --------------------------------------------------------------------------
# What a hydrogen is
# --------------------------------------------------------------------------


def test_a_mercury_ion_written_without_an_element_is_not_a_hydrogen(tmp_path):
    """The old name-based test read HG as a hydrogen and would strip it."""
    structure = read_pdb(_write(tmp_path / "hg.pdb", [_line(1, "HG", "HG", 1, 0.0)]))

    assert structure.atoms[0].element == "Hg"
    assert not is_hydrogen(structure.atoms[0])


def test_a_serine_hydrogen_named_hg_is_a_hydrogen(tmp_path):
    structure = read_pdb(
        _write(
            tmp_path / "ser.pdb",
            [_line(1, "OG", "SER", 1, 0.0, "O"), _line(2, "HG", "SER", 1, 0.96, "H")],
        )
    )

    assert [is_hydrogen(atom) for atom in structure.atoms] == [False, True]


def test_deuterium_counts_as_hydrogen(tmp_path):
    structure = read_pdb(
        _write(
            tmp_path / "d.pdb",
            [_line(1, "N", "LYS", 1, 0.0, "N"), _line(2, "D", "LYS", 1, 1.0, "D")],
        )
    )

    assert is_hydrogen(structure.atoms[1])


def test_an_atom_built_without_an_element_goes_by_its_name():
    atom = AtomRecord(
        serial=None, name="1HB", altloc=None, resname="LIG", chain_id="A", resid=1,
        icode=None, x=0.0, y=0.0, z=0.0, occupancy=None, bfactor=None, element=None,
        record_name="HETATM", original_line="",
    )

    assert is_hydrogen(atom)


@pytest.mark.parametrize("name", ["HG21", "HO2'", "HE21", "1HG2"])
def test_four_character_hydrogen_names_without_elements_are_hydrogens(name):
    """A four-character name fills the field, so its start says nothing."""
    assert infer_element(name.lstrip("0123456789"), resname="LIG", record_name="HETATM", atom_field=name) == "H"


def test_a_left_aligned_two_letter_element_is_still_that_element():
    assert infer_element("HG", resname="EMC", record_name="HETATM", atom_field="HG  ") == "Hg"


# --------------------------------------------------------------------------
# Removing them
# --------------------------------------------------------------------------


def test_removal_is_counted_by_kind_and_keeps_the_chain_boundary(tmp_path):
    path = _write(
        tmp_path / "s.pdb",
        [
            _line(1, "N", "ALA", 1, 0.0, "N"),
            _line(2, "H", "ALA", 1, 1.0, "H"),
            "TER",
            _line(3, "O", "HOH", 2, 5.0, "O"),
            _line(4, "H1", "HOH", 2, 5.96, "H"),
            _line(5, "CL", "CL", 3, 9.0, "CL"),
            _line(6, "H04", "CL", 3, 10.3, "H"),
            "CONECT    5    6",
        ],
    )

    stripped, removal = strip_input_hydrogens(read_pdb(path))

    assert [atom.name for atom in stripped.atoms] == ["N", "O", "CL"]
    assert (removal.protein, removal.water, removal.other) == (1, 1, 1)
    assert removal.other_residues == (("A:CL3", 1),)
    # The TER followed the alanine's hydrogen; it now follows its nitrogen.
    assert stripped.ter_after_serials == {1}
    assert stripped.conect_bonds == set()
    assert len(stripped.residues) == 3


def test_a_residue_made_only_of_hydrogens_is_left_whole(tmp_path):
    """Removing every atom would remove the residue, which is not this step's call."""
    path = _write(
        tmp_path / "h.pdb",
        [_line(1, "N", "ALA", 1, 0.0, "N"), _line(2, "H1", "HYD", 2, 4.0, "H")],
    )

    stripped, removal = strip_input_hydrogens(read_pdb(path))

    assert [atom.resname for atom in stripped.atoms] == ["ALA", "HYD"]
    assert removal.removed == 0


def test_residues_kept_on_purpose_are_recorded(tmp_path):
    path = _write(
        tmp_path / "k.pdb",
        [_line(1, "C1", "SUB", 1, 0.0, "C"), _line(2, "H1", "SUB", 1, 1.09, "H")],
    )

    stripped, removal = strip_input_hydrogens(read_pdb(path), keep=lambda residue: True)

    assert len(stripped.atoms) == 2
    assert removal.kept_residues == (("A:SUB1", 1),)


# --------------------------------------------------------------------------
# Which ligands keep theirs
# --------------------------------------------------------------------------


def _ligand(**fields) -> LigandConfig:
    base = {
        "id": "lig",
        "selector": {"chain": "A", "resname": "LIG", "resid": 1},
        "net_charge": 0,
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
    }
    return LigandConfig.model_validate({**base, **fields})


def test_a_smiles_replaces_the_input_hydrogens():
    assert not ligand_keeps_input_hydrogens(_ligand(smiles="C"), remove_input_hydrogens=True)


def test_without_a_smiles_the_input_hydrogens_are_the_protonation():
    assert ligand_keeps_input_hydrogens(_ligand(), remove_input_hydrogens=True)


def test_a_user_mol2_keeps_the_hydrogens_it_lists():
    ligand = _ligand(smiles="C", charge_method="user_mol2", user_mol2="lig.mol2")

    assert ligand_keeps_input_hydrogens(ligand, remove_input_hydrogens=True)


def test_what_the_ligand_says_wins():
    assert ligand_keeps_input_hydrogens(
        _ligand(smiles="C", keep_input_hydrogens=True), remove_input_hydrogens=True
    )
    assert not ligand_keeps_input_hydrogens(
        _ligand(smiles="C", keep_input_hydrogens=False), remove_input_hydrogens=False
    )


def test_nothing_is_removed_when_input_hydrogens_are_kept():
    assert ligand_keeps_input_hydrogens(_ligand(smiles="C"), remove_input_hydrogens=False)


def test_dropping_a_ligands_hydrogens_needs_something_to_place_them():
    with pytest.raises(ValueError, match="requires smiles"):
        _ligand(keep_input_hydrogens=False)


def test_a_user_mol2_cannot_lose_its_hydrogens():
    with pytest.raises(ValueError, match="user_mol2"):
        _ligand(
            smiles="C",
            charge_method="user_mol2",
            user_mol2="lig.mol2",
            keep_input_hydrogens=False,
        )


# --------------------------------------------------------------------------
# Each program's output plans like the deposition
# --------------------------------------------------------------------------


def _decision_signature(plan) -> list[tuple]:
    return [
        (decision.id, decision.kind, decision.default, tuple(c.value for c in decision.choices))
        for decision in plan.decisions
    ]


def _findings(plan) -> dict:
    findings = dict(plan.findings)
    # Where the removal is recorded, and the reader's note on reduce's serials.
    findings.pop("input_hydrogens")
    findings.pop("structure_warnings")
    # PyMOL and the converter reorder a residue's atoms.
    findings["ligand_atom_names"] = {
        key: sorted(names) for key, names in findings["ligand_atom_names"].items()
    }
    return findings


@pytest.mark.parametrize("tool", TOOLS)
def test_a_structure_another_program_protonated_plans_like_the_deposition(tool, tmp_path):
    """Before this, PyMOL's and the converter's files were refused for HCl, reduce's for its CONECT records."""
    # One file name, so the project name proposed from it is the same.
    clean = tmp_path / "clean" / "site.pdb"
    variant = tmp_path / tool / "site.pdb"
    for source, target in ((CLEAN, clean), (DATA / f"syrb2_site.{tool}.pdb", variant)):
        target.parent.mkdir()
        shutil.copy(source, target)

    expected = plan_decisions(clean)
    plan = plan_decisions(variant)

    assert _decision_signature(plan) == _decision_signature(expected)
    assert _findings(plan) == _findings(expected)
    hydrogens = sum(is_hydrogen(atom) for atom in read_pdb(variant).atoms)
    assert plan.findings["input_hydrogens"]["removed"] == hydrogens > 0
    assert not any(is_hydrogen(atom) for atom in plan.structure.atoms)


def test_every_ligand_is_asked_for_its_smiles_and_told_what_it_had():
    plan = plan_decisions(DATA / "syrb2_site.pdbfixer.pdb")

    smiles = plan.by_id("ligand.a_akg_897.smiles")
    assert any("has 6 hydrogens on this residue" in line for line in smiles.evidence)


def test_a_riding_hydrogen_deposition_plans_like_one_without(tmp_path):
    source = (DATA / "riding_peptide.pdb").read_text(encoding="utf-8").splitlines()
    clean = tmp_path / "clean" / "peptide.pdb"
    clean.parent.mkdir()
    clean.write_text(
        "\n".join(line for line in source if not (line.startswith(("ATOM", "HETATM")) and line[76:78].strip() == "H"))
        + "\n",
        encoding="utf-8",
    )
    variant = tmp_path / "riding" / "peptide.pdb"
    variant.parent.mkdir()
    shutil.copy(DATA / "riding_peptide.pdb", variant)

    expected = plan_decisions(clean)
    plan = plan_decisions(variant)

    assert _decision_signature(plan) == _decision_signature(expected)
    assert _findings(plan) == _findings(expected)
    assert plan.findings["input_hydrogens"]["removed"] == 76


# --------------------------------------------------------------------------
# ... and builds like it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_the_structure_stage_removes_what_each_program_added(tool):
    expected = normalize_structure_stage(_manifest(CLEAN))
    result = normalize_structure_stage(_manifest(DATA / f"syrb2_site.{tool}.pdb"))

    assert _heavy_atoms(result.normalized_structure) == _heavy_atoms(expected.normalized_structure)
    assert not any(is_hydrogen(atom) for atom in result.normalized_structure.atoms)
    report = result.to_report_dict()["input_hydrogens"]
    assert report["removed"] == result.input_hydrogens.removed > 0
    assert not any("not applied during" in warning for warning in result.warnings)


def test_input_hydrogens_are_kept_when_the_manifest_says_so():
    result = normalize_structure_stage(
        _manifest(DATA / "syrb2_site.pymol_polar.pdb", remove_input_hydrogens=False)
    )

    assert sum(is_hydrogen(atom) for atom in result.normalized_structure.atoms) == 18
    assert result.input_hydrogens.removed == 0


def test_the_riding_hydrogen_lysine_reaches_protonation_without_them():
    result = normalize_structure_stage(_manifest(DATA / "riding_peptide.pdb"))

    lysine = next(r for r in result.normalized_structure.residues if r.id.resname == "LYS")
    assert [atom.name for atom in lysine.atoms] == ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"]


def test_pdbfixer_gets_the_structure_without_the_input_hydrogens(tmp_path):
    """reduce's repeated serial 0 used to select the wrong lines here."""
    prepared = tmp_path / "repair_input.pdb"

    _write_repair_input(
        DATA / "syrb2_site.reduce.pdb",
        prepared,
        chains=[],
        altloc_policy="highest_occupancy",
        keep_hydrogens=lambda residue: False,
    )

    assert _heavy_atoms(read_pdb(prepared)) == _heavy_atoms(read_pdb(CLEAN))
    assert not any(is_hydrogen(atom) for atom in read_pdb(prepared).atoms)


# --------------------------------------------------------------------------
# A ligand's hydrogens come from its SMILES
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_a_ligands_hydrogens_are_the_smiles_whatever_the_file_had(tool):
    pytest.importorskip("rdkit", reason="RDKit places ligand hydrogens")
    expected = normalize_structure_stage(_manifest(CLEAN, [_akg()]))
    result = normalize_structure_stage(_manifest(DATA / f"syrb2_site.{tool}.pdb", [_akg()]))

    def ligand_hydrogens(structure):
        return sorted(
            (atom.name, atom.x, atom.y, atom.z)
            for atom in structure.atoms
            if atom.resname == "AKG" and is_hydrogen(atom)
        )

    assert ligand_hydrogens(result.normalized_structure) == ligand_hydrogens(
        expected.normalized_structure
    )
    assert len(ligand_hydrogens(result.normalized_structure)) == 4
    assert result.ligand_hydrogens[0]["added_hydrogens"] == 4


def test_kept_hydrogens_from_another_programs_protonation_are_refused():
    """PDBFixer protonates AKG as the neutral acid; the SMILES says dianion."""
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")

    with pytest.raises(StructureNormalizationError, match="O2 has 1 in the input and 0 in the SMILES"):
        normalize_structure_stage(
            _manifest(DATA / "syrb2_site.pdbfixer.pdb", [_akg(keep_input_hydrogens=True)])
        )


def test_kept_hydrogens_that_match_the_smiles_are_used_as_they_are():
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")
    path = DATA / "syrb2_site.pymol.pdb"
    before = [
        (atom.name, atom.x, atom.y, atom.z)
        for atom in read_pdb(path).atoms
        if atom.resname == "AKG" and is_hydrogen(atom)
    ]

    result = normalize_structure_stage(_manifest(path, [_akg(keep_input_hydrogens=True)]))

    after = [
        (atom.name, atom.x, atom.y, atom.z)
        for atom in result.normalized_structure.atoms
        if atom.resname == "AKG" and is_hydrogen(atom)
    ]
    assert after == before
    assert result.ligand_hydrogens[0]["skipped"] is True
    assert result.input_hydrogens.kept_residues == (("A:AKG897", 4),)


# --------------------------------------------------------------------------
# The temporary hydrogenation for xTB
# --------------------------------------------------------------------------


def _residue_structure(names: list[str]) -> PdbStructure:
    atoms = [
        AtomRecord(
            serial=index + 1, name=name, altloc=None, resname="LYS", chain_id="A",
            resid=84, icode=None, x=float(index), y=0.0, z=0.0, occupancy=1.0,
            bfactor=0.0, element="H" if name.startswith("H") else name[0],
            record_name="ATOM", original_line="",
        )
        for index, name in enumerate(names)
    ]
    return PdbStructure(path=Path("x.pdb"), atoms=atoms, residues=build_residues(atoms), model_count=1)


def test_a_hydrogen_the_assigned_state_lacks_may_go():
    """A riding HZ3 on a lysine PropKa made neutral stopped the run."""
    from mdprep.protonation.temporary_hydrogenation import _restore_and_validate

    reference = _residue_structure(["NZ", "HZ1", "HZ2", "HZ3"])
    observed = _residue_structure(["NZ", "HZ1", "HZ2"])

    restored, _ = _restore_and_validate(reference, observed, output_path=Path("out.pdb"))

    assert [atom.name for atom in restored.atoms] == ["NZ", "HZ1", "HZ2"]


def test_a_heavy_atom_going_missing_still_stops_it():
    from mdprep.protonation.temporary_hydrogenation import (
        TemporaryHydrogenationError,
        _restore_and_validate,
    )

    reference = _residue_structure(["CE", "NZ", "HZ1"])
    observed = _residue_structure(["CE", "HZ1"])

    with pytest.raises(TemporaryHydrogenationError, match="NZ"):
        _restore_and_validate(reference, observed, output_path=Path("out.pdb"))
