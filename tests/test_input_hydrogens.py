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


def test_a_ligands_hydrogens_survive_the_reading_to_be_compared():
    """Thrown away as the structure is read, they could not be checked against the SMILES."""
    assert ligand_keeps_input_hydrogens(_ligand(smiles="C"))
    assert ligand_keeps_input_hydrogens(_ligand())
    assert ligand_keeps_input_hydrogens(
        _ligand(smiles="C", charge_method="user_mol2", user_mol2="lig.mol2")
    )


def test_a_ligand_can_have_them_replaced_without_comparing():
    assert not ligand_keeps_input_hydrogens(_ligand(smiles="C", keep_input_hydrogens=False))


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
    manifest = _manifest(DATA / "riding_peptide.pdb").model_dump()
    # The fragment has a selenomethionine in its chain, which a manifest has to
    # account for; it used to be dropped from the chain as an unknown heterogen.
    manifest["noncanonical_residues"] = [
        {
            "resname": "MSE",
            "residues": [{"chain": "A", "resname": "MSE", "resid": 83}],
            "parameters": "amber_library",
            "template": "MSE",
        }
    ]
    result = normalize_structure_stage(ManifestConfig.model_validate(manifest))

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


def test_a_ligands_hydrogens_go_round_pdbfixer_rather_than_through_it(tmp_path):
    """PDBFixer keeps one of several atoms sharing a name; the converter names every hydrogen H.

    Six went into the repair and one came out, before the comparison with
    the SMILES ever saw them.
    """
    from mdprep.structure.repair import _return_held_hydrogens

    prepared = tmp_path / "repair_input.pdb"

    held = _write_repair_input(
        DATA / "syrb2_site.converter.pdb",
        prepared,
        chains=[],
        altloc_policy="highest_occupancy",
        keep_hydrogens=lambda residue: residue.id.resname == "AKG",
    )

    assert not any(is_hydrogen(atom) for atom in read_pdb(prepared).atoms)
    assert [atom.name for atom in held[("A", "AKG", 897, None)]] == ["H"] * 6
    returned = _return_held_hydrogens(read_pdb(prepared), held)
    akg = next(residue for residue in returned.residues if residue.id.resname == "AKG")
    assert sum(is_hydrogen(atom) for atom in akg.atoms) == 6
    assert [atom for residue in returned.residues for atom in residue.atoms] == returned.atoms


def test_a_returned_hydrogen_does_not_end_up_beyond_its_chain_boundary(tmp_path):
    from mdprep.structure.repair import _return_held_hydrogens

    path = _write(
        tmp_path / "l.pdb",
        [_line(1, "C1", "SUB", 1, 0.0, "C"), _line(2, "H1", "SUB", 1, 1.09, "H"), "TER"],
    )
    structure = read_pdb(path)
    stripped, _ = strip_input_hydrogens(structure)

    returned = _return_held_hydrogens(stripped, {("A", "SUB", 1, None): [structure.atoms[1]]})

    assert [atom.name for atom in returned.atoms] == ["C1", "H1"]
    assert returned.ter_after_serials == {returned.atoms[-1].serial}


def test_what_the_reader_said_about_the_input_reaches_the_report_after_repair(
    tmp_path, monkeypatch
):
    """PDBFixer's copy no longer has the unlabelled conformers to report."""
    from types import SimpleNamespace

    lines = [
        _line(1, "N", "ALA", 1, 0.0, "N"),
        _line(2, "CA", "ALA", 1, 1.0, "C"),
        _line(3, "CB", "ALA", 1, 2.0, "C").replace("  1.00  0.00", "  0.50  0.00"),
        _line(4, "CB", "ALA", 1, 2.5, "C").replace("  1.00  0.00", "  0.50  0.00"),
    ]
    path = _write(tmp_path / "converter.pdb", lines)
    repaired = read_pdb(_write(tmp_path / "repaired.pdb", lines[:3]))
    monkeypatch.setattr(
        "mdprep.structure.normalize.repair_structure",
        lambda manifest, output_path: SimpleNamespace(structure=repaired, warnings=()),
    )

    result = normalize_structure_stage(
        _manifest(path, repair={"backend": "pdbfixer", "add_missing_heavy_atoms": True}),
        output_path=tmp_path / "normalized.pdb",
    )

    assert any("A:ALA1 CB" in warning for warning in result.warnings)


def test_atoms_pdbfixer_merges_are_reported_not_lost():
    from mdprep.structure.repair import StructureRepairError, _restore_existing_atom_metadata

    def structure(names):
        atoms = [
            AtomRecord(
                serial=index + 1, name=name, altloc=None, resname="SUB", chain_id="A",
                resid=1, icode=None, x=float(index), y=0.0, z=0.0, occupancy=1.0,
                bfactor=0.0, element="C", record_name="HETATM", original_line="",
            )
            for index, name in enumerate(names)
        ]
        return PdbStructure(path=Path("x.pdb"), atoms=atoms, residues=build_residues(atoms), model_count=1)

    with pytest.raises(StructureRepairError, match=r"A:SUB1 C \(2 in, 1 out\)"):
        _restore_existing_atom_metadata(
            structure(["C", "C", "O"]), structure(["C", "O"]), output_path=Path("out.pdb")
        )


# --------------------------------------------------------------------------
# A ligand's hydrogens come from its SMILES
# --------------------------------------------------------------------------


def _hydrogens_per_heavy_atom(structure: PdbStructure, resname: str) -> dict[str, int]:
    atoms = [atom for atom in structure.atoms if atom.resname == resname]
    heavy = [atom for atom in atoms if not is_hydrogen(atom)]
    counts = {atom.name: 0 for atom in heavy}
    for hydrogen in (atom for atom in atoms if is_hydrogen(atom)):
        parent = min(
            heavy,
            key=lambda atom: (atom.x - hydrogen.x) ** 2
            + (atom.y - hydrogen.y) ** 2
            + (atom.z - hydrogen.z) ** 2,
        )
        counts[parent.name] += 1
    return counts


@pytest.mark.parametrize("tool", TOOLS)
def test_a_ligand_ends_up_with_the_smiles_hydrogens_whatever_the_file_had(tool):
    """Kept where they agree with the SMILES, replaced where they do not."""
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")
    expected = normalize_structure_stage(_manifest(CLEAN, [_akg()]))
    source = read_pdb(DATA / f"syrb2_site.{tool}.pdb")
    had = sum(1 for atom in source.atoms if atom.resname == "AKG" and is_hydrogen(atom))

    result = normalize_structure_stage(_manifest(DATA / f"syrb2_site.{tool}.pdb", [_akg()]))

    assert _hydrogens_per_heavy_atom(result.normalized_structure, "AKG") == (
        _hydrogens_per_heavy_atom(expected.normalized_structure, "AKG")
    )
    outcome = result.ligand_hydrogens[0]
    if had == 0:
        assert outcome["added_hydrogens"] == 4
    elif outcome["skipped"]:
        # PyMOL and reduce protonate it as the dianion the SMILES states.
        assert had == 4 and outcome["added_hydrogens"] == 0
    else:
        assert outcome["replaced_hydrogens"] == had
        assert any("LIGAND_HYDROGENS_REPLACED" in warning for warning in result.warnings)


def test_another_programs_protonation_is_replaced_and_the_difference_reported():
    """PDBFixer protonates AKG as the neutral acid; the SMILES says dianion."""
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")

    result = normalize_structure_stage(_manifest(DATA / "syrb2_site.pdbfixer.pdb", [_akg()]))

    outcome = result.ligand_hydrogens[0]
    assert outcome["replaced_hydrogens"] == 6
    assert "O2 has 1 in the input and 0 in the SMILES" in outcome["differences"]
    assert "O4 has 1 in the input and 0 in the SMILES" in outcome["differences"]
    report = result.to_report_dict()
    assert report["ligand_hydrogens"][0]["replaced_hydrogens"] == 6


def test_keeping_another_programs_protonation_on_purpose_is_refused():
    """keep_input_hydrogens: true turns the disagreement into an error."""
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
# A docked ligand: only its polar hydrogens
# --------------------------------------------------------------------------


DOCKED = DATA / "docked_polar_only.pdb"
LIDOCAINE = "CCN(CC)CC(=O)Nc1c(C)cccc1C"


def _docked(**extra) -> dict:
    return {
        "id": "substrate",
        "selector": {"chain": "L", "resname": "LIG", "resid": 1},
        "net_charge": 0,
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
        **extra,
    }


def test_a_docked_ligand_without_a_smiles_is_refused_before_anything_runs():
    """It used to reach antechamber, after protonation, with an odd number of electrons."""
    with pytest.raises(StructureNormalizationError, match="arrives with 1 hydrogen.s. and no SMILES"):
        normalize_structure_stage(_manifest(DOCKED, [_docked()]))


def test_a_docked_ligand_gets_every_hydrogen_its_smiles_states():
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")

    result = normalize_structure_stage(_manifest(DOCKED, [_docked(smiles=LIDOCAINE)]))

    hydrogens = [atom for atom in result.normalized_structure.atoms if is_hydrogen(atom)]
    assert len(hydrogens) == 22
    outcome = result.ligand_hydrogens[0]
    assert outcome["replaced_hydrogens"] == 1
    # The methyls a docking program leaves bare are among the differences.
    assert "C1 has 0 in the input and 3 in the SMILES" in outcome["differences"]
    assert any("LIGAND_HYDROGENS_REPLACED" in warning for warning in result.warnings)
    markdown = _structure_markdown(result)
    assert "replaced its 1 input hydrogen(s)" in markdown


@pytest.mark.parametrize(
    ("path", "ligands"),
    [(DOCKED, [_docked(smiles=LIDOCAINE)]), (DATA / "syrb2_site.pdbfixer.pdb", [_akg()])],
)
def test_the_residues_handed_on_hold_the_atoms_handed_on(path, ligands):
    """The temporary hydrogenation for xTB checks every atom of every residue.

    It found a replaced hydrogen 2.4 A from where a stale residue record still
    had it, on a docked substrate whose two polar hydrogens had been replaced.
    """
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")

    structure = normalize_structure_stage(_manifest(path, ligands)).normalized_structure

    assert [atom for residue in structure.residues for atom in residue.atoms] == structure.atoms


def test_a_ligand_without_hydrogens_or_a_smiles_is_refused_early():
    with pytest.raises(StructureNormalizationError, match="no hydrogens in the input and no SMILES"):
        normalize_structure_stage(_manifest(CLEAN, [{**_akg(), "smiles": None}]))


def test_the_electron_count_has_to_fit_the_multiplicity():
    """antechamber only mentioned this in passing, before its semi-empirical step failed."""
    pytest.importorskip("rdkit", reason="RDKit reads the SMILES")

    with pytest.raises(StructureNormalizationError, match="128 electrons .* multiplicity 2 cannot have"):
        normalize_structure_stage(
            _manifest(DOCKED, [_docked(smiles=LIDOCAINE, multiplicity=2)])
        )


def _structure_markdown(result) -> str:
    from mdprep.reports.structure_report import _render_markdown

    return _render_markdown(result.to_report_dict())


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
