"""Crystal waters sitting on another atom, left out by the builders.

A water in a predicted model, or in a dimer assembled by hand, was placed for
other coordinates: a user's AlphaFold dimer carried 85 waters closer than
2.2 A to a protein atom or to each other, some 0.4 A from a side chain's,
and the refinement then could not place a lysine's hydrogens. A
high-resolution crystal has a few too, for the side-chain conformer the
build does not use. Each is left out, with the atom it overlaps, unless what
it overlaps is itself left out; a water on a metal never is.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mdprep.config.decisions import build_manifest, plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.structure.models import PdbStructure, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb

DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
AKG_SMILES = "[O-]C(=O)CCC(=O)C(=O)[O-]"
ANSWERS = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": True,
    "structure.repair.add_missing_heavy_atoms": True,
    "ligand.a_dsu_896.include": False,
    "ligand.a_akg_897.net_charge": -2,
    "ligand.a_akg_897.smiles": AKG_SMILES,
    "ligand.a_akg_897.charge_method": "am1bcc",
    "metal.a_fe2_900_site.model": "nonbonded",
}
# A crystal water of chain A 2.14 A from Lys174's carbonyl oxygen: shorter
# than any hydrogen bond, and left out of 2FCT as deposited.
DEPOSITED = ("A", 1034)


@pytest.fixture(scope="module")
def deposition() -> PdbStructure:
    return read_pdb(DEPOSITION)


def with_waters_at(structure: PdbStructure, places: dict[tuple[str, int], tuple[float, float, float]]):
    atoms = [
        replace(atom, x=places[key][0], y=places[key][1], z=places[key][2], original_line="")
        if (key := (atom.chain_id, atom.resid)) in places and atom.resname == "HOH"
        else atom
        for atom in structure.atoms
    ]
    return PdbStructure(path=structure.path, atoms=atoms, residues=build_residues(atoms), model_count=1)


def atom_of(structure: PdbStructure, chain: str, resid: int, name: str):
    return next(a for a in structure.atoms if (a.chain_id, a.resid, a.name) == (chain, resid, name))


def removed_waters(structure: PdbStructure, tmp_path: Path, answers=ANSWERS) -> dict[int, str]:
    path = tmp_path / "waters.pdb"
    write_pdb(structure, path)
    data = build_manifest(plan_decisions(path), answers)
    ManifestConfig.model_validate(data)
    return {
        item["selector"]["resid"]: item["reason"]
        for item in data["structure"].get("remove_residues", [])
        if item["selector"]["resname"] == "HOH"
    }


def test_a_water_inside_a_side_chain_is_left_out(tmp_path, deposition):
    nz = atom_of(deposition, "A", 174, "NZ")
    structure = with_waters_at(deposition, {("A", 904): (nz.x + 0.4, nz.y, nz.z)})

    removed = removed_waters(structure, tmp_path)

    assert removed[904].startswith("crystal water overlapping A:LYS174@NZ (0.40 A away)")
    assert removed[DEPOSITED[1]].startswith("crystal water overlapping A:LYS174@O (2.14 A away)")


def test_of_two_waters_on_one_spot_the_second_goes(tmp_path, deposition):
    first = atom_of(deposition, "A", 905, "O")
    structure = with_waters_at(deposition, {("A", 906): (first.x + 0.3, first.y, first.z)})

    removed = removed_waters(structure, tmp_path)

    assert 905 not in removed
    assert removed[906].startswith("crystal water overlapping A:HOH905@O (0.30 A away)")


def test_a_water_on_the_iron_is_never_left_out(tmp_path, deposition):
    """A:HOH1072 is the iron's water. Put a protein atom's worth of crowding
    beside it and it is still part of the site."""

    water = atom_of(deposition, "A", 1072, "O")
    iron = atom_of(deposition, "A", 900, "FE")
    # One angstrom further out along the iron's bond: crowding the water,
    # not bound to the iron itself.
    away = [w - f for w, f in zip((water.x, water.y, water.z), (iron.x, iron.y, iron.z))]
    norm = sum(v * v for v in away) ** 0.5
    place = tuple(w + v / norm for w, v in zip((water.x, water.y, water.z), away))
    structure = with_waters_at(deposition, {("A", 907): place})

    removed = removed_waters(structure, tmp_path)

    assert 1072 not in removed
    assert removed[907].startswith("crystal water overlapping A:HOH1072@O (1.00 A away)")


def test_a_water_overlapping_only_what_is_left_out_stays(tmp_path, deposition):
    """The detergent is declined and removed, so a water in its place is in
    nobody's way."""

    detergent = atom_of(deposition, "A", 896, "O1N")
    structure = with_waters_at(
        deposition, {("A", 908): (detergent.x + 0.5, detergent.y, detergent.z)}
    )

    assert 908 not in removed_waters(structure, tmp_path)


def test_without_the_crystal_waters_nothing_is_said(tmp_path, deposition):
    removed = removed_waters(
        deposition, tmp_path, {**ANSWERS, "structure.keep_crystal_waters": False}
    )

    assert removed == {}


# --------------------------------------------------------------------------
# Atoms structure repair builds
# --------------------------------------------------------------------------
#
# The builders look before PDBFixer has built anything. In 2FCT as
# deposited, Lys A3's side chain beyond CB is unresolved, and a crystal
# water was modelled where PDBFixer then puts its NZ. Normalisation checks
# the kept waters against what repair built.

# Declining the aKG as well spares these tests RDKit: nothing here needs a
# ligand's hydrogens.
WITHOUT_LIGANDS = {
    key: value for key, value in ANSWERS.items() if not key.startswith("ligand.a_akg_897")
} | {"ligand.a_akg_897.include": False}


def heavy_atom(name, resname, resid, x, element="C"):
    from mdprep.structure.models import AtomRecord

    return AtomRecord(
        serial=None, name=name, altloc=None, resname=resname, chain_id="A", resid=resid,
        icode=None, x=x, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0, element=element,
        record_name="ATOM", original_line="",
    )


def structure_of(atoms):
    return PdbStructure(path=Path("x.pdb"), atoms=atoms, residues=build_residues(atoms), model_count=1)


def test_the_atoms_repair_built_are_the_ones_no_input_atom_stands_for():
    """By number, name and place: a replacement keeps the atoms it shares
    with the modified residue, and an atom PDBFixer moved is built anew."""

    from mdprep.structure.repair import _built_heavy_atoms

    original = structure_of(
        [
            heavy_atom("CA", "MSE", 1, 0.0),
            heavy_atom("CE", "MSE", 1, 1.0),
            heavy_atom("SE", "MSE", 1, 2.0, "SE"),
            heavy_atom("CB", "LYS", 2, 5.0),
            heavy_atom("CD", "GLU", 3, 10.0),
        ]
    )
    repaired = structure_of(
        [
            heavy_atom("CA", "MET", 1, 0.0),
            heavy_atom("CE", "MET", 1, 1.005),
            heavy_atom("SD", "MET", 1, 2.5, "S"),
            heavy_atom("CB", "LYS", 2, 5.0),
            heavy_atom("CG", "LYS", 2, 6.0),
            heavy_atom("HG2", "LYS", 2, 6.5, "H"),
            heavy_atom("CD", "GLU", 3, 11.0),
        ]
    )

    built = _built_heavy_atoms(original, repaired)

    assert [(atom.resid, atom.name) for atom in built] == [(1, "SD"), (2, "CG"), (3, "CD")]


def near(atom, distance):
    return (atom.x + distance, atom.y, atom.z)


def normalized_with_built_atoms(tmp_path, monkeypatch, deposition, built, answers=WITHOUT_LIGANDS, **repair):
    """Normalisation of 2FCT with a stand-in for PDBFixer that built ``built``."""

    from mdprep.structure.normalize import normalize_structure_stage
    from mdprep.structure.repair import StructureRepairResult

    def fake_repair(manifest, output_path):
        return StructureRepairResult(
            backend="pdbfixer", backend_version="test", input_path=DEPOSITION,
            sequence_source_path=None, raw_output_path=Path(output_path),
            output_path=Path(output_path), structure=deposition,
            missing_residues_detected=(), missing_heavy_atoms_detected=(),
            added_atom_count=len(built), peptide_bond_checks=(), random_seed=0,
            platform="Reference", warnings=(), built_heavy_atoms=tuple(built),
        )

    monkeypatch.setattr("mdprep.structure.normalize.repair_structure", fake_repair)
    data = build_manifest(plan_decisions(DEPOSITION), answers)
    data["structure"]["repair"].update(repair)
    return normalize_structure_stage(
        ManifestConfig.model_validate(data), output_path=tmp_path / "normalized.pdb"
    )


def built_at(place, name="NZ", resname="LYS", resid=3):
    from dataclasses import replace as replaced

    return replaced(heavy_atom(name, resname, resid, place[0], "N"), y=place[1], z=place[2])


def water_numbers(items):
    return {item["resid"] for item in items if item["resname"] == "HOH"}


def test_a_water_on_an_atom_repair_built_is_left_out_with_the_atom(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", 905, "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.0))]
    )

    [removed] = [item for item in result.waters_removed if item["resid"] == 905]
    assert removed["overlaps"] == "A:LYS3@NZ"
    assert removed["distance_angstrom"] == 1.0
    assert removed["reason"].startswith(
        "crystal water overlapping A:LYS3@NZ (1.00 A away), an atom structure repair built"
    )
    assert 905 not in water_numbers(result.waters_kept)
    assert not any(atom.resid == 905 and atom.resname == "HOH" for atom in result.normalized_structure.atoms)
    assert any(
        warning.startswith("1 crystal water left out for sitting on an atom structure repair built")
        and "A:HOH905 (A:LYS3@NZ, 1.00 A)" in warning
        for warning in result.warnings
    )


def test_a_water_two_point_two_angstrom_away_stays(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", 905, "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 2.25))]
    )

    assert 905 in water_numbers(result.waters_kept)
    assert result.waters_removed == []


def test_a_metals_water_on_a_built_atom_is_kept_and_said(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", 1072, "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.5))]
    )

    assert 1072 in water_numbers(result.waters_kept)
    assert result.waters_removed == []
    assert any(
        warning.startswith("A:HOH1072 is a metal's water, kept as part of the site although "
                           "overlapping A:LYS3@NZ (1.50 A away)")
        for warning in result.warnings
    )


def test_a_water_the_manifest_already_removes_is_not_reported_again(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", DEPOSITED[1], "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.0))]
    )

    assert DEPOSITED[1] in water_numbers(result.residues_removed)
    assert result.waters_removed == []


def test_the_manifest_can_keep_a_water_on_a_built_atom(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", 905, "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.0))],
        remove_waters_on_built_atoms=False,
    )

    assert 905 in water_numbers(result.waters_kept)
    assert result.waters_removed == []
    assert any(
        warning.startswith("A:HOH905 is kept, as structure.repair.remove_waters_on_built_atoms: "
                           "false asks, although overlapping A:LYS3@NZ (1.00 A away)")
        for warning in result.warnings
    )


def test_without_the_crystal_waters_a_built_atom_says_nothing(tmp_path, monkeypatch, deposition):
    water = atom_of(deposition, "A", 905, "O")

    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.0))],
        answers={**WITHOUT_LIGANDS, "structure.keep_crystal_waters": False},
    )

    assert 905 in water_numbers(result.waters_removed)
    assert not any("overlaps" in item for item in result.waters_removed)
    assert not any("structure repair built" in warning for warning in result.warnings)


def test_the_structure_report_gives_each_water_its_reason(tmp_path, monkeypatch, deposition):
    import json

    from mdprep.reports.structure_report import write_structure_reports

    water = atom_of(deposition, "A", 905, "O")
    result = normalized_with_built_atoms(
        tmp_path, monkeypatch, deposition, [built_at(near(water, 1.0))]
    )

    write_structure_reports(
        result, json_path=tmp_path / "structure_report.json", markdown_path=tmp_path / "structure_report.md"
    )

    markdown = (tmp_path / "structure_report.md").read_text(encoding="utf-8")
    assert (
        "  - A:HOH905: crystal water overlapping A:LYS3@NZ (1.00 A away), an atom "
        "structure repair built: closer than any hydrogen bond"
    ) in markdown
    report = json.loads((tmp_path / "structure_report.json").read_text(encoding="utf-8"))
    [removed] = report["waters_removed"]
    assert (removed["resid"], removed["overlaps"], removed["distance_angstrom"]) == (905, "A:LYS3@NZ", 1.0)


@pytest.mark.external
@pytest.mark.openmm
@pytest.mark.parametrize("remove", [True, False])
def test_2fct_waters_on_its_rebuilt_lysine_and_arginine(tmp_path, remove):
    """PDBFixer builds Lys A3's side chain beyond CB onto A:HOH1079, and
    Arg296's onto A:HOH1189."""

    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    from mdprep.structure.normalize import normalize_structure_stage

    data = build_manifest(plan_decisions(DEPOSITION), WITHOUT_LIGANDS)
    data["project"]["output_dir"] = str(tmp_path)
    data["structure"]["repair"]["remove_waters_on_built_atoms"] = remove

    result = normalize_structure_stage(
        ManifestConfig.model_validate(data), output_path=tmp_path / "normalized.pdb"
    )

    kept = water_numbers(result.waters_kept)
    assert DEPOSITED[1] in water_numbers(result.residues_removed)
    if remove:
        assert {
            item["resid"]: (item["overlaps"], item["distance_angstrom"])
            for item in result.waters_removed
        } == {1079: ("A:LYS3@NZ", 1.58), 1189: ("A:ARG296@NH1", 2.0)}
        assert not {1079, 1189} & kept
        assert not {
            atom.resid for atom in result.normalized_structure.atoms if atom.resname == "HOH"
        } & {1079, 1189}
    else:
        assert result.waters_removed == []
        assert {1079, 1189} <= kept
        assert sum("remove_waters_on_built_atoms: false" in warning for warning in result.warnings) == 2
