from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig
from mdprep.protonation.histidine_geometry import is_hydrogen_like, parent_heavy_atom
from mdprep.protonation.temporary_hydrogenation import add_temporary_protein_hydrogens
from mdprep.structure.pdb import read_pdb
from mdprep.structure.repair import repair_structure
from tests.test_config_validation import base_manifest


pytestmark = [pytest.mark.external, pytest.mark.openmm]


def test_pdbfixer_temporary_histidine_environment_preserves_original_atoms(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    structure = read_pdb("tests/data/protein_histidine_ring.pdb")

    result = add_temporary_protein_hydrogens(
        structure,
        ph=7.0,
        work_dir=tmp_path / "temporary_hydrogenation",
    )

    assert result.added_hydrogen_count > 0
    assert result.maximum_original_atom_displacement_angstrom <= 0.002
    assert result.final_prepared_pdb_modified is False
    assert result.restored_output_path.is_file()
    observed = {
        atom.atom_identity: (atom.x, atom.y, atom.z)
        for atom in result.structure.atoms
    }
    for atom in structure.atoms:
        assert observed[atom.atom_identity] == pytest.approx((atom.x, atom.y, atom.z), abs=0.002)


def test_pdbfixer_temporary_histidine_environment_is_seed_reproducible(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    structure = read_pdb("tests/data/protein_histidine_ring.pdb")

    first = add_temporary_protein_hydrogens(
        structure,
        ph=7.0,
        random_seed=41,
        work_dir=tmp_path / "first",
    )
    second = add_temporary_protein_hydrogens(
        structure,
        ph=7.0,
        random_seed=41,
        work_dir=tmp_path / "second",
    )

    assert first.random_seed == second.random_seed == 41
    assert [
        (atom.name, atom.x, atom.y, atom.z) for atom in first.structure.atoms
    ] == [
        (atom.name, atom.x, atom.y, atom.z) for atom in second.structure.atoms
    ]


def test_pdbfixer_repairs_7e07_declared_gap_and_heavy_atoms(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    input_path = Path("examples/tutorials/7E07_prepared.pdb")
    sequence_path = Path("examples/tutorials/7E07.pdb")
    if not input_path.is_file() or not sequence_path.is_file():
        pytest.skip("7E07 tutorial structures are unavailable")
    data = base_manifest()
    data["project"]["input_structure"] = str(input_path)
    data["structure"]["repair"] = {
        "backend": "pdbfixer",
        "sequence_source": str(sequence_path),
        "add_missing_heavy_atoms": True,
        "missing_residues": [
            {"chain": "A", "resname": "GLN", "resid": 54, "icode": None}
        ],
        "random_seed": 20260722,
        "platform": "Reference",
    }
    manifest = ManifestConfig.model_validate(data)

    result = repair_structure(
        manifest,
        output_path=tmp_path / "repaired.pdb",
    )

    assert result is not None
    assert result.added_atom_count == 23
    assert result.missing_residues_detected[0]["resid"] == 54
    assert len(result.missing_heavy_atoms_detected) == 5
    assert len(result.peptide_bond_checks) == 2
    assert all(item["ok"] for item in result.peptide_bond_checks)
    assert len(
        [
            residue
            for residue in result.structure.residues
            if residue.id.resname == "HOH"
        ]
    ) == 199
    terminal = next(
        residue
        for residue in result.structure.residues
        if residue.id.chain_id == "A" and residue.id.resid == 266
    )
    assert terminal.atoms[-1].name == "OXT"
    assert terminal.atoms[-1].serial in result.structure.ter_after_serials
    assert not any(
        atom.serial in result.structure.ter_after_serials
        for atom in terminal.atoms[:-1]
    )


FREE_LYSINE_PDB = "tests/data/protein_with_hydrogenated_free_lysine.pdb"


def _hydrogen_naming_schemes(residue) -> dict[str, set[str]]:
    """The naming schemes each heavy atom's hydrogens are spelled in.

    Amber and OpenMM number the hydrogens on one parent -- CB carries HB2 and
    HB3 -- while mdprep's SMILES hydrogenation letters them, giving HB and HBA.
    Either is fine on its own. Both at one heavy atom means that atom was
    hydrogenated twice, once by each.
    """

    schemes: dict[str, set[str]] = {}
    for atom in residue.atoms:
        if not is_hydrogen_like(atom):
            continue
        parent = parent_heavy_atom(atom, residue)
        if parent is None:
            continue
        # HB2 and HB3 are CB's hydrogens numbered; HB and HBA are the same
        # two lettered.  What is left after the parent's own suffix says which.
        suffix = atom.name.strip()[1:].upper()
        parent_suffix = parent.name.strip()[1:].upper()
        if not suffix.startswith(parent_suffix):
            continue
        tail = suffix[len(parent_suffix):]
        if tail.isdigit():
            scheme = "numbered"
        elif tail.isalpha():
            scheme = "lettered"
        else:
            scheme = "plain"
        schemes.setdefault(parent.name.strip(), set()).add(scheme)
    return schemes


def test_temporary_hydrogenation_leaves_an_already_hydrogenated_residue_alone(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    structure = read_pdb(FREE_LYSINE_PDB)
    lysine = next(
        residue for residue in structure.residues if residue.id.resid == 301
    )
    before = [atom.name for atom in lysine.atoms]

    result = add_temporary_protein_hydrogens(
        structure,
        ph=7.0,
        work_dir=tmp_path / "temporary_hydrogenation",
    )

    assert result.preserved_residues == ("A:LIG301",)
    rebuilt = next(
        residue for residue in result.structure.residues if residue.id.resid == 301
    )
    # The substrate's hydrogens came from the manifest SMILES; a second set in
    # OpenMM's own names would make the cluster a lysine with 27 hydrogens.
    assert [atom.name for atom in rebuilt.atoms] == before
    # The protein around it is still hydrogenated, which is what the step is for.
    histidine = next(
        residue for residue in result.structure.residues if residue.id.resid == 2
    )
    assert "HB2" in {atom.name for atom in histidine.atoms}


def test_temporary_hydrogenation_gives_no_residue_two_hydrogen_naming_schemes(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    structure = read_pdb(FREE_LYSINE_PDB)

    result = add_temporary_protein_hydrogens(
        structure,
        ph=7.0,
        work_dir=tmp_path / "temporary_hydrogenation",
    )

    doubled = {
        residue.id.display(): {
            position: sorted(schemes)
            for position, schemes in _hydrogen_naming_schemes(residue).items()
            if {"numbered", "lettered"} <= schemes
        }
        for residue in result.structure.residues
    }
    assert not any(doubled.values()), doubled
