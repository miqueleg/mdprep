from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig
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
