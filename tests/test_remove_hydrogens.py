from mdprep.protonation.apply import apply_protonation_stage
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def run_with_remove_hydrogens(remove: bool):
    data = manifest_data("tests/data/protein_with_hydrogens.pdb")
    data["protonation"]["method"] = "manual_only"
    data["structure"]["remove_input_hydrogens"] = remove
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    return normalized, apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )


def test_hydrogens_are_removed_when_configured():
    _, result = run_with_remove_hydrogens(True)

    assert result.hydrogen_atoms_removed == 2
    assert [atom.name for atom in result.structure.atoms] == ["N", "CA", "C"]


def test_hydrogens_are_preserved_when_configured_false():
    _, result = run_with_remove_hydrogens(False)

    assert result.hydrogen_atoms_removed == 0
    assert [atom.name for atom in result.structure.atoms] == ["N", "H", "CA", "1HD2", "C"]


def test_heavy_atom_order_is_preserved_after_hydrogen_removal():
    normalized, result = run_with_remove_hydrogens(True)
    heavy_before = [atom.name for atom in normalized.normalized_structure.atoms if atom.name in {"N", "CA", "C"}]

    assert [atom.name for atom in result.structure.atoms] == heavy_before


def test_configured_ligand_hydrogens_are_preserved_when_protein_hydrogens_removed(tmp_path):
    pdb = tmp_path / "protein_ligand_h.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  H   ALA A   1       0.000   0.900   0.000  1.00 20.00           H",
                "ATOM      3  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C",
                "HETATM    4  C1  SUB B 501       5.000   5.000   5.000  1.00 20.00           C",
                "HETATM    5  H1  SUB B 501       5.000   5.000   6.090  1.00 20.00           H",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    data = manifest_data(str(pdb))
    data["structure"]["remove_input_hydrogens"] = True
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "user_mol2",
            "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
        }
    ]
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )

    assert result.hydrogen_atoms_removed == 1
    assert [atom.name for atom in result.structure.atoms] == ["N", "CA", "C1", "H1"]
    assert any("Configured ligand residues were excluded" in warning for warning in result.warnings)
