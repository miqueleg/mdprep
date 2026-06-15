from mdprep.protonation.apply import apply_protonation_stage
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_structure_normalize import make_manifest, manifest_data


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

