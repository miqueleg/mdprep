import pytest

from mdprep.config.models import ManifestConfig
from mdprep.structure.normalize import StructureNormalizationError, normalize_structure_stage


def manifest_data(input_structure: str) -> dict:
    return {
        "project": {
            "name": "test",
            "input_structure": input_structure,
            "output_dir": "prepared/test",
        },
        "structure": {
            "keep_crystal_waters": True,
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": True,
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
        },
        "protein": {"forcefield": "ff19SB", "water_model": "TIP3P"},
        "protonation": {
            "ph": 7.0,
            "method": "propka_xtb_his",
            "overrides": [],
            "histidine": {
                "neutral_tautomer_method": "xtb",
                "xtb": {
                    "executable": "xtb",
                    "model": "gfn2",
                    "mode": "opt",
                    "opt_level": "loose",
                    "solvent": "water",
                    "cutoff_angstrom": 5.0,
                    "extra_args": [],
                },
            },
        },
        "disulfides": {
            "auto_detect": True,
            "detection_cutoff_angstrom": 2.2,
            "force": [],
            "forbid": [],
        },
        "ligands": [],
        "solvation": {
            "enabled": True,
            "box": "truncated_octahedron",
            "buffer_angstrom": 10.0,
            "neutralize": True,
            "salt_concentration_molar": 0.15,
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
        },
        "validation": {
            "run_openmm_energy_check": True,
            "fail_on_warnings": False,
            "fail_on_missing_parameters": True,
            "fail_on_noninteger_ligand_charge": True,
        },
    }


def ligand_entry(ligand_id: str, chain: str, resname: str, resid: int) -> dict:
    return {
        "id": ligand_id,
        "selector": {"chain": chain, "resname": resname, "resid": resid, "icode": None},
        "net_charge": 0,
        "multiplicity": 1,
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
        "user_mol2": None,
        "qmmesp": None,
    }


def make_manifest(data: dict) -> ManifestConfig:
    return ManifestConfig.model_validate(data)


def test_keep_waters_when_configured():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["structure"]["keep_crystal_waters"] = True
    result = normalize_structure_stage(make_manifest(data))

    assert len(result.waters_kept) == 1
    assert len(result.waters_removed) == 0
    assert "HOH" in [residue.id.resname for residue in result.normalized_structure.residues]


def test_remove_waters_when_configured():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["structure"]["keep_crystal_waters"] = False
    result = normalize_structure_stage(make_manifest(data))

    assert len(result.waters_kept) == 0
    assert len(result.waters_removed) == 1
    assert "HOH" not in [residue.id.resname for residue in result.normalized_structure.residues]


def test_configured_ligand_is_kept():
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        ligand_entry("SUB_501", "B", "SUB", 501),
        ligand_entry("COF_601", "C", "COF", 601),
    ]

    result = normalize_structure_stage(make_manifest(data))

    assert [ligand.ligand_id for ligand in result.configured_ligands_kept] == ["SUB_501", "COF_601"]
    assert [residue.id.resname for residue in result.normalized_structure.residues] == ["ALA", "SUB", "COF"]


def test_unknown_heterogen_fails_when_not_allowed():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["structure"]["remove_unknown_heterogens"] = False

    with pytest.raises(StructureNormalizationError) as excinfo:
        normalize_structure_stage(make_manifest(data))

    assert "Unknown heterogens are present" in str(excinfo.value)
    assert "Add them to ligands:" in str(excinfo.value)


def test_unknown_heterogen_removed_when_allowed():
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["structure"]["remove_unknown_heterogens"] = True
    result = normalize_structure_stage(make_manifest(data))

    assert [entry["resname"] for entry in result.unknown_heterogens_removed] == ["SO4"]
    assert "SO4" not in [residue.id.resname for residue in result.normalized_structure.residues]


def test_missing_ligand_selector_fails_clearly():
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("MISSING", "B", "SUB", 999)]

    with pytest.raises(StructureNormalizationError) as excinfo:
        normalize_structure_stage(make_manifest(data))

    assert "selector did not resolve exactly one residue" in str(excinfo.value)


def test_altloc_policy_is_applied_during_normalization():
    data = manifest_data("tests/data/protein_altloc.pdb")
    manifest = make_manifest(data)
    result = normalize_structure_stage(manifest)
    ca = next(atom for atom in result.normalized_structure.atoms if atom.name == "CA")

    assert ca.altloc == "B"
    assert ca.x == 2.0


def test_forced_disulfide_selector_is_validated():
    data = manifest_data("tests/data/protein_disulfide.pdb")
    data["disulfides"]["force"] = [
        {
            "a": {"chain": "A", "resname": "CYS", "resid": 10, "icode": None},
            "b": {"chain": "A", "resname": "CYS", "resid": 999, "icode": None},
            "reason": "bad selector",
        }
    ]

    with pytest.raises(StructureNormalizationError) as excinfo:
        normalize_structure_stage(make_manifest(data))

    assert "Disulfide force[1].b selector" in str(excinfo.value)
