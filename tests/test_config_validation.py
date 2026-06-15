import pytest
from pydantic import ValidationError

from mdprep.config.models import ManifestConfig


def base_manifest() -> dict:
    return {
        "project": {
            "name": "test",
            "input_structure": "data/input.pdb",
            "output_dir": "prepared/test",
        },
        "structure": {
            "keep_crystal_waters": True,
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": False,
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
        },
        "protein": {
            "forcefield": "ff19SB",
            "water_model": "TIP3P",
        },
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


def test_invalid_forcefield_fails():
    data = base_manifest()
    data["protein"]["forcefield"] = "ff99SB"

    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)


def test_invalid_ligand_charge_method_fails():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "BAD_1",
            "selector": {"chain": "A", "resname": "BAD", "resid": 1, "icode": None},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "resp",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]

    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)


def test_gxtb_opt_mode_is_allowed():
    data = base_manifest()
    data["protonation"]["histidine"]["xtb"]["model"] = "gxtb"
    data["protonation"]["histidine"]["xtb"]["mode"] = "opt"

    manifest = ManifestConfig.model_validate(data)

    assert manifest.protonation.histidine.xtb.model == "gxtb"
    assert manifest.protonation.histidine.xtb.mode == "opt"


def test_user_mol2_requires_path():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "USR_1",
            "selector": {"chain": "A", "resname": "USR", "resid": 1, "icode": None},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "user_mol2",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "charge_method: user_mol2 requires user_mol2" in str(excinfo.value)


def test_qmmesp_pyscf_requires_qmmesp_block():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "SUB_501",
            "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
            "net_charge": -1,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "qmmesp_pyscf",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "charge_method: qmmesp_pyscf requires qmmesp" in str(excinfo.value)
