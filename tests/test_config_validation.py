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


def qmmesp_ligand() -> dict:
    return {
        "id": "SUB_501",
        "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
        "net_charge": 0,
        "multiplicity": 1,
        "atom_types": "gaff2",
        "charge_method": "qmmesp_pyscf",
        "user_mol2": None,
        "qmmesp": {"qm_engine": "pyscf", "method": "HF", "basis": "STO-3G"},
    }


def test_invalid_forcefield_fails():
    data = base_manifest()
    data["protein"]["forcefield"] = "ff99SB"

    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)


def test_pdbfixer_repair_requires_explicit_sequence_for_missing_residues():
    data = base_manifest()
    data["structure"]["repair"] = {
        "backend": "pdbfixer",
        "add_missing_heavy_atoms": True,
        "missing_residues": [
            {"chain": "A", "resname": "GLN", "resid": 54, "icode": None}
        ],
    }

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "requires sequence_source" in str(excinfo.value)


def test_explicit_pdbfixer_repair_configuration_validates():
    data = base_manifest()
    data["structure"]["repair"] = {
        "backend": "pdbfixer",
        "sequence_source": "data/original.pdb",
        "add_missing_heavy_atoms": True,
        "missing_residues": [
            {"chain": "A", "resname": "GLN", "resid": 54, "icode": None}
        ],
        "random_seed": 1234,
        "platform": "Reference",
    }

    manifest = ManifestConfig.model_validate(data)

    assert manifest.structure.repair.backend == "pdbfixer"
    assert manifest.structure.repair.missing_residues[0].resid == 54
    assert manifest.structure.repair.random_seed == 1234


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


def test_ligand_selector_can_use_resname_only():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "SUB",
            "selector": {"resname": "SUB"},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "am1bcc",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]

    manifest = ManifestConfig.model_validate(data)

    assert manifest.ligands[0].selector.chain is None
    assert manifest.ligands[0].selector.resid is None
    assert manifest.ligands[0].selector.resname == "SUB"


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


def test_gas_resp_pyscf_requires_qmmesp_block():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "SUB_501",
            "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "gas_resp_pyscf",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "charge_method: gas_resp_pyscf requires qmmesp" in str(excinfo.value)


def test_qmmesp_rejects_self_ligand_embedding():
    data = base_manifest()
    data["ligands"] = [
        {
            "id": "SUB_501",
            "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "qmmesp_pyscf",
            "user_mol2": None,
            "qmmesp": {
                "qm_engine": "pyscf",
                "method": "HF",
                "basis": "STO-3G",
                "embedding_cutoff_angstrom": 12.0,
                "environment": {"exclude_self_ligand": False},
            },
        }
    ]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "target ligand self-embedding is not allowed" in str(excinfo.value)


def test_qmmesp_defaults_to_full_embedding_and_canonical_resp():
    data = base_manifest()
    data["ligands"] = [qmmesp_ligand()]

    manifest = ManifestConfig.model_validate(data)
    qmmesp = manifest.ligands[0].qmmesp

    assert qmmesp is not None
    assert qmmesp.embedding_cutoff_angstrom is None
    assert qmmesp.grid.type == "merz_kollman"
    assert qmmesp.grid.vdw_scale_factors == [1.4, 1.6, 1.8, 2.0]
    assert qmmesp.grid.exclude_inside_vdw_scale == 1.4
    assert qmmesp.resp_fitting.backend == "ambertools"
    assert qmmesp.resp_fitting.stage_2 is True


def test_qmmesp_rejects_simplified_resp_backend():
    data = base_manifest()
    ligand = qmmesp_ligand()
    ligand["qmmesp"]["resp_fitting"] = {"backend": "native", "stage_2": True}
    data["ligands"] = [ligand]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert "ambertools" in str(excinfo.value)


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"net_charge": 1, "qmmesp": {"scf_charge": 0}}, "must equal ligand net_charge"),
        ({"multiplicity": 2, "qmmesp": {"scf_spin": 0}}, "must equal multiplicity - 1"),
    ],
)
def test_qmmesp_rejects_inconsistent_scf_state(updates, expected):
    data = base_manifest()
    ligand = qmmesp_ligand()
    qmmesp_updates = updates["qmmesp"]
    ligand.update({key: value for key, value in updates.items() if key != "qmmesp"})
    ligand["qmmesp"].update(qmmesp_updates)
    data["ligands"] = [ligand]

    with pytest.raises(ValidationError) as excinfo:
        ManifestConfig.model_validate(data)

    assert expected in str(excinfo.value)
