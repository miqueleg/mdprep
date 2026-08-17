import pytest
from pydantic import ValidationError

from mdprep.config.models import ManifestConfig
from tests.test_config_validation import base_manifest


def _ligand(*, multiplicity: int | None = 1) -> dict:
    data = {
        "id": "substrate",
        "selector": {"chain": "B", "resname": "SUB", "resid": 501},
        "net_charge": 0,
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
    }
    if multiplicity is not None:
        data["multiplicity"] = multiplicity
    return data


def _enabled_refinement() -> dict:
    return {
        "enabled": True,
        "qm_components": {"ligands": ["substrate"]},
        "ash": {
            "python_executable": "/opt/ash/bin/python",
            "num_cores": 2,
        },
    }


def test_ligand_refinement_configuration_validates():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = _enabled_refinement()

    manifest = ManifestConfig.model_validate(data)

    assert manifest.refinement.enabled is True
    assert manifest.refinement.active_region_cutoff_angstrom == 4.0
    assert manifest.refinement.active_water_cutoff_angstrom == 8.0
    assert manifest.refinement.include_contact_waters is True
    assert manifest.refinement.movable_atoms == "all_active_region"
    assert manifest.refinement.ash is not None
    assert manifest.refinement.ash.xtb_executable == "xtb"
    assert manifest.refinement.ash.allow_unusual_link_boundaries is False


def test_refinement_accepts_explicit_hydrogen_only_movable_atoms():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = {
        **_enabled_refinement(),
        "movable_atoms": "active_region_hydrogens",
        "post_refinement_protonation": "reuse_initial",
    }

    manifest = ManifestConfig.model_validate(data)

    assert manifest.refinement.movable_atoms == "active_region_hydrogens"
    assert manifest.refinement.post_refinement_protonation == "reuse_initial"


def test_hydrogen_only_refinement_rejects_discarding_relaxed_hydrogens():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = {
        **_enabled_refinement(),
        "movable_atoms": "active_region_hydrogens",
        "post_refinement_protonation": "rerun",
    }

    with pytest.raises(ValidationError, match="optimized hydrogen coordinates"):
        ManifestConfig.model_validate(data)


def test_gxtb_refinement_requires_explicit_mechanical_embedding():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = {
        **_enabled_refinement(),
        "qm_method": "gxtb",
    }

    with pytest.raises(ValidationError, match="embedding: mechanical"):
        ManifestConfig.model_validate(data)

    data["refinement"]["embedding"] = "mechanical"
    manifest = ManifestConfig.model_validate(data)
    assert manifest.refinement.qm_method == "gxtb"
    assert manifest.refinement.embedding == "mechanical"


def test_mace_refinement_requires_pinned_checkpoint_and_license():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = {
        **_enabled_refinement(),
        "qm_method": "mace_polar1",
        "embedding": "mechanical",
    }

    with pytest.raises(ValidationError, match="requires explicit"):
        ManifestConfig.model_validate(data)

    data["refinement"]["mace_polar1"] = {
        "model": "polar-1-m",
        "model_path": "/models/MACEPOLAR1Mmodel",
        "expected_model_sha256": "a" * 64,
        "accept_model_license": False,
    }
    with pytest.raises(ValidationError, match="Academic Software License"):
        ManifestConfig.model_validate(data)

    data["refinement"]["mace_polar1"]["accept_model_license"] = True
    manifest = ManifestConfig.model_validate(data)
    assert manifest.refinement.mace_polar1 is not None
    assert manifest.refinement.mace_polar1.expected_model_sha256 == "a" * 64


def test_refinement_rejects_implicit_ligand_multiplicity():
    data = base_manifest()
    data["ligands"] = [_ligand(multiplicity=None)]
    data["refinement"] = _enabled_refinement()

    with pytest.raises(ValidationError, match="must explicitly set multiplicity"):
        ManifestConfig.model_validate(data)


def test_nonbonded_metal_refinement_requires_coordinating_residues_and_spin():
    data = base_manifest()
    data["metals"] = [
        {
            "id": "zinc",
            "model": "nonbonded",
            "ions": [
                {
                    "selector": {
                        "chain": "Z",
                        "resname": "ZN",
                        "resid": 1,
                        "atom_name": "ZN",
                    },
                    "element": "Zn",
                    "charge": 2,
                }
            ],
            "nonbonded": {"parameter_set": "12_6"},
        }
    ]
    data["refinement"] = {
        "enabled": True,
        "qm_components": {"metal_sites": ["zinc"]},
        "ash": {"python_executable": "/opt/ash/bin/python"},
    }

    with pytest.raises(ValidationError, match="coordinating residues"):
        ManifestConfig.model_validate(data)

    data["refinement"]["qm_components"]["metal_coordinating_residues"] = {
        "zinc": [{"chain": "A", "resname": "ASP", "resid": 10}]
    }
    with pytest.raises(ValidationError, match="must explicitly set multiplicity"):
        ManifestConfig.model_validate(data)

    data["metals"][0]["ions"][0]["multiplicity"] = 1
    assert ManifestConfig.model_validate(data).metals[0].ions[0].multiplicity == 1


def test_multiple_open_shell_components_require_total_multiplicity():
    data = base_manifest()
    data["ligands"] = [
        _ligand(multiplicity=2),
        {
            **_ligand(multiplicity=3),
            "id": "cofactor",
            "selector": {"chain": "C", "resname": "COF", "resid": 601},
        },
    ]
    data["refinement"] = {
        **_enabled_refinement(),
        "qm_components": {"ligands": ["substrate", "cofactor"]},
    }

    with pytest.raises(ValidationError, match="Multiple open-shell"):
        ManifestConfig.model_validate(data)

    data["refinement"]["total_qm_multiplicity"] = 2
    assert ManifestConfig.model_validate(data).refinement.total_qm_multiplicity == 2


def test_refinement_cutoffs_and_water_inclusion_are_fixed_protocol_values():
    data = base_manifest()
    data["ligands"] = [_ligand()]
    data["refinement"] = {
        **_enabled_refinement(),
        "active_region_cutoff_angstrom": 5.0,
    }

    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)

    data["refinement"]["active_region_cutoff_angstrom"] = 4.0
    data["refinement"]["active_water_cutoff_angstrom"] = 6.0
    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)

    data["refinement"]["active_water_cutoff_angstrom"] = 8.0
    data["refinement"]["include_contact_waters"] = False
    with pytest.raises(ValidationError):
        ManifestConfig.model_validate(data)


def test_refinement_requires_retaining_crystal_waters():
    data = base_manifest()
    data["structure"]["keep_crystal_waters"] = False
    data["ligands"] = [_ligand()]
    data["refinement"] = _enabled_refinement()

    with pytest.raises(ValidationError, match="keep_crystal_waters: true"):
        ManifestConfig.model_validate(data)


def test_mcpb_qmmm_geometry_source_requires_selected_refined_metal_site():
    data = base_manifest()
    ion_selector = {
        "chain": "Z",
        "resname": "ZN",
        "resid": 500,
        "atom_name": "ZN",
    }
    data["metals"] = [
        {
            "id": "catalytic_zinc",
            "model": "bonded_mcpb",
            "ions": [
                {
                    "selector": ion_selector,
                    "element": "Zn",
                    "charge": 2,
                    "multiplicity": 1,
                }
            ],
            "mcpb": {
                "workflow": "pyscf",
                "provisional_nonbonded_parameter_set": "12_6",
                "bonds": [
                    {
                        "ion": ion_selector,
                        "coordinator": {
                            "chain": "A",
                            "resname": "HIS",
                            "resid": 2,
                            "atom_name": "NE2",
                        },
                    }
                ],
                "small_model_charge": 2,
                "small_model_spin": 1,
                "large_model_charge": 2,
                "large_model_spin": 1,
                "software_version": "gau",
                "pyscf": {
                    "geometry_source": "qmmm_refinement",
                    "hessian_backend": "xtb",
                    "xtb": {"model": "gfn2", "executable": "xtb"},
                    "method": "B3LYP",
                    "basis": "6-31G*",
                },
            },
        }
    ]

    with pytest.raises(ValidationError, match="requires refinement.enabled"):
        ManifestConfig.model_validate(data)

    data["refinement"] = {
        "enabled": True,
        "qm_components": {"metal_sites": ["catalytic_zinc"]},
        "ash": {"python_executable": "/opt/ash/bin/python"},
    }
    manifest = ManifestConfig.model_validate(data)
    assert manifest.metals[0].mcpb is not None
    assert manifest.metals[0].mcpb.pyscf is not None
    assert manifest.metals[0].mcpb.pyscf.geometry_source == "qmmm_refinement"
