import json

import pytest
import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.external.discovery import which_executable
from mdprep.validation.openmm_check import openmm_available, run_openmm_energy_check


pytestmark = [pytest.mark.external, pytest.mark.ambertools, pytest.mark.tleap]


def _manifest(tmp_path, parameter_set: str) -> dict:
    return {
        "project": {
            "name": "external_zinc",
            "input_structure": "tests/data/zinc_only.pdb",
            "output_dir": str(tmp_path / f"prepared_{parameter_set}"),
        },
        "structure": {
            "keep_crystal_waters": True,
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": False,
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
        },
        "protein": {"forcefield": "ff19SB", "water_model": "TIP3P"},
        "protonation": {
            "ph": 7.0,
            "method": "manual_only",
            "overrides": [],
        },
        "disulfides": {
            "auto_detect": False,
            "detection_cutoff_angstrom": 2.2,
            "force": [],
            "forbid": [],
        },
        "ligands": [],
        "metals": [
            {
                "id": "zinc",
                "model": "nonbonded",
                "ions": [
                    {
                        "selector": {
                            "chain": "Z",
                            "resname": "ZN",
                            "resid": 500,
                            "icode": None,
                            "atom_name": "ZN",
                        },
                        "element": "Zn",
                        "charge": 2,
                    }
                ],
                "nonbonded": {"parameter_set": parameter_set},
            }
        ],
        "solvation": {
            "enabled": False,
            "box": "truncated_octahedron",
            "buffer_angstrom": 10.0,
            "neutralize": False,
            "salt_concentration_molar": 0.0,
            "positive_ion": "Na+",
            "negative_ion": "Cl-",
        },
        "validation": {
            "run_openmm_energy_check": False,
            "fail_on_warnings": False,
            "fail_on_missing_parameters": True,
            "fail_on_noninteger_ligand_charge": True,
        },
    }


def test_real_nonbonded_zinc_tleap_build_when_available(tmp_path):
    if which_executable("tleap") is None:
        pytest.skip("tleap is required for this external integration test")
    manifest = tmp_path / "zinc.yaml"
    manifest.write_text(yaml.safe_dump(_manifest(tmp_path, "12_6"), sort_keys=False), encoding="utf-8")
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])
    assert result.exit_code == 0, result.output
    report = json.loads(
        (tmp_path / "prepared_12_6" / "reports" / "metal_report.json").read_text(encoding="utf-8")
    )
    parameter = report["nonbonded_sites"][0]["ions"][0]["parameter"]
    assert parameter["atom_type"] == "Zn2+"
    assert parameter["rmin_over_2_angstrom"] > 0
    assert parameter["epsilon_kcal_mol"] > 0


@pytest.mark.parmed
def test_real_1264_build_contains_c4_flag_when_available(tmp_path):
    if which_executable("tleap") is None or which_executable("parmed") is None:
        pytest.skip("tleap and parmed are required for this external integration test")
    manifest = tmp_path / "zinc_1264.yaml"
    manifest.write_text(
        yaml.safe_dump(_manifest(tmp_path, "12_6_4"), sort_keys=False),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])
    assert result.exit_code == 0, result.output
    topology = tmp_path / "prepared_12_6_4" / "final" / "system.prmtop"
    assert "%FLAG LENNARD_JONES_CCOEF" in topology.read_text(encoding="utf-8")
    tleap_report = json.loads(
        (tmp_path / "prepared_12_6_4" / "reports" / "tleap_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert tleap_report["c4_postprocessing"][0]["nonzero_c4_coefficients"] > 0
    if not openmm_available():
        pytest.skip("OpenMM is required to verify 12-6-4 runtime force construction")
    check = run_openmm_energy_check(
        topology,
        tmp_path / "prepared_12_6_4" / "final" / "system.inpcrd",
    )
    assert check["status"] == "ok", check
    assert check["amber_12_6_4_detected"] is True
    assert check["openmm_12_6_4_force_present"] is True
    assert "CustomNonbondedForce" in check["force_classes"]


def test_real_mcpb_step1_uses_hydrogenated_histidine_when_available(tmp_path):
    if which_executable("tleap") is None or which_executable("MCPB.py") is None:
        pytest.skip("tleap and MCPB.py are required for this external integration test")
    data = _manifest(tmp_path, "12_6")
    data["project"]["name"] = "external_mcpb"
    data["project"]["input_structure"] = "tests/data/protein_histidine_zinc.pdb"
    data["project"]["output_dir"] = str(tmp_path / "prepared_mcpb")
    data["metals"] = [
        {
            "id": "zinc_site",
            "model": "bonded_mcpb",
            "ions": data["metals"][0]["ions"],
            "mcpb": {
                "workflow": "prepare_inputs",
                "provisional_nonbonded_parameter_set": "12_6",
                "bonds": [
                    {
                        "ion": data["metals"][0]["ions"][0]["selector"],
                        "coordinator": {
                            "chain": "A",
                            "resname": "HIS",
                            "resid": 2,
                            "icode": None,
                            "atom_name": "NE2",
                        },
                    }
                ],
                "cutoff_angstrom": 2.3,
                "force_constant_method": "seminario",
                "charge_restraint": "backbone_heavy",
                "software_version": "g16",
                "small_model_charge": 2,
                "small_model_spin": 1,
                "large_model_charge": 2,
                "large_model_spin": 1,
            },
        }
    ]
    manifest = tmp_path / "mcpb.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "metals"])
    assert result.exit_code == 0, result.output
    report = json.loads(
        (tmp_path / "prepared_mcpb" / "reports" / "metal_report.json").read_text(encoding="utf-8")
    )
    assert report["bonded_mcpb_site"]["complete"] is False
    checks = report["pre_mcpb_hydrogenation"]["donor_protonation_checks"]
    assert checks[0]["state"] == "HID"
    assert checks[0]["donor_atom"] == "NE2"
    assert checks[0]["ok"] is True
