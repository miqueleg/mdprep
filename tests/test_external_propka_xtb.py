import json

import pytest
import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.external.discovery import which_executable
from mdprep.structure.pdb import read_pdb


pytestmark = pytest.mark.external


def manifest_data(input_structure: str, output_dir: str) -> dict:
    return {
        "project": {
            "name": "external_propka_xtb",
            "input_structure": input_structure,
            "output_dir": output_dir,
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
                    "mode": "sp",
                    "opt_level": "loose",
                    "solvent": None,
                    "cutoff_angstrom": 5.0,
                    "extra_args": [],
                    "energy_close_call_kcal_mol": 0.5,
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


def test_real_propka_xtb_his_workflow_runs_when_tools_are_available(tmp_path):
    if which_executable("propka3") is None or which_executable("xtb") is None:
        pytest.skip("propka3 and xtb are required for this external integration test")

    data = manifest_data("tests/data/protein_histidine_ring.pdb", str(tmp_path / "prepared"))
    manifest = tmp_path / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "protonation"],
    )

    assert result.exit_code == 0, result.output
    report_path = tmp_path / "prepared" / "reports" / "protonation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["propka"]["executable"].endswith("propka3")
    assert report["parsed_pkas"]
    assert report["xtb_histidines"]
    assert report["xtb_histidines"][0]["selected_state"] in {"HID", "HIE"}

    assigned = read_pdb(tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb")
    residue_names = [residue.id.resname for residue in assigned.residues]
    assert report["xtb_histidines"][0]["selected_state"] in residue_names
    assert (tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "energies.json").exists()
