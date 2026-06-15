import json

import pytest
import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.external.discovery import which_executable


pytestmark = [pytest.mark.external, pytest.mark.ambertools]


def test_real_ambertools_ligand_stage_runs_when_tools_are_available(tmp_path):
    if which_executable("antechamber") is None or which_executable("parmchk2") is None:
        pytest.skip("antechamber and parmchk2 are required for this external integration test")

    data = {
        "project": {
            "name": "external_ambertools",
            "input_structure": "tests/data/protein_two_ligands.pdb",
            "output_dir": str(tmp_path / "prepared"),
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
            "method": "manual_only",
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
        "ligands": [
            {
                "id": "sub_501",
                "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
                "net_charge": 0,
                "multiplicity": 1,
                "atom_types": "gaff2",
                "charge_method": "am1bcc",
                "user_mol2": None,
                "user_frcmod": None,
                "preserve_atom_names": True,
                "preserve_coordinates": True,
                "allow_atom_renaming": False,
                "allow_coordinate_changes": False,
                "qmmesp": None,
            }
        ],
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
    manifest = tmp_path / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "ligands"])

    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "prepared"
    final_mol2 = output_dir / "ligands" / "sub_501" / "parameters" / "sub_501.final.mol2"
    final_frcmod = output_dir / "ligands" / "sub_501" / "parameters" / "sub_501.frcmod"
    assert final_mol2.exists()
    assert final_frcmod.exists()

    report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    ligand = report["ligands"][0]
    assert ligand["status"] == "ok"
    assert ligand["antechamber"]["command"][0].endswith("antechamber")
    assert ligand["parmchk2"]["command"][0].endswith("parmchk2")
