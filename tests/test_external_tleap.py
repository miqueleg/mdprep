import json

import pytest
import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.external.discovery import which_executable


pytestmark = [pytest.mark.external, pytest.mark.tleap]


def test_real_tleap_ligand_only_build_when_available(tmp_path):
    if which_executable("tleap") is None:
        pytest.skip("tleap is required for this external integration test")

    data = {
        "project": {
            "name": "external_tleap",
            "input_structure": "tests/data/ligands/ligand_sub.pdb",
            "output_dir": str(tmp_path / "prepared"),
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
                "charge_method": "user_mol2",
                "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
                "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
                "preserve_atom_names": True,
                "preserve_coordinates": True,
                "allow_atom_renaming": False,
                "allow_coordinate_changes": False,
                "qmmesp": None,
            }
        ],
        "solvation": {
            "enabled": False,
            "box": "truncated_octahedron",
            "buffer_angstrom": 10.0,
            "neutralize": True,
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
    manifest = tmp_path / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "tleap"])

    assert result.exit_code == 0, result.output
    output = tmp_path / "prepared"
    assert (output / "final" / "system.prmtop").exists()
    assert (output / "final" / "system.inpcrd").exists()
    assert (output / "final" / "system.pdb").exists()
    validation = json.loads((output / "reports" / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["errors"] == []
