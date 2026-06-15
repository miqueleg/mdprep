from pathlib import Path

import pytest

from mdprep.ligands.extract import extract_ligand
from mdprep.ligands.pyscf_charges import derive_pyscf_charges
from mdprep.qm.pyscf_runner import pyscf_available
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_ligand_workflow_mocked import qmmesp_block
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


pytestmark = [pytest.mark.external, pytest.mark.pyscf]


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_gas_resp_charge_derivation_tiny_ligand(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["structure"]["remove_unknown_heterogens"] = True
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "gas_resp_pyscf",
            "qmmesp": {
                **qmmesp_block(),
                "basis": "STO-3G",
                "max_cycle": 50,
                "grid": {
                    "type": "connolly",
                    "vdw_scale_factors": [1.6],
                    "points_per_atom_per_shell": 12,
                    "exclude_inside_vdw_scale": 1.1,
                    "max_points": 200,
                },
                "resp_fitting": {
                    "backend": "native",
                    "restraint": "none",
                    "stage_2": False,
                },
            },
        }
    ]
    manifest = make_manifest(data)
    structure = normalize_structure_stage(manifest).normalized_structure
    extracted = extract_ligand(structure, manifest.ligands[0], output_dir=tmp_path)

    result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path=Path("tests/data/ligands/ligand_sub.good.mol2"),
        output_mol2_path=tmp_path / "sub.pyscf.mol2",
        output_dir=tmp_path,
        method_name="gas_resp_pyscf",
    )

    assert result.charged_mol2_path.exists()
    assert result.grid_point_count > len(extracted.atoms)
    assert result.fit_result["charge_sum_final"] == pytest.approx(0.0, abs=1.0e-6)
