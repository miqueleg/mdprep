import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mdprep.ambertools.mol2 import read_mol2
from mdprep.charges.esp_grid import EspGrid
from mdprep.charges.resp_fit import RespFitResult
from mdprep.ligands.extract import extract_ligand
from mdprep.ligands.pyscf_charges import QMMESP_CONFIRMATION, derive_pyscf_charges
from mdprep.qm.point_charges import PointCharge, PointChargeSelection
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_ligand_workflow_mocked import qmmesp_block
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def extracted_qmmesp_ligand(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["structure"]["remove_unknown_heterogens"] = True
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": qmmesp_block(),
        }
    ]
    manifest = make_manifest(data)
    structure = normalize_structure_stage(manifest).normalized_structure
    return extract_ligand(structure, manifest.ligands[0], output_dir=tmp_path)


def embedded_point_charges():
    return PointChargeSelection(
        target_atom_indices=[20, 21],
        point_charges=[
            PointCharge(
                x=10.0,
                y=10.0,
                z=10.0,
                charge=100.0,
                residue_name="ALA",
                residue_number=1,
                atom_name="CA",
                category="protein",
            )
        ],
        total_before_cutoff=1,
        total_after_cutoff=1,
        net_embedding_charge=100.0,
        min_distance=4.0,
        max_distance=4.0,
        categories={"protein": 1},
    )


def test_mm_potential_is_not_added_to_fitted_esp_target(monkeypatch, tmp_path):
    extracted = extracted_qmmesp_ligand(tmp_path)
    point_charges = embedded_point_charges()
    captured = {}

    def fake_grid(**kwargs):
        return EspGrid(
            points=np.asarray([[8.0, 5.0, 5.0], [5.0, 8.0, 5.0], [5.0, 5.0, 8.0]], dtype=float),
            atom_indices=[0, 0, 1],
            shell_scales=[1.4, 1.4, 1.4],
        )

    def fake_run_pyscf_scf(**kwargs):
        captured["mm_charges"] = kwargs["mm_charges"].copy()
        captured["mm_coordinates"] = kwargs["mm_coordinates"].copy()
        return SimpleNamespace(
            mol=object(),
            mf=object(),
            stdout="",
            stderr="",
            warnings=[],
            to_dict=lambda: {"converged": True, "energy_hartree": -1.0},
        )

    def fake_evaluate_ligand_esp(**kwargs):
        return np.asarray([1.0, 2.0, 3.0], dtype=float)

    def fake_fit_resp_charges(**kwargs):
        captured["fit_esp_values"] = kwargs["esp_values"].copy()
        captured["fit_atom_count"] = len(kwargs["atom_coordinates"])
        return RespFitResult(
            charges=np.asarray([0.25, -0.25], dtype=float),
            charge_sum_before_correction=0.0,
            charge_correction_applied=0.0,
            charge_sum_final=0.0,
            rms_error=0.0,
            relative_rms_error=0.0,
            max_error=0.0,
            iterations=1,
            converged=True,
            fitting_mode="mock",
            warnings=[],
        )

    monkeypatch.setattr("mdprep.ligands.pyscf_charges.generate_connolly_grid", fake_grid)
    monkeypatch.setattr("mdprep.ligands.pyscf_charges.run_pyscf_scf", fake_run_pyscf_scf)
    monkeypatch.setattr("mdprep.ligands.pyscf_charges.evaluate_ligand_esp", fake_evaluate_ligand_esp)
    monkeypatch.setattr("mdprep.ligands.pyscf_charges.fit_resp_charges", fake_fit_resp_charges)

    result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path="tests/data/ligands/ligand_sub.good.mol2",
        output_mol2_path=tmp_path / "sub.pyscf.mol2",
        output_dir=tmp_path,
        method_name="qmmesp_pyscf",
        point_charges=point_charges,
    )

    assert captured["mm_charges"] == pytest.approx([100.0])
    assert captured["fit_esp_values"] == pytest.approx([1.0, 2.0, 3.0])
    assert captured["fit_esp_values"] != pytest.approx([101.0, 102.0, 103.0])
    assert captured["fit_atom_count"] == len(extracted.atoms)
    assert result.fit_result["external_mm_potential_included_in_fit"] is False
    assert result.fit_result["confirmation"] == QMMESP_CONFIRMATION


def test_fitted_charge_centers_are_ligand_atoms_only(monkeypatch, tmp_path):
    extracted = extracted_qmmesp_ligand(tmp_path)
    point_charges = embedded_point_charges()

    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.generate_connolly_grid",
        lambda **kwargs: EspGrid(
            points=np.asarray([[8.0, 5.0, 5.0], [5.0, 8.0, 5.0], [5.0, 5.0, 8.0]], dtype=float),
            atom_indices=[0, 0, 1],
            shell_scales=[1.4, 1.4, 1.4],
        ),
    )
    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.run_pyscf_scf",
        lambda **kwargs: SimpleNamespace(
            mol=object(),
            mf=object(),
            stdout="",
            stderr="",
            warnings=[],
            to_dict=lambda: {"converged": True, "energy_hartree": -1.0},
        ),
    )
    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.evaluate_ligand_esp",
        lambda **kwargs: np.asarray([1.0, 2.0, 3.0], dtype=float),
    )
    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.fit_resp_charges",
        lambda **kwargs: RespFitResult(
            charges=np.asarray([0.25, -0.25], dtype=float),
            charge_sum_before_correction=0.0,
            charge_correction_applied=0.0,
            charge_sum_final=0.0,
            rms_error=0.0,
            relative_rms_error=0.0,
            max_error=0.0,
            iterations=1,
            converged=True,
            fitting_mode="mock",
            warnings=[],
        ),
    )

    result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path="tests/data/ligands/ligand_sub.good.mol2",
        output_mol2_path=tmp_path / "sub.pyscf.mol2",
        output_dir=tmp_path,
        method_name="qmmesp_pyscf",
        point_charges=point_charges,
    )

    with result.fitted_charges_csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["atom_name"] for row in rows] == ["C1", "O1"]
    assert len(rows) == len(extracted.atoms)
    assert {row["atom_name"] for row in rows}.isdisjoint({"CA", "O", "N1"})
    assert read_mol2(result.charged_mol2_path).total_charge == pytest.approx(0.0, abs=1.0e-6)

    fit_report = json.loads(Path(result.fit_report_path).read_text(encoding="utf-8"))
    assert fit_report["fitted_charge_center_count"] == len(extracted.atoms)
    assert fit_report["fitted_charge_centers"] == ["C1", "O1"]
    assert fit_report["confirmation"] == QMMESP_CONFIRMATION


def test_gas_resp_path_has_no_environment(monkeypatch, tmp_path):
    extracted = extracted_qmmesp_ligand(tmp_path)
    captured = {}

    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.generate_connolly_grid",
        lambda **kwargs: EspGrid(
            points=np.asarray([[8.0, 5.0, 5.0], [5.0, 8.0, 5.0], [5.0, 5.0, 8.0]], dtype=float),
            atom_indices=[0, 0, 1],
            shell_scales=[1.4, 1.4, 1.4],
        ),
    )

    def fake_run_pyscf_scf(**kwargs):
        captured["mm_charges"] = kwargs["mm_charges"]
        captured["mm_coordinates"] = kwargs["mm_coordinates"]
        return SimpleNamespace(
            mol=object(),
            mf=object(),
            stdout="",
            stderr="",
            warnings=[],
            to_dict=lambda: {"converged": True, "energy_hartree": -1.0},
        )

    monkeypatch.setattr("mdprep.ligands.pyscf_charges.run_pyscf_scf", fake_run_pyscf_scf)
    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.evaluate_ligand_esp",
        lambda **kwargs: np.asarray([1.0, 2.0, 3.0], dtype=float),
    )
    monkeypatch.setattr(
        "mdprep.ligands.pyscf_charges.fit_resp_charges",
        lambda **kwargs: RespFitResult(
            charges=np.asarray([0.25, -0.25], dtype=float),
            charge_sum_before_correction=0.0,
            charge_correction_applied=0.0,
            charge_sum_final=0.0,
            rms_error=0.0,
            relative_rms_error=0.0,
            max_error=0.0,
            iterations=1,
            converged=True,
            fitting_mode="mock",
            warnings=[],
        ),
    )

    result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path="tests/data/ligands/ligand_sub.good.mol2",
        output_mol2_path=tmp_path / "sub.gas.mol2",
        output_dir=tmp_path,
        method_name="gas_resp_pyscf",
        point_charges=None,
    )

    assert captured["mm_charges"] is None
    assert captured["mm_coordinates"] is None
    assert result.embedding_summary is None
    assert result.fit_result["confirmation"] == "Gas-phase ligand ESP fit; no MM point charges were used."
    assert read_mol2(result.charged_mol2_path).total_charge == pytest.approx(0.0, abs=1.0e-6)
