from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mdprep.config.models import McpbPySCFConfig
from mdprep.metals.pyscf_mcpb import (
    McpbPySCFError,
    _elements_and_coordinates,
    _run_small_model,
    _validate_user_optimized_small_model,
)
from mdprep.structure.pdb import read_pdb


def _write_model(path: Path, *, second_name: str = "H", shift: float = 0.0) -> Path:
    def line(serial: int, name: str, x: float, element: str) -> str:
        return (
            f"ATOM  {serial:5d} {name:>4s} HID A{1:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {element:>2s}  \n"
        )

    path.write_text(
        line(1, "N", 0.0 + shift, "N")
        + line(2, second_name, 1.0 + shift, "H")
        + "END\n",
        encoding="utf-8",
    )
    return path


def _config(optimized_pdb: Path) -> McpbPySCFConfig:
    return McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(optimized_pdb),
            "small_model_geometry_status": "user_optimized",
            "method": "HF",
            "basis": "STO-3G",
        }
    )


def test_pyscf_mcpb_requires_user_optimized_geometry_path():
    with pytest.raises(ValidationError, match="optimized_small_model_pdb"):
        McpbPySCFConfig.model_validate({"method": "HF", "basis": "STO-3G"})


def test_pyscf_mcpb_rejects_removed_internal_optimizer_controls(tmp_path):
    with pytest.raises(ValidationError, match="optimization_method"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "HF",
                "basis": "STO-3G",
                "optimization_method": "scipy_lbfgsb",
            }
        )


def test_pyscf_mcpb_validates_explicit_scf_convergence_controls(tmp_path):
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
            "small_model_geometry_status": "user_optimized",
            "method": "B3LYP",
            "basis": "6-31G*",
            "scf_algorithm": "newton",
            "initial_guess": "atom",
            "level_shift_hartree": 0.5,
            "damping_factor": 0.2,
            "diis_space": 12,
            "large_model_density_fitting": True,
            "large_model_auxbasis": "def2-universal-jkfit",
        }
    )

    assert config.scf_algorithm == "newton"
    assert config.initial_guess == "atom"
    assert config.level_shift_hartree == pytest.approx(0.5)
    assert config.damping_factor == pytest.approx(0.2)
    assert config.diis_space == 12
    assert config.large_model_density_fitting is True
    assert config.large_model_auxbasis == "def2-universal-jkfit"


def test_pyscf_mcpb_accepts_adiis_then_newton_with_coarse_precondition_grid(
    tmp_path,
):
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
            "small_model_geometry_status": "user_optimized",
            "method": "B3LYP",
            "basis": "6-31G*",
            "scf_algorithm": "adiis_then_newton",
            "adiis_precondition_cycles": 12,
            "adiis_precondition_dft_grid_level": 1,
        }
    )

    assert config.scf_algorithm == "adiis_then_newton"
    assert config.adiis_precondition_cycles == 12
    assert config.adiis_precondition_dft_grid_level == 1


def test_pyscf_mcpb_requires_pinned_restart_checkpoint_and_newton(tmp_path):
    checkpoint = tmp_path / "restart.chk"
    digest = "a" * 64
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
            "small_model_geometry_status": "user_optimized",
            "method": "B3LYP",
            "basis": "6-31G*",
            "scf_algorithm": "newton",
            "large_model_restart_checkpoint": str(checkpoint),
            "large_model_restart_checkpoint_sha256": digest,
            "large_model_restart_max_displacement_angstrom": 0.01,
        }
    )

    assert config.large_model_restart_checkpoint == str(checkpoint)
    assert config.large_model_restart_checkpoint_sha256 == digest
    assert config.large_model_restart_max_displacement_angstrom == pytest.approx(0.01)

    with pytest.raises(ValidationError, match="must be provided together"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "B3LYP",
                "basis": "6-31G*",
                "scf_algorithm": "newton",
                "large_model_restart_checkpoint": str(checkpoint),
            }
        )

    with pytest.raises(ValidationError, match="requires scf_algorithm: newton"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "B3LYP",
                "basis": "6-31G*",
                "large_model_restart_checkpoint": str(checkpoint),
                "large_model_restart_checkpoint_sha256": digest,
            }
        )


def test_pyscf_mcpb_validates_dynamic_level_shift_for_diis(tmp_path):
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
            "small_model_geometry_status": "user_optimized",
            "method": "B3LYP",
            "basis": "6-31G*",
            "scf_algorithm": "diis",
            "level_shift_mode": "dynamic",
            "level_shift_hartree": 0.5,
        }
    )

    assert config.level_shift_mode == "dynamic"


def test_pyscf_mcpb_rejects_dynamic_level_shift_with_newton(tmp_path):
    with pytest.raises(ValidationError, match="available only with scf_algorithm: diis"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "B3LYP",
                "basis": "6-31G*",
                "scf_algorithm": "newton",
                "level_shift_mode": "dynamic",
                "level_shift_hartree": 0.5,
            }
        )


def test_pyscf_mcpb_validates_adiis_then_diis_preconditioner(tmp_path):
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
            "small_model_geometry_status": "user_optimized",
            "method": "B3LYP",
            "basis": "6-31G*",
            "scf_algorithm": "adiis_then_diis",
            "adiis_precondition_cycles": 12,
            "adiis_precondition_dft_grid_level": 1,
            "level_shift_hartree": 0.5,
        }
    )

    assert config.scf_algorithm == "adiis_then_diis"
    assert config.adiis_precondition_cycles == 12
    assert config.adiis_precondition_dft_grid_level == 1


def test_pyscf_mcpb_rejects_adiis_grid_without_preconditioner(tmp_path):
    with pytest.raises(ValidationError, match="requires scf_algorithm"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "B3LYP",
                "basis": "6-31G*",
                "adiis_precondition_dft_grid_level": 1,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("level_shift_hartree", -0.1),
        ("damping_factor", 1.0),
        ("diis_space", 1),
    ],
)
def test_pyscf_mcpb_rejects_invalid_scf_convergence_controls(
    tmp_path, field, value
):
    data = {
        "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
        "small_model_geometry_status": "user_optimized",
        "method": "HF",
        "basis": "STO-3G",
        field: value,
    }
    with pytest.raises(ValidationError):
        McpbPySCFConfig.model_validate(data)


def test_pyscf_mcpb_rejects_auxbasis_without_density_fitting(tmp_path):
    with pytest.raises(ValidationError, match="large_model_density_fitting"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": str(tmp_path / "optimized.pdb"),
                "small_model_geometry_status": "user_optimized",
                "method": "HF",
                "basis": "STO-3G",
                "large_model_auxbasis": "weigend",
            }
        )


def test_user_optimized_geometry_must_preserve_exact_atom_identity(tmp_path):
    generated = read_pdb(_write_model(tmp_path / "generated.pdb"))
    user = read_pdb(
        _write_model(tmp_path / "user.pdb", second_name="HA", shift=0.25)
    )
    generated_elements, _ = _elements_and_coordinates(generated)
    user_elements, _ = _elements_and_coordinates(user)
    with pytest.raises(McpbPySCFError, match="Only coordinates may differ"):
        _validate_user_optimized_small_model(
            generated,
            generated_elements=generated_elements,
            user_structure=user,
            user_elements=user_elements,
        )


def test_small_model_uses_one_fixed_geometry_scf_and_no_optimizer(monkeypatch, tmp_path):
    generated_path = _write_model(tmp_path / "generated.pdb")
    optimized_path = _write_model(tmp_path / "optimized.pdb", shift=0.25)
    calls: list[np.ndarray] = []

    def fake_scf(**kwargs):
        calls.append(np.asarray(kwargs["coordinates"], dtype=float))
        return SimpleNamespace(
            mf=object(),
            energy_hartree=-123.456,
            stdout="fixed geometry SCF\n",
            stderr="",
        )

    monkeypatch.setattr("mdprep.metals.pyscf_mcpb.run_pyscf_scf", fake_scf)
    monkeypatch.setattr(
        "mdprep.metals.pyscf_mcpb.run_pyscf_analytic_hessian",
        lambda mf, *, num_threads: np.eye(6),
    )
    result = _run_small_model(
        generated_path,
        output_fchk=tmp_path / "site_small_opt.fchk",
        charge=0,
        multiplicity=1,
        config=_config(optimized_path),
        user_optimized_pdb=optimized_path,
        work_dir=tmp_path / "qm",
    )

    assert len(calls) == 1
    assert calls[0][:, 0].tolist() == pytest.approx([0.25, 1.25])
    assert result.geometry_optimization_performed is False
    assert result.scf_calculation_count == 1
    assert result.energy_hartree == pytest.approx(-123.456)
    assert result.evaluated_pdb_path.is_file()
    assert result.fchk_path.is_file()
    report = result.calculation_report_path.read_text(encoding="utf-8")
    assert '"geometry_optimization_performed": false' in report
    assert "all nuclei remained fixed" in report


def test_unoptimized_test_geometry_is_labeled_in_result(monkeypatch, tmp_path):
    generated_path = _write_model(tmp_path / "generated.pdb")
    test_path = _write_model(tmp_path / "test_geometry.pdb")
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(test_path),
            "small_model_geometry_status": "test_unoptimized",
            "method": "HF",
            "basis": "STO-3G",
        }
    )
    monkeypatch.setattr(
        "mdprep.metals.pyscf_mcpb.run_pyscf_scf",
        lambda **kwargs: SimpleNamespace(
            mf=object(),
            energy_hartree=-1.0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        "mdprep.metals.pyscf_mcpb.run_pyscf_analytic_hessian",
        lambda mf, *, num_threads: np.eye(6),
    )

    result = _run_small_model(
        generated_path,
        output_fchk=tmp_path / "site_small_opt.fchk",
        charge=0,
        multiplicity=1,
        config=config,
        user_optimized_pdb=test_path,
        work_dir=tmp_path / "qm",
    )

    assert result.geometry_status == "test_unoptimized"
    assert result.scientific_warning is not None
    assert "not production-quality" in result.scientific_warning


def test_small_model_accepts_completed_qmmm_refinement_geometry(monkeypatch, tmp_path):
    generated_path = _write_model(tmp_path / "generated.pdb", shift=0.4)
    refinement_source = _write_model(tmp_path / "full_qmmm_refined.pdb", shift=0.4)
    config = McpbPySCFConfig.model_validate(
        {
            "geometry_source": "qmmm_refinement",
            "method": "HF",
            "basis": "STO-3G",
        }
    )
    calls: list[np.ndarray] = []

    def fake_scf(**kwargs):
        calls.append(np.asarray(kwargs["coordinates"], dtype=float))
        return SimpleNamespace(
            mf=object(), energy_hartree=-5.0, stdout="", stderr=""
        )

    monkeypatch.setattr("mdprep.metals.pyscf_mcpb.run_pyscf_scf", fake_scf)
    monkeypatch.setattr(
        "mdprep.metals.pyscf_mcpb.run_pyscf_analytic_hessian",
        lambda mf, *, num_threads: np.eye(6),
    )

    result = _run_small_model(
        generated_path,
        output_fchk=tmp_path / "site_small_opt.fchk",
        charge=0,
        multiplicity=1,
        config=config,
        user_optimized_pdb=None,
        qmmm_refinement_source_pdb=refinement_source,
        work_dir=tmp_path / "qm",
    )

    assert result.geometry_status == "qmmm_refined"
    assert result.qmmm_refinement_source_pdb_path == refinement_source
    assert result.scientific_warning is None
    assert calls[0][:, 0].tolist() == pytest.approx([0.4, 1.4])
    report = result.to_dict()
    assert report["qmmm_refinement_source_pdb_sha256"] is not None
