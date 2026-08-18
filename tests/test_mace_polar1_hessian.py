from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mdprep.config.models import McpbMacePolar1HessianConfig, McpbPySCFConfig
from mdprep.external.runner import CommandResult
from mdprep.metals.mace_polar1_hessian import run_mace_polar1_hessian
from mdprep.metals.mace_polar1_worker import (
    _cartesian_matrix,
    _finite_difference_check,
    _require_supported_mace_version,
)
from mdprep.metals.pyscf_mcpb import _run_small_model


def _write_two_atom_pdb(path: Path) -> Path:
    path.write_text(
        "ATOM      1    N HID A   1       0.000   0.000   0.000  1.00  0.00"
        "           N  \n"
        "ATOM      2    H HID A   1       1.000   0.000   0.000  1.00  0.00"
        "           H  \n"
        "END\n",
        encoding="utf-8",
    )
    return path


def test_mace_polar1_backend_requires_settings_and_license_acceptance():
    base = {
        "optimized_small_model_pdb": "optimized.pdb",
        "small_model_geometry_status": "user_optimized",
        "hessian_backend": "mace_polar1",
        "method": "B3LYP",
        "basis": "6-31G*",
    }
    with pytest.raises(ValidationError, match="requires explicit mace_polar1 settings"):
        McpbPySCFConfig.model_validate(base)

    with pytest.raises(ValidationError, match="Academic Software License"):
        McpbPySCFConfig.model_validate(
            {**base, "mace_polar1": {"python_executable": "/opt/mace/bin/python"}}
        )

    config = McpbPySCFConfig.model_validate(
        {
            **base,
            "mace_polar1": {
                "python_executable": "/opt/mace/bin/python",
                "accept_model_license": True,
            },
        }
    )
    assert config.mace_polar1 is not None
    assert config.mace_polar1.model == "polar-1-m"
    assert config.mace_polar1.device == "cpu"


def test_worker_normalizes_supported_hessian_layouts():
    matrix = np.arange(36, dtype=float).reshape((6, 6))
    atom_coordinate_layout = matrix.reshape((2, 3, 2, 3))
    atom_atom_layout = atom_coordinate_layout.transpose(0, 2, 1, 3)
    flattened_coordinate_layout = matrix.reshape((6, 2, 3))

    assert np.array_equal(_cartesian_matrix(matrix, 2), matrix)
    assert np.array_equal(_cartesian_matrix(flattened_coordinate_layout, 2), matrix)
    assert np.array_equal(_cartesian_matrix(atom_coordinate_layout, 2), matrix)
    assert np.array_equal(_cartesian_matrix(atom_atom_layout, 2), matrix)


def test_worker_finite_difference_check_uses_force_sign_and_cartesian_order():
    hessian = np.diag(np.arange(1.0, 7.0))

    class HarmonicAtoms:
        def __init__(self) -> None:
            self.positions = np.zeros((2, 3), dtype=float)

        def __len__(self) -> int:
            return 2

        def get_positions(self) -> np.ndarray:
            return self.positions.copy()

        def set_positions(self, positions: np.ndarray) -> None:
            self.positions = np.asarray(positions, dtype=float).copy()

        def get_forces(self) -> np.ndarray:
            return (-(hessian @ self.positions.reshape(-1))).reshape((2, 3))

    validation = _finite_difference_check(
        HarmonicAtoms(),
        hessian,
        step_angstrom=0.001,
    )

    assert validation["relative_error"] < 1.0e-12


def test_worker_requires_polar_hessian_capable_mace_version(monkeypatch):
    monkeypatch.setattr(
        "mdprep.metals.mace_polar1_worker._version",
        lambda distribution: "0.3.15",
    )
    with pytest.raises(RuntimeError, match="0.3.16 or newer"):
        _require_supported_mace_version()

    monkeypatch.setattr(
        "mdprep.metals.mace_polar1_worker._version",
        lambda distribution: "0.3.16+cpu",
    )
    assert _require_supported_mace_version() == "0.3.16+cpu"


def test_mace_polar1_runner_records_and_validates_external_worker(
    monkeypatch,
    tmp_path,
):
    python = tmp_path / "python"
    python.write_bytes(b"fake MACE Python")
    python.chmod(0o755)
    model = tmp_path / "MACE-POLAR-1-M.model"
    model.write_bytes(b"reviewed fake checkpoint")
    model_sha = __import__("hashlib").sha256(model.read_bytes()).hexdigest()
    xyz = tmp_path / "small.xyz"
    xyz.write_text("2\nfixed\nN 0 0 0\nH 1 0 0\n", encoding="utf-8")
    matrix = np.eye(6)
    calls: list[tuple[str, ...]] = []

    def fake_run_command(command, *, cwd, env):
        calls.append(tuple(command))
        if "--probe" in command:
            return CommandResult(
                command=tuple(command),
                cwd=str(cwd),
                returncode=0,
                stdout=json.dumps(
                    {
                        "status": "available",
                        "versions": {"mace_torch": "0.3.16"},
                    }
                )
                + "\n",
                stderr="",
                runtime_seconds=0.1,
            )
        output_hessian = Path(cwd) / command[command.index("--output-hessian") + 1]
        output_report = Path(cwd) / command[command.index("--output-report") + 1]
        np.save(output_hessian, matrix)
        report = {
            "status": "complete",
            "backend": "MACE-POLAR-1",
            "model": "polar-1-m",
            "model_path": str(model),
            "model_sha256": model_sha,
            "versions": {
                "python": "3.11 test",
                "mace_torch": "0.3.16",
                "torch": "2.8.0",
                "ase": "3.26.0",
                "graph_longrange": "0.4.0",
            },
            "device": "cpu",
            "default_dtype": "float64",
            "atom_count": 2,
            "molecular_charge": 0,
            "multiplicity": 6,
            "energy_hartree": -10.0,
            "gradient_norm_hartree_per_bohr": 0.2,
            "output_hessian_units": "hartree/bohr^2",
            "geometry_optimization_performed": False,
            "finite_difference_validation": {"relative_error": 1.0e-5},
        }
        output_report.write_text(json.dumps(report), encoding="utf-8")
        assert env["OMP_NUM_THREADS"] == "2"
        assert env["MPLCONFIGDIR"] == str(tmp_path / ".matplotlib")
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=0,
            stdout=json.dumps(report) + "\n",
            stderr="",
            runtime_seconds=3.5,
        )

    monkeypatch.setattr(
        "mdprep.metals.mace_polar1_hessian.run_command",
        fake_run_command,
    )
    result = run_mace_polar1_hessian(
        xyz_path=xyz,
        atom_count=2,
        charge=0,
        multiplicity=6,
        config=McpbMacePolar1HessianConfig(
            python_executable=str(python),
            model="polar-1-m",
            num_threads=2,
            expected_model_sha256=model_sha,
            accept_model_license=True,
        ),
        work_dir=tmp_path,
    )

    assert len(calls) == 2
    assert np.array_equal(result.hessian_hartree_per_bohr2, matrix)
    assert result.energy_hartree == pytest.approx(-10.0)
    assert result.multiplicity == 6
    assert result.model_sha256 == model_sha
    assert result.finite_difference_relative_error == pytest.approx(1.0e-5)
    assert result.command_result.runtime_seconds == pytest.approx(3.5)
    assert result.report_path.is_file()


def test_small_model_can_use_mace_without_small_model_pyscf(monkeypatch, tmp_path):
    generated = _write_two_atom_pdb(tmp_path / "generated.pdb")
    optimized = _write_two_atom_pdb(tmp_path / "optimized.pdb")
    python = tmp_path / "python"
    python.write_bytes(b"fake")
    stdout = tmp_path / "mace.stdout"
    stderr = tmp_path / "mace.stderr"
    stdout.write_text("complete\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    def fail_pyscf(**kwargs):
        raise AssertionError("PySCF small-model SCF must not run for MACE Hessian")

    monkeypatch.setattr("mdprep.metals.pyscf_mcpb.run_pyscf_scf", fail_pyscf)
    monkeypatch.setattr(
        "mdprep.metals.mace_polar1_hessian.run_mace_polar1_hessian",
        lambda **kwargs: SimpleNamespace(
            hessian_hartree_per_bohr2=np.eye(6),
            energy_hartree=-20.0,
            command_result=SimpleNamespace(runtime_seconds=2.0),
            versions={"mace_torch": "0.3.16"},
            model="polar-1-m",
            stdout_path=stdout,
            stderr_path=stderr,
            to_dict=lambda: {
                "probe_command": {
                    "command": ["python", "worker.py", "--probe"],
                    "cwd": str(tmp_path),
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "runtime_seconds": 0.1,
                },
                "hessian_command": {
                    "command": ["python", "worker.py"],
                    "cwd": str(tmp_path),
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "runtime_seconds": 2.0,
                },
            },
        ),
    )
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(optimized),
            "small_model_geometry_status": "test_unoptimized",
            "hessian_backend": "mace_polar1",
            "mace_polar1": {
                "python_executable": str(python),
                "accept_model_license": True,
            },
            "method": "B3LYP",
            "basis": "6-31G*",
        }
    )

    result = _run_small_model(
        generated,
        output_fchk=tmp_path / "site_small_opt.fchk",
        charge=0,
        multiplicity=1,
        config=config,
        user_optimized_pdb=optimized,
        work_dir=tmp_path / "qm",
    )

    assert result.hessian_backend == "mace_polar1"
    assert result.checkpoint_path is None
    assert result.scf_runtime_seconds is None
    assert result.hessian_runtime_seconds == pytest.approx(2.0)
    assert result.scf_calculation_count is None
    assert result.fchk_path.is_file()
    report = json.loads(result.calculation_report_path.read_text(encoding="utf-8"))
    assert report["molecular_charge"] == 0
    assert report["multiplicity"] == 1
    assert report["hessian_backend"] == "mace_polar1"
