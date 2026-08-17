from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mdprep.config.models import (
    McpbGxtbHessianConfig,
    McpbPySCFConfig,
    McpbXtbHessianConfig,
)
from mdprep.external.runner import CommandResult
from mdprep.metals.gxtb_hessian import (
    GxtbHessianError,
    build_gxtb_hessian_command,
    parse_gxtb_hessian,
    run_gxtb_hessian,
)
from mdprep.metals.pyscf_mcpb import McpbPySCFError, _run_small_model


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


def _gxtb_stdout(*, charge: int = 0, spin: int = 5) -> str:
    return (
        "   for g-xTB:\n"
        f"molecular charge               {float(charge):.13E} e\n"
        f"number of unpaired electrons   {float(spin):.13E} e\n"
        "wavefunction                   unrestricted\n"
        "total energy                  -1.2345678900000E+02 Eh\n"
        "normal termination of xtb\n"
    )


def test_gxtb_command_is_fixed_geometry_hessian():
    command = build_gxtb_hessian_command(
        executable="/opt/gxtb/xtb",
        xyz_name="small.xyz",
        charge=0,
        spin=5,
        accuracy=0.2,
    )

    assert command == [
        "/opt/gxtb/xtb",
        "small.xyz",
        "--gxtb",
        "--chrg",
        "0",
        "--uhf",
        "5",
        "--hess",
        "--acc",
        "0.2",
    ]
    assert "--opt" not in command
    assert "--ohess" not in command


def test_gfn2_command_is_fixed_geometry_hessian():
    command = build_gxtb_hessian_command(
        executable="/opt/xtb",
        xyz_name="small.xyz",
        charge=-1,
        spin=2,
        accuracy=0.1,
        model="gfn2",
    )

    assert command[:5] == ["/opt/xtb", "small.xyz", "--gfn", "2", "--chrg"]
    assert "--gxtb" not in command
    assert "--hess" in command
    assert "--opt" not in command


def test_gxtb_command_accepts_explicit_scf_iteration_limit():
    command = build_gxtb_hessian_command(
        executable="/opt/gxtb/xtb",
        xyz_name="small.xyz",
        charge=0,
        spin=5,
        accuracy=0.001,
        max_scf_iterations=1000,
    )

    assert command[-2:] == ["--iterations", "1000"]


def test_parse_gxtb_hessian_preserves_full_row_major_matrix(tmp_path):
    matrix = np.arange(36, dtype=float).reshape((6, 6))
    path = tmp_path / "hessian"
    path.write_text(
        "$hessian\n"
        + "\n".join(
            " ".join(f"{value:.10E}" for value in row) for row in matrix
        )
        + "\n$end\n",
        encoding="utf-8",
    )

    parsed = parse_gxtb_hessian(path, atom_count=2)

    assert parsed.shape == (6, 6)
    assert np.array_equal(parsed, matrix)


def test_parse_gxtb_hessian_rejects_wrong_dimension(tmp_path):
    path = tmp_path / "hessian"
    path.write_text("$hessian\n1.0 2.0\n", encoding="utf-8")

    with pytest.raises(GxtbHessianError, match="expected 36"):
        parse_gxtb_hessian(path, atom_count=2)


def test_gxtb_backend_requires_explicit_settings():
    with pytest.raises(ValidationError, match="requires explicit gxtb settings"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": "optimized.pdb",
                "small_model_geometry_status": "user_optimized",
                "hessian_backend": "gxtb",
                "method": "B3LYP",
                "basis": "6-31G*",
            }
        )


def test_generic_xtb_backend_requires_explicit_settings():
    with pytest.raises(ValidationError, match="requires explicit xtb settings"):
        McpbPySCFConfig.model_validate(
            {
                "optimized_small_model_pdb": "optimized.pdb",
                "small_model_geometry_status": "user_optimized",
                "hessian_backend": "xtb",
                "method": "B3LYP",
                "basis": "6-31G*",
            }
        )

    config = McpbPySCFConfig.model_validate(
        {
            "geometry_source": "qmmm_refinement",
            "hessian_backend": "xtb",
            "xtb": {"model": "gfn2", "executable": "/opt/xtb"},
            "method": "B3LYP",
            "basis": "6-31G*",
        }
    )
    assert config.xtb is not None
    assert config.xtb.model == "gfn2"


def test_gxtb_hessian_uses_tight_deterministic_defaults():
    config = McpbGxtbHessianConfig()

    assert config.accuracy == pytest.approx(0.001)
    assert config.num_threads == 1
    assert config.max_scf_iterations is None


def test_gfn2_hessian_uses_tight_deterministic_defaults():
    config = McpbXtbHessianConfig()

    assert config.model == "gfn2"
    assert config.accuracy == pytest.approx(0.001)


def test_gxtb_runner_records_commands_and_validates_hessian(monkeypatch, tmp_path):
    executable = tmp_path / "xtb"
    executable.write_bytes(b"pinned fake executable")
    executable.chmod(0o755)
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    xyz = tmp_path / "small.xyz"
    xyz.write_text("2\nfixed\nN 0 0 0\nH 1 0 0\n", encoding="utf-8")
    matrix = np.eye(6)
    calls: list[tuple[str, ...]] = []

    def fake_run_command(command, *, cwd, env):
        calls.append(tuple(command))
        if "--version" in command:
            return CommandResult(
                command=tuple(command),
                cwd=str(cwd),
                returncode=0,
                stdout="xtb version 6.7.1 (test)\n",
                stderr="",
                runtime_seconds=0.01,
            )
        (Path(cwd) / "hessian").write_text(
            "$hessian\n"
            + "\n".join(
                " ".join(f"{value:.10E}" for value in row) for row in matrix
            )
            + "\n",
            encoding="utf-8",
        )
        assert env["OMP_NUM_THREADS"] == "1"
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=0,
            stdout=_gxtb_stdout(),
            stderr="",
            runtime_seconds=2.5,
        )

    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_command",
        fake_run_command,
    )
    result = run_gxtb_hessian(
        xyz_path=xyz,
        atom_count=2,
        charge=0,
        spin=5,
        config=McpbGxtbHessianConfig(
            executable=str(executable),
            release_tag="v2.0.1",
            expected_executable_sha256=executable_sha,
            accuracy=0.2,
            num_threads=1,
        ),
        work_dir=tmp_path,
    )

    assert len(calls) == 2
    assert result.energy_hartree == pytest.approx(-123.456789)
    assert result.molecular_charge == 0
    assert result.unpaired_electrons == 5
    assert np.array_equal(result.hessian_hartree_per_bohr2, matrix)
    assert result.command_result.runtime_seconds == pytest.approx(2.5)
    report = result.report_path.read_text(encoding="utf-8")
    assert '"release_tag": "v2.0.1"' in report
    assert '"runtime_seconds": 2.5' in report


def test_small_model_can_use_gxtb_without_small_model_pyscf(
    monkeypatch,
    tmp_path,
):
    generated = _write_two_atom_pdb(tmp_path / "generated.pdb")
    optimized = _write_two_atom_pdb(tmp_path / "optimized.pdb")
    executable = tmp_path / "xtb"
    executable.write_bytes(b"fake")
    stdout = tmp_path / "gxtb.stdout"
    stderr = tmp_path / "gxtb.stderr"
    stdout.write_text(_gxtb_stdout(spin=0), encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    def fail_pyscf(**kwargs):
        raise AssertionError("PySCF small-model SCF must not run for g-xTB Hessian")

    monkeypatch.setattr("mdprep.metals.pyscf_mcpb.run_pyscf_scf", fail_pyscf)
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_gxtb_hessian",
        lambda **kwargs: SimpleNamespace(
            hessian_hartree_per_bohr2=np.eye(6),
            energy_hartree=-123.0,
            command_result=SimpleNamespace(runtime_seconds=1.25),
            version_text="6.7.1 (test)",
            stdout_path=stdout,
            stderr_path=stderr,
            raw_hessian_asymmetry_warning_count=0,
            to_dict=lambda: {
                "version_command": {
                    "command": ["xtb", "--version"],
                    "cwd": str(tmp_path),
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "runtime_seconds": 0.01,
                },
                "hessian_command": {
                    "command": ["xtb", "--gxtb", "--hess"],
                    "cwd": str(tmp_path),
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "runtime_seconds": 1.25,
                },
            },
        ),
    )
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(optimized),
            "small_model_geometry_status": "test_unoptimized",
            "hessian_backend": "gxtb",
            "gxtb": {
                "executable": str(executable),
                "release_tag": "v2.0.1",
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

    assert result.hessian_backend == "gxtb"
    assert result.checkpoint_path is None
    assert result.scf_runtime_seconds is None
    assert result.hessian_runtime_seconds == pytest.approx(1.25)
    assert result.scf_calculation_count is None
    assert result.fchk_path.is_file()
    report = json.loads(result.calculation_report_path.read_text(encoding="utf-8"))
    assert report["molecular_charge"] == 0
    assert report["multiplicity"] == 1
    assert report["spin"] == 0


def test_production_gxtb_hessian_rejects_raw_asymmetry_warnings(
    monkeypatch,
    tmp_path,
):
    generated = _write_two_atom_pdb(tmp_path / "generated.pdb")
    optimized = _write_two_atom_pdb(tmp_path / "optimized.pdb")
    executable = tmp_path / "xtb"
    executable.write_bytes(b"fake")
    stdout = tmp_path / "gxtb.stdout"
    stderr = tmp_path / "gxtb.stderr"
    stdout.write_text(_gxtb_stdout(spin=0), encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_gxtb_hessian",
        lambda **kwargs: SimpleNamespace(
            raw_hessian_asymmetry_warning_count=1,
        ),
    )
    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": str(optimized),
            "small_model_geometry_status": "user_optimized",
            "hessian_backend": "gxtb",
            "gxtb": {"executable": str(executable)},
            "method": "B3LYP",
            "basis": "6-31G*",
        }
    )

    with pytest.raises(McpbPySCFError, match="raw numerical Hessian-asymmetry"):
        _run_small_model(
            generated,
            output_fchk=tmp_path / "site_small_opt.fchk",
            charge=0,
            multiplicity=1,
            config=config,
            user_optimized_pdb=optimized,
            work_dir=tmp_path / "qm",
        )
