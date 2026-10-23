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


def test_the_default_xtb_backend_needs_no_settings_block():
    """hessian_backend: xtb is the default, so a manifest that wants the
    defaults should not have to restate them."""

    config = McpbPySCFConfig.model_validate(
        {
            "optimized_small_model_pdb": "optimized.pdb",
            "small_model_geometry_status": "user_optimized",
            "hessian_backend": "xtb",
            "method": "B3LYP",
            "basis": "6-31G*",
        }
    )

    assert config.xtb is not None
    assert config.xtb.model == "gxtb"


def test_generic_xtb_backend_accepts_explicit_settings():
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


def test_a_metal_site_hessian_defaults_to_gxtb():
    """GFN2 is parameterised for main-group chemistry and gets transition
    metals wrong -- on the BesD iron it gives one Fe-O bond a negative force
    constant -- so a metal site asks for g-xTB unless told otherwise."""

    config = McpbXtbHessianConfig()

    assert config.model == "gxtb"
    assert config.accuracy == pytest.approx(0.001)
    # Left unset so the resolver can look for a g-xTB build; an unqualified
    # "xtb" on PATH is usually the GFN-only conda package.
    assert config.executable is None


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
                "scf_starts": [],
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


def test_a_gxtb_build_is_looked_for_before_falling_back_to_path(monkeypatch):
    """The xtb conda installs is GFN-only, so an unqualified "xtb" on PATH is
    usually the wrong binary for a g-xTB Hessian."""

    from mdprep.metals.gxtb_hessian import (
        GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE,
        gxtb_executable_candidates,
    )

    monkeypatch.delenv(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE, raising=False)
    assert gxtb_executable_candidates(None, "gxtb") == ["g-xtb", "xtb"]

    monkeypatch.setenv(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE, "/opt/g-xtb/bin/xtb")
    assert gxtb_executable_candidates(None, "gxtb")[0] == "/opt/g-xtb/bin/xtb"

    # An explicit path in the manifest wins, and is the only thing tried.
    assert gxtb_executable_candidates("/my/xtb", "gxtb") == ["/my/xtb"]
    # GFN2 wants the ordinary binary.
    assert gxtb_executable_candidates(None, "gfn2") == ["xtb"]


def test_a_missing_gxtb_build_says_how_to_get_one(monkeypatch, tmp_path):
    from mdprep.metals.gxtb_hessian import (
        GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE,
        GxtbHessianError,
        _resolve_executable,
    )

    monkeypatch.delenv(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.which_executable", lambda name: None
    )

    with pytest.raises(GxtbHessianError) as excinfo:
        _resolve_executable(None, model="gxtb")

    message = str(excinfo.value)
    assert "github.com/grimme-lab/g-xtb" in message
    assert GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE in message
    # And names the escape hatch, with the reason it is not the default.
    assert "model: gfn2" in message


# --------------------------------------------------------------------------
# Which SCF state the Hessian is taken in
# --------------------------------------------------------------------------


def _fake_xtb_with_states(energies: dict[str, float], calls: list, *, hessian_state=None):
    """An xtb whose SCF lands in a state named by how it was started.

    Straight at the cold temperature it lands in "direct"; converged hot first,
    it lands in the state named after the hot temperature; a Hessian stays in
    whatever state its restart file names, unless told otherwise.
    """

    def stdout(energy: float) -> str:
        return _gxtb_stdout(charge=-1, spin=5).replace(
            "-1.2345678900000E+02", f"{energy:.13E}"
        )

    def fake_run_command(command, *, cwd, env):
        cwd = Path(cwd)
        calls.append((tuple(command), cwd.name, env["OMP_NUM_THREADS"]))
        if "--version" in command:
            return CommandResult(
                command=tuple(command), cwd=str(cwd), returncode=0,
                stdout="xtb version 6.7.1 (test)\n", stderr="", runtime_seconds=0.01,
            )
        kelvin = command[command.index("--etemp") + 1]
        restart = cwd / "xtbrestart"
        if "--hess" in command:
            state = hessian_state or restart.read_text(encoding="utf-8")
            (cwd / "hessian").write_text(
                "$hessian\n" + "\n".join(" ".join(f"{v:.10E}" for v in row) for row in np.eye(6)) + "\n",
                encoding="utf-8",
            )
        elif kelvin == "1000":
            state = restart.read_text(encoding="utf-8") if restart.exists() else "direct"
        else:
            state = kelvin
        if energies.get(state) is None:
            return CommandResult(
                command=tuple(command), cwd=str(cwd), returncode=128,
                stdout="SCF not converged in 3000 cycles\n",
                stderr="abnormal termination of xtb\n", runtime_seconds=1.0,
            )
        # As xTB does, before its setup summary.
        read = "Wavefunction read from restart file\n" if restart.exists() else ""
        restart.write_text(state, encoding="utf-8")
        return CommandResult(
            command=tuple(command), cwd=str(cwd), returncode=0,
            stdout=read + stdout(energies[state]), stderr="", runtime_seconds=1.0,
        )

    return fake_run_command


def _searching_config(executable: Path, **settings) -> McpbXtbHessianConfig:
    return McpbXtbHessianConfig(
        executable=str(executable),
        num_threads=8,
        electronic_temperature_kelvin=1000.0,
        max_scf_iterations=3000,
        scf_annealing_kelvin=[2000.0, 5000.0],
        **settings,
    )


def _two_atoms(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "xtb"
    executable.write_bytes(b"fake")
    xyz = tmp_path / "work" / "small.xyz"
    xyz.parent.mkdir()
    xyz.write_text("2\nfixed\nFe 0 0 0\nN 2 0 0\n", encoding="utf-8")
    return executable, xyz


def test_the_hessian_is_taken_in_the_lowest_state_any_start_reaches(monkeypatch, tmp_path):
    """2QFK's high-spin Fe(III) settled straight at 1000 K 27 kcal/mol above the
    state a start annealed from a hotter SCF reached, and the Fe-N force
    constants of the two differ by up to 1.7 times."""

    executable, xyz = _two_atoms(tmp_path)
    calls: list = []
    energies = {"direct": -100.000, "2000": -100.043, "5000": -100.040}
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_command", _fake_xtb_with_states(energies, calls)
    )

    result = run_gxtb_hessian(
        xyz_path=xyz, atom_count=2, charge=-1, spin=5,
        config=_searching_config(executable), work_dir=xyz.parent,
    )

    assert result.energy_hartree == pytest.approx(-100.043)
    assert [start.label for start in result.scf_starts] == ["from 2000 K", "from 5000 K", "direct"]
    # Every start is single-threaded; the Hessian uses the threads asked for.
    searched = [call for call in calls if "--hess" not in call[0] and "--version" not in call[0]]
    assert len(searched) == 5 and {call[2] for call in searched} == {"1"}
    (hessian,) = [call for call in calls if "--hess" in call[0]]
    assert hessian[2] == "8"
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert [start["energy_hartree"] for start in report["scf_starts"]] == [-100.043, -100.04, -100.0]
    # Each single point's output is kept beside it.
    assert sorted(path.name for path in (xyz.parent / "scf_start_2000K").glob("*.stdout.txt")) == [
        "gxtb.sp_1000K.stdout.txt", "gxtb.sp_2000K.stdout.txt",
    ]


def test_a_start_that_does_not_converge_is_recorded_and_passed_over(monkeypatch, tmp_path):
    executable, xyz = _two_atoms(tmp_path)
    calls: list = []
    energies = {"direct": -100.000, "2000": None, "5000": -100.040}
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_command", _fake_xtb_with_states(energies, calls)
    )

    result = run_gxtb_hessian(
        xyz_path=xyz, atom_count=2, charge=-1, spin=5,
        config=_searching_config(executable), work_dir=xyz.parent,
    )

    assert result.energy_hartree == pytest.approx(-100.040)
    failed = [start for start in result.scf_starts if not start.converged]
    assert [(start.label, start.energy_hartree) for start in failed] == [("from 2000 K", None)]


def test_no_start_converging_says_which_were_tried(monkeypatch, tmp_path):
    executable, xyz = _two_atoms(tmp_path)
    energies = {"direct": None, "2000": None, "5000": None}
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_command", _fake_xtb_with_states(energies, [])
    )

    with pytest.raises(GxtbHessianError, match=r"direct, from 2000 K, from 5000 K.*max_scf_iterations"):
        run_gxtb_hessian(
            xyz_path=xyz, atom_count=2, charge=-1, spin=5,
            config=_searching_config(executable), work_dir=xyz.parent,
        )


def test_a_hessian_that_leaves_the_chosen_state_is_refused(monkeypatch, tmp_path):
    executable, xyz = _two_atoms(tmp_path)
    energies = {"direct": -100.000, "2000": -100.043, "5000": -100.040}
    monkeypatch.setattr(
        "mdprep.metals.gxtb_hessian.run_command",
        _fake_xtb_with_states(energies, [], hessian_state="direct"),
    )

    with pytest.raises(GxtbHessianError, match="ended in another"):
        run_gxtb_hessian(
            xyz_path=xyz, atom_count=2, charge=-1, spin=5,
            config=_searching_config(executable), work_dir=xyz.parent,
        )


def test_annealing_starts_hotter_than_it_ends():
    with pytest.raises(ValidationError, match="hotter than"):
        McpbXtbHessianConfig(electronic_temperature_kelvin=1000.0, scf_annealing_kelvin=[800.0])


def test_a_run_from_a_restart_file_is_still_read_as_unrestricted():
    """xTB announces the restart as "Wavefunction read from restart file",
    which the check once took for the wavefunction's kind."""
    from mdprep.metals.gxtb_hessian import _validate_open_shell_metadata

    restarted = "Wavefunction read from restart file\n" + _gxtb_stdout(charge=-1, spin=5)
    _validate_open_shell_metadata(restarted, expected_charge=-1, expected_spin=5)

    restricted = restarted.replace("unrestricted", "restricted")
    with pytest.raises(GxtbHessianError, match="unrestricted"):
        _validate_open_shell_metadata(restricted, expected_charge=-1, expected_spin=5)
