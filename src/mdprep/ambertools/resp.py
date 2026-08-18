"""Canonical two-stage Amber RESP fitting for PySCF ESP values.

The RESP executable consumes a fixed-width ESP file.  Keeping that conversion
here, rather than delegating it to a molecule-conversion fallback, lets mdprep
fit a PySCF electrostatic potential while using Amber's reference RESP
implementation and ``respgen`` equivalence rules.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mdprep.ambertools.mol2 import read_mol2
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command


class AmberRespError(RuntimeError):
    """Raised when canonical Amber RESP fitting cannot complete safely."""


@dataclass(frozen=True)
class RespCommandRun:
    """One preserved AmberTools command invocation."""

    name: str
    command_result: CommandResult
    stdout_path: Path
    stderr_path: Path
    record_path: Path
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": list(self.command_result.command),
            "cwd": self.command_result.cwd,
            "returncode": self.command_result.returncode,
            "stdout": self.command_result.stdout,
            "stderr": self.command_result.stderr,
            "runtime_seconds": self.command_result.runtime_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "record_path": str(self.record_path),
            "output_paths": [str(path) for path in self.output_paths],
        }


@dataclass(frozen=True)
class AmberRespFitResult:
    """Results and audit artifacts from a two-stage Amber RESP fit."""

    charges: np.ndarray
    stage_1_charges: np.ndarray
    charge_sum_before_correction: float
    charge_correction_applied: float
    charge_sum_final: float
    rms_error: float
    relative_rms_error: float | None
    max_error: float
    iterations: int
    stage_1_iterations: int
    stage_2_iterations: int
    converged: bool
    fitting_mode: str
    equivalence_mode: str
    stage_1_ivary: tuple[int, ...]
    stage_2_ivary: tuple[int, ...]
    esp_path: Path
    ac_path: Path
    stage_1_input_path: Path
    stage_2_input_path: Path
    stage_1_output_path: Path
    stage_2_output_path: Path
    stage_1_qout_path: Path
    stage_2_qout_path: Path
    command_runs: tuple[RespCommandRun, ...]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "charges": [float(value) for value in self.charges],
            "stage_1_charges": [float(value) for value in self.stage_1_charges],
            "charge_sum_before_correction": self.charge_sum_before_correction,
            "charge_correction_applied": self.charge_correction_applied,
            "charge_sum_final": self.charge_sum_final,
            "rms_error": self.rms_error,
            "relative_rms_error": self.relative_rms_error,
            "max_error": self.max_error,
            "iterations": self.iterations,
            "stage_1_iterations": self.stage_1_iterations,
            "stage_2_iterations": self.stage_2_iterations,
            "converged": self.converged,
            "fitting_mode": self.fitting_mode,
            "equivalence_mode": self.equivalence_mode,
            "stage_1_ivary": list(self.stage_1_ivary),
            "stage_2_ivary": list(self.stage_2_ivary),
            "esp_path": str(self.esp_path),
            "ac_path": str(self.ac_path),
            "stage_1_input_path": str(self.stage_1_input_path),
            "stage_2_input_path": str(self.stage_2_input_path),
            "stage_1_output_path": str(self.stage_1_output_path),
            "stage_2_output_path": str(self.stage_2_output_path),
            "stage_1_qout_path": str(self.stage_1_qout_path),
            "stage_2_qout_path": str(self.stage_2_qout_path),
            "commands": [run.to_dict() for run in self.command_runs],
            "warnings": self.warnings,
        }


def amber_resp_available() -> bool:
    """Return whether every executable needed by the RESP pipeline is present."""

    return all(which_executable(name) is not None for name in ("antechamber", "respgen", "resp"))


def write_amber_esp(
    *,
    atom_coordinates_bohr: np.ndarray,
    grid_coordinates_bohr: np.ndarray,
    esp_values_au: np.ndarray,
    path: str | Path,
) -> Path:
    """Write the fixed-width ESP input consumed by Amber ``resp``.

    Amber reads the first record as ``2I5``, atom coordinates as
    ``17X,3E16.7``, and potential/point coordinates as ``1X,4E16.7``.
    The fixed-width header is particularly important: adding a separator can
    make RESP silently interpret, for example, 430 grid points as 43.
    """

    atoms = np.asarray(atom_coordinates_bohr, dtype=float)
    points = np.asarray(grid_coordinates_bohr, dtype=float)
    values = np.asarray(esp_values_au, dtype=float)
    if atoms.ndim != 2 or atoms.shape[1] != 3 or len(atoms) == 0:
        raise AmberRespError("RESP atom coordinates must have shape (n_atoms, 3) and be non-empty.")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise AmberRespError("RESP grid coordinates must have shape (n_points, 3) and be non-empty.")
    if values.shape != (len(points),):
        raise AmberRespError("RESP ESP values must contain exactly one value per grid point.")
    if len(atoms) > 99999 or len(points) > 99999:
        raise AmberRespError("Amber RESP's 2I5 ESP header supports at most 99999 atoms or grid points.")
    if not np.all(np.isfinite(atoms)) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(values)):
        raise AmberRespError("RESP ESP input contains non-finite coordinates or potential values.")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(atoms):5d}{len(points):5d}"]
    lines.extend(" " * 17 + "".join(f"{value:16.7E}" for value in coordinate) for coordinate in atoms)
    lines.extend(
        " " + f"{potential:16.7E}" + "".join(f"{value:16.7E}" for value in coordinate)
        for potential, coordinate in zip(values, points, strict=True)
    )
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output


def parse_resp_qout(path: str | Path, *, atom_count: int) -> np.ndarray:
    """Parse an Amber RESP ``qout`` file and require exact atom coverage."""

    qout = Path(path)
    try:
        tokens = qout.read_text(encoding="utf-8").split()
        charges = np.asarray([float(token) for token in tokens], dtype=float)
    except (OSError, ValueError) as exc:
        raise AmberRespError(f"Could not parse RESP charge output {qout}: {exc}") from exc
    if charges.shape != (atom_count,):
        raise AmberRespError(
            f"RESP charge output {qout} has {len(charges)} charges; expected exactly {atom_count}."
        )
    if not np.all(np.isfinite(charges)):
        raise AmberRespError(f"RESP charge output {qout} contains non-finite values.")
    return charges


def run_amber_resp_fit(
    *,
    provisional_mol2_path: str | Path,
    atom_coordinates_bohr: np.ndarray,
    grid_coordinates_bohr: np.ndarray,
    esp_values_au: np.ndarray,
    total_charge: int,
    multiplicity: int,
    atom_types: str,
    work_dir: str | Path,
    antechamber_executable: str = "antechamber",
    respgen_executable: str = "respgen",
    resp_executable: str = "resp",
) -> AmberRespFitResult:
    """Run Amber's standard ``respgen``/two-stage ``resp`` workflow."""

    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    mol2_path = Path(provisional_mol2_path).resolve()
    mol2 = read_mol2(mol2_path)
    local_mol2_path = work / "ligand_input.mol2"
    shutil.copyfile(mol2_path, local_mol2_path)
    atom_count = len(mol2.atoms)
    atoms = np.asarray(atom_coordinates_bohr, dtype=float)
    points = np.asarray(grid_coordinates_bohr, dtype=float)
    values = np.asarray(esp_values_au, dtype=float)
    if atoms.shape != (atom_count, 3):
        raise AmberRespError(
            f"RESP atom-coordinate count {len(atoms)} does not match mol2 atom count {atom_count}."
        )

    executables = {
        "antechamber": _resolve_executable(antechamber_executable),
        "respgen": _resolve_executable(respgen_executable),
        "resp": _resolve_executable(resp_executable),
    }
    ac_path = work / "ligand.ac"
    esp_path = write_amber_esp(
        atom_coordinates_bohr=atoms,
        grid_coordinates_bohr=points,
        esp_values_au=values,
        path=work / "ligand.esp",
    )
    resp1_input = work / "resp1.in"
    resp2_input = work / "resp2.in"
    resp1_output = work / "resp1.out"
    resp2_output = work / "resp2.out"
    qout1 = work / "qout_stage1"
    qout2 = work / "qout_stage2"
    resp1_punch = work / "resp1.pch"
    resp2_punch = work / "resp2.pch"
    resp1_esout = work / "resp1.esout"
    resp2_esout = work / "resp2.esout"

    runs: list[RespCommandRun] = []
    runs.append(
        _run_checked(
            name="antechamber_mol2_to_ac",
            command=[
                executables["antechamber"],
                "-i",
                local_mol2_path.name,
                "-fi",
                "mol2",
                "-o",
                ac_path.name,
                "-fo",
                "ac",
                "-at",
                atom_types,
                "-nc",
                str(total_charge),
                "-m",
                str(multiplicity),
                "-j",
                "0",
                "-an",
                "n",
                "-du",
                "n",
                "-seq",
                "n",
                "-s",
                "2",
            ],
            work_dir=work,
            output_paths=[ac_path],
        )
    )
    _assert_ac_atom_order(ac_path, [atom.name for atom in mol2.atoms])

    for stage, output_path in (("resp1", resp1_input), ("resp2", resp2_input)):
        runs.append(
            _run_checked(
                name=f"respgen_{stage}",
                command=[
                    executables["respgen"],
                    "-i",
                    ac_path.name,
                    "-o",
                    output_path.name,
                    "-f",
                    stage,
                ],
                work_dir=work,
                output_paths=[output_path],
            )
        )

    stage_1_ivary = _validate_respin(
        resp1_input,
        atom_count=atom_count,
        total_charge=total_charge,
        expected_qwt=0.0005,
        require_iqopt=False,
    )
    stage_2_ivary = _validate_respin(
        resp2_input,
        atom_count=atom_count,
        total_charge=total_charge,
        expected_qwt=0.001,
        require_iqopt=True,
    )

    runs.append(
        _run_checked(
            name="resp_stage1",
            command=[
                executables["resp"],
                "-O",
                "-i",
                resp1_input.name,
                "-o",
                resp1_output.name,
                "-p",
                resp1_punch.name,
                "-t",
                qout1.name,
                "-e",
                esp_path.name,
                "-s",
                resp1_esout.name,
            ],
            work_dir=work,
            output_paths=[resp1_output, qout1, resp1_punch, resp1_esout],
        )
    )
    stage_1_charges = parse_resp_qout(qout1, atom_count=atom_count)
    runs.append(
        _run_checked(
            name="resp_stage2",
            command=[
                executables["resp"],
                "-O",
                "-i",
                resp2_input.name,
                "-o",
                resp2_output.name,
                "-p",
                resp2_punch.name,
                "-q",
                qout1.name,
                "-t",
                qout2.name,
                "-e",
                esp_path.name,
                "-s",
                resp2_esout.name,
            ],
            work_dir=work,
            output_paths=[resp2_output, qout2, resp2_punch, resp2_esout],
        )
    )
    charges = parse_resp_qout(qout2, atom_count=atom_count)
    stage_1_iterations = _convergence_iterations(resp1_output)
    stage_2_iterations = _convergence_iterations(resp2_output)
    charge_sum = float(np.sum(charges))
    rounding_tolerance = max(1.0e-5, atom_count * 5.1e-7)
    if not math.isclose(charge_sum, float(total_charge), rel_tol=0.0, abs_tol=rounding_tolerance):
        raise AmberRespError(
            f"Two-stage RESP charge sum is {charge_sum:.8f} e, but the requested ligand charge is "
            f"{total_charge}; qout rounding tolerance is {rounding_tolerance:.2e} e."
        )
    rms, relative_rms, max_error = _fit_statistics(
        atom_coordinates_bohr=atoms,
        grid_coordinates_bohr=points,
        esp_values_au=values,
        charges=charges,
    )
    result = AmberRespFitResult(
        charges=charges,
        stage_1_charges=stage_1_charges,
        charge_sum_before_correction=charge_sum,
        charge_correction_applied=0.0,
        charge_sum_final=charge_sum,
        rms_error=rms,
        relative_rms_error=relative_rms,
        max_error=max_error,
        iterations=stage_1_iterations + stage_2_iterations,
        stage_1_iterations=stage_1_iterations,
        stage_2_iterations=stage_2_iterations,
        converged=True,
        fitting_mode="ambertools_two_stage_resp",
        equivalence_mode="respgen_atomic_paths",
        stage_1_ivary=stage_1_ivary,
        stage_2_ivary=stage_2_ivary,
        esp_path=esp_path,
        ac_path=ac_path,
        stage_1_input_path=resp1_input,
        stage_2_input_path=resp2_input,
        stage_1_output_path=resp1_output,
        stage_2_output_path=resp2_output,
        stage_1_qout_path=qout1,
        stage_2_qout_path=qout2,
        command_runs=tuple(runs),
        warnings=[],
    )
    (work / "resp_fit.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _resolve_executable(name: str) -> str:
    executable = which_executable(name)
    if executable is None:
        raise AmberRespError(
            f"AmberTools executable not found: {name}. Canonical two-stage RESP requires "
            "antechamber, respgen, and resp; mdprep will not substitute simplified or Mulliken charges."
        )
    return executable


def _run_checked(
    *,
    name: str,
    command: Sequence[str],
    work_dir: Path,
    output_paths: Sequence[Path],
) -> RespCommandRun:
    for path in output_paths:
        if path.exists():
            path.unlink()
    result = run_command(command, cwd=work_dir)
    stdout_path = work_dir / f"{name}_stdout.txt"
    stderr_path = work_dir / f"{name}_stderr.txt"
    record_path = work_dir / f"{name}_command.json"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    run = RespCommandRun(
        name=name,
        command_result=result,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        record_path=record_path,
        output_paths=tuple(output_paths),
    )
    record_path.write_text(json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.returncode != 0:
        raise AmberRespError(
            f"{name} failed with exit code {result.returncode}. Command: {' '.join(result.command)}. "
            f"See {stdout_path} and {stderr_path}. stderr tail: {_tail(result.stderr)}"
        )
    missing = [path for path in output_paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise AmberRespError(
            f"{name} did not produce non-empty expected output(s): "
            + ", ".join(str(path) for path in missing)
        )
    return run


def _assert_ac_atom_order(path: Path, expected_names: list[str]) -> None:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0] == "ATOM" and len(fields) >= 3:
            names.append(fields[2])
    if names != expected_names:
        raise AmberRespError(
            "antechamber mol2-to-AC conversion changed ligand atom count, names, or order: "
            f"expected {expected_names}, observed {names}."
        )


def _validate_respin(
    path: Path,
    *,
    atom_count: int,
    total_charge: int,
    expected_qwt: float,
    require_iqopt: bool,
) -> tuple[int, ...]:
    text = path.read_text(encoding="utf-8")
    qwt_match = re.search(r"\bqwt\s*=\s*([0-9.EeDd+-]+)", text, flags=re.IGNORECASE)
    if qwt_match is None or not math.isclose(
        float(qwt_match.group(1).replace("D", "E").replace("d", "e")),
        expected_qwt,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise AmberRespError(f"respgen produced a non-standard restraint weight in {path}.")
    has_iqopt_2 = re.search(r"\biqopt\s*=\s*2\b", text, flags=re.IGNORECASE) is not None
    if has_iqopt_2 != require_iqopt:
        raise AmberRespError(f"respgen produced an unexpected iqopt setting in {path}.")

    lines = text.splitlines()
    header_index: int | None = None
    parsed_charge: int | None = None
    for index, line in enumerate(lines[:-1]):
        if not line.strip().lower().startswith("resp charges"):
            continue
        next_index = _next_nonempty_line(lines, index + 1)
        if next_index is None:
            continue
        match = re.fullmatch(r"\s*(-?\d+)\s+(\d+)\s*", lines[next_index])
        if match is not None:
            parsed_charge = int(match.group(1))
            if int(match.group(2)) == atom_count:
                header_index = next_index
                break
    if header_index is None or parsed_charge != total_charge:
        raise AmberRespError(
            f"respgen input {path} does not encode requested charge {total_charge} and atom count {atom_count}."
        )

    ivary: list[int] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            if ivary:
                break
            continue
        match = re.fullmatch(r"\s*\d+\s+(-?\d+)\s*", line)
        if match is None:
            if ivary:
                break
            continue
        ivary.append(int(match.group(1)))
        if len(ivary) == atom_count:
            break
    if len(ivary) != atom_count:
        raise AmberRespError(f"respgen input {path} has {len(ivary)} atom constraints; expected {atom_count}.")
    return tuple(ivary)


def _next_nonempty_line(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _convergence_iterations(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Convergence\s+in\s+(\d+)\s+iterations", text, flags=re.IGNORECASE)
    if not matches:
        raise AmberRespError(f"RESP output {path} does not report convergence.")
    return int(matches[-1])


def _fit_statistics(
    *,
    atom_coordinates_bohr: np.ndarray,
    grid_coordinates_bohr: np.ndarray,
    esp_values_au: np.ndarray,
    charges: np.ndarray,
) -> tuple[float, float | None, float]:
    distances = np.linalg.norm(
        grid_coordinates_bohr[:, None, :] - atom_coordinates_bohr[None, :, :],
        axis=2,
    )
    if np.any(distances < 1.0e-10):
        raise AmberRespError("ESP grid contains a point on a fitted atom center.")
    predicted = (1.0 / distances) @ charges
    errors = predicted - esp_values_au
    rms = float(np.sqrt(np.mean(errors * errors)))
    esp_rms = float(np.sqrt(np.mean(esp_values_au * esp_values_au)))
    relative = rms / esp_rms if esp_rms > 0.0 else None
    return rms, relative, float(np.max(np.abs(errors)))


def _tail(text: str, *, lines: int = 20) -> str:
    stripped = text.strip()
    if not stripped:
        return "<empty>"
    return "\n".join(stripped.splitlines()[-lines:])
