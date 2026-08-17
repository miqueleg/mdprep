"""Validated fixed-geometry MACE-POLAR-1 Hessians for MCPB.py."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdprep.config.models import McpbMacePolar1HessianConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command


class MacePolar1HessianError(RuntimeError):
    """Raised when MACE-POLAR-1 cannot produce a validated Hessian."""


@dataclass(frozen=True)
class MacePolar1HessianResult:
    hessian_hartree_per_bohr2: np.ndarray
    model: str
    energy_hartree: float
    atom_count: int
    molecular_charge: int
    multiplicity: int
    python_executable_path: Path
    python_executable_sha256: str
    model_path: Path
    model_sha256: str
    versions: dict[str, str | None]
    device: str
    probe_result: CommandResult
    command_result: CommandResult
    hessian_path: Path
    stdout_path: Path
    stderr_path: Path
    probe_stdout_path: Path
    probe_stderr_path: Path
    worker_report_path: Path
    report_path: Path
    antisymmetry_max: float
    gradient_norm_hartree_per_bohr: float
    finite_difference_relative_error: float

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": "MACE-POLAR-1",
            "model": self.model,
            "versions": self.versions,
            "device": self.device,
            "default_dtype": "float64",
            "python_executable_path": str(self.python_executable_path),
            "python_executable_sha256": self.python_executable_sha256,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "energy_hartree": self.energy_hartree,
            "atom_count": self.atom_count,
            "molecular_charge": self.molecular_charge,
            "multiplicity": self.multiplicity,
            "spin_input_semantics": "spin multiplicity (unpaired electrons + 1)",
            "hessian_path": str(self.hessian_path),
            "hessian_sha256": _sha256(self.hessian_path),
            "hessian_units": "hartree/bohr^2",
            "hessian_antisymmetry_max": self.antisymmetry_max,
            "gradient_norm_hartree_per_bohr": self.gradient_norm_hartree_per_bohr,
            "finite_difference_hessian_vector_relative_error": (
                self.finite_difference_relative_error
            ),
            "probe_command": _command_to_dict(self.probe_result),
            "hessian_command": _command_to_dict(self.command_result),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "probe_stdout_path": str(self.probe_stdout_path),
            "probe_stderr_path": str(self.probe_stderr_path),
            "worker_report_path": str(self.worker_report_path),
            "report_path": str(self.report_path),
            "geometry_optimization_performed": False,
            "hessian_interpretation": (
                "Full unweighted analytical Cartesian Hessian from MACE-POLAR-1, "
                "evaluated at the input geometry in float64 and independently checked "
                "against a deterministic central difference of MACE forces."
            ),
        }


def run_mace_polar1_hessian(
    *,
    xyz_path: str | Path,
    atom_count: int,
    charge: int,
    multiplicity: int,
    config: McpbMacePolar1HessianConfig,
    work_dir: str | Path,
) -> MacePolar1HessianResult:
    """Run MACE-POLAR-1 in its configured Python environment and validate it."""

    if atom_count < 1:
        raise MacePolar1HessianError("MACE-POLAR-1 Hessian requires at least one atom.")
    if multiplicity < 1:
        raise MacePolar1HessianError("MACE-POLAR-1 multiplicity must be positive.")
    if not config.accept_model_license:
        raise MacePolar1HessianError(
            "MACE-POLAR-1 model license was not explicitly accepted."
        )
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    source_xyz = Path(xyz_path)
    if source_xyz.resolve().parent != work.resolve():
        raise MacePolar1HessianError(
            "MACE-POLAR-1 input XYZ must already be staged in its dedicated working directory."
        )
    if not source_xyz.is_file() or source_xyz.stat().st_size == 0:
        raise MacePolar1HessianError(
            f"MACE-POLAR-1 input XYZ is missing or empty: {source_xyz}"
        )

    python = _resolve_python(config.python_executable)
    worker = Path(__file__).with_name("mace_polar1_worker.py").resolve()
    environment = dict(os.environ)
    thread_count = str(config.num_threads)
    matplotlib_config = work / ".matplotlib"
    matplotlib_config.mkdir(exist_ok=True)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "VECLIB_MAXIMUM_THREADS": thread_count,
        }
    )
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    probe_result = run_command(
        [str(python), str(worker), "--probe"],
        cwd=work,
        env=environment,
    )
    probe_stdout_path = work / "mace_polar1.probe.stdout.txt"
    probe_stderr_path = work / "mace_polar1.probe.stderr.txt"
    probe_stdout_path.write_text(probe_result.stdout, encoding="utf-8")
    probe_stderr_path.write_text(probe_result.stderr, encoding="utf-8")
    if probe_result.returncode != 0:
        raise MacePolar1HessianError(
            "MACE-POLAR-1 dependency probe failed with exit code "
            f"{probe_result.returncode}. See {probe_stdout_path} and {probe_stderr_path}."
        )
    probe = _last_json_object(probe_result.stdout, label="dependency probe")
    if probe.get("status") != "available":
        raise MacePolar1HessianError("MACE-POLAR-1 dependency probe was not successful.")

    hessian_path = work / "mace_polar1_hessian.npy"
    worker_report_path = work / "mace_polar1_worker_report.json"
    hessian_path.unlink(missing_ok=True)
    worker_report_path.unlink(missing_ok=True)
    command = [
        str(python),
        str(worker),
        "--xyz",
        source_xyz.name,
        "--atom-count",
        str(atom_count),
        "--charge",
        str(charge),
        "--multiplicity",
        str(multiplicity),
        "--model",
        config.model,
        "--device",
        config.device,
        "--finite-difference-step-angstrom",
        f"{config.finite_difference_step_angstrom:.12g}",
        "--max-hessian-vector-relative-error",
        f"{config.max_hessian_vector_relative_error:.12g}",
        "--output-hessian",
        hessian_path.name,
        "--output-report",
        worker_report_path.name,
    ]
    if config.expected_model_sha256 is not None:
        command.extend(
            ["--expected-model-sha256", config.expected_model_sha256.lower()]
        )
    command_result = run_command(command, cwd=work, env=environment)
    stdout_path = work / "mace_polar1.hessian.stdout.txt"
    stderr_path = work / "mace_polar1.hessian.stderr.txt"
    stdout_path.write_text(command_result.stdout, encoding="utf-8")
    stderr_path.write_text(command_result.stderr, encoding="utf-8")
    if command_result.returncode != 0:
        raise MacePolar1HessianError(
            "MACE-POLAR-1 Hessian failed with exit code "
            f"{command_result.returncode}. See {stdout_path} and {stderr_path}."
        )
    if not worker_report_path.is_file() or worker_report_path.stat().st_size == 0:
        raise MacePolar1HessianError(
            f"MACE-POLAR-1 did not produce its worker report: {worker_report_path}"
        )
    metadata = json.loads(worker_report_path.read_text(encoding="utf-8"))
    _validate_metadata(
        metadata,
        atom_count=atom_count,
        charge=charge,
        multiplicity=multiplicity,
        config=config,
    )
    if not hessian_path.is_file() or hessian_path.stat().st_size == 0:
        raise MacePolar1HessianError(
            f"MACE-POLAR-1 did not produce its Hessian: {hessian_path}"
        )
    hessian = np.asarray(np.load(hessian_path, allow_pickle=False), dtype=float)
    expected_shape = (atom_count * 3, atom_count * 3)
    if hessian.shape != expected_shape:
        raise MacePolar1HessianError(
            f"MACE-POLAR-1 Hessian shape is {hessian.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(hessian)):
        raise MacePolar1HessianError("MACE-POLAR-1 Hessian contains non-finite values.")
    antisymmetry = float(np.max(np.abs(hessian - hessian.T)))
    if antisymmetry > 1.0e-8:
        raise MacePolar1HessianError(
            "MACE-POLAR-1 Cartesian Hessian is not symmetric "
            f"(maximum antisymmetry {antisymmetry:.3e} Hartree/Bohr^2)."
        )
    hessian = 0.5 * (hessian + hessian.T)

    model_path = Path(str(metadata["model_path"])).resolve()
    result = MacePolar1HessianResult(
        hessian_hartree_per_bohr2=hessian,
        model=config.model,
        energy_hartree=float(metadata["energy_hartree"]),
        atom_count=atom_count,
        molecular_charge=charge,
        multiplicity=multiplicity,
        python_executable_path=python,
        python_executable_sha256=_sha256(python),
        model_path=model_path,
        model_sha256=str(metadata["model_sha256"]),
        versions=dict(metadata["versions"]),
        device=config.device,
        probe_result=probe_result,
        command_result=command_result,
        hessian_path=hessian_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        probe_stdout_path=probe_stdout_path,
        probe_stderr_path=probe_stderr_path,
        worker_report_path=worker_report_path,
        report_path=work / "mace_polar1_hessian.json",
        antisymmetry_max=antisymmetry,
        gradient_norm_hartree_per_bohr=float(
            metadata["gradient_norm_hartree_per_bohr"]
        ),
        finite_difference_relative_error=float(
            metadata["finite_difference_validation"]["relative_error"]
        ),
    )
    result.report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _validate_metadata(
    metadata: dict[str, object],
    *,
    atom_count: int,
    charge: int,
    multiplicity: int,
    config: McpbMacePolar1HessianConfig,
) -> None:
    expected = {
        "status": "complete",
        "backend": "MACE-POLAR-1",
        "model": config.model,
        "atom_count": atom_count,
        "molecular_charge": charge,
        "multiplicity": multiplicity,
        "default_dtype": "float64",
        "output_hessian_units": "hartree/bohr^2",
        "geometry_optimization_performed": False,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise MacePolar1HessianError(
            "MACE-POLAR-1 worker metadata mismatch: " + "; ".join(mismatches)
        )
    model_sha = str(metadata.get("model_sha256", ""))
    if len(model_sha) != 64:
        raise MacePolar1HessianError("MACE-POLAR-1 worker omitted a valid model digest.")
    if (
        config.expected_model_sha256 is not None
        and model_sha.lower() != config.expected_model_sha256.lower()
    ):
        raise MacePolar1HessianError("MACE-POLAR-1 model digest does not match configuration.")
    relative_error = float(
        dict(metadata.get("finite_difference_validation", {})).get(
            "relative_error", float("nan")
        )
    )
    if not np.isfinite(relative_error) or (
        relative_error > config.max_hessian_vector_relative_error
    ):
        raise MacePolar1HessianError(
            "MACE-POLAR-1 Hessian-vector finite-difference validation failed."
        )


def _last_json_object(text: str, *, label: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise MacePolar1HessianError(f"Could not parse JSON from MACE-POLAR-1 {label}.")


def _resolve_python(configured: str) -> Path:
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        # Do not resolve a virtual-environment Python symlink: executing its
        # base interpreter path would silently discard that environment's
        # site-packages and therefore its pinned MACE installation.
        return candidate.absolute()
    found = which_executable(configured)
    if found is None:
        raise MacePolar1HessianError(
            f"MACE-POLAR-1 Python executable not found: {configured}"
        )
    return Path(found).absolute()


def _command_to_dict(result: CommandResult) -> dict[str, object]:
    return {
        "command": list(result.command),
        "cwd": result.cwd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "runtime_seconds": result.runtime_seconds,
    }


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
