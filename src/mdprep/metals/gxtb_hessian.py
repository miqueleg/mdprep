"""Fixed-geometry GFN2-xTB/g-xTB Hessians for MCPB.py Seminario parameters."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdprep.config.models import McpbXtbHessianConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command


class GxtbHessianError(RuntimeError):
    """Raised when xTB cannot produce a validated Cartesian Hessian."""


@dataclass(frozen=True)
class GxtbHessianResult:
    """Validated result and complete external-command provenance."""

    hessian_hartree_per_bohr2: np.ndarray
    model: str
    energy_hartree: float
    atom_count: int
    molecular_charge: int
    unpaired_electrons: int
    executable_path: Path
    executable_sha256: str
    release_tag: str | None
    version_text: str
    version_result: CommandResult
    command_result: CommandResult
    hessian_path: Path
    stdout_path: Path
    stderr_path: Path
    version_stdout_path: Path
    version_stderr_path: Path
    report_path: Path
    antisymmetry_max: float
    gradient_norm_hartree_per_bohr: float | None
    raw_hessian_asymmetry_warning_count: int
    unoptimized_geometry_warning: bool
    # The ways into the SCF tried before the Hessian, the lowest first among
    # those that converged; empty when no search was asked for.
    scf_starts: tuple["ScfStart", ...] = ()

    def to_dict(self) -> dict[str, object]:
        backend = "GFN2-xTB" if self.model == "gfn2" else "g-xTB"
        return {
            "backend": backend,
            "model": self.model,
            "release_tag": self.release_tag,
            "version_text": self.version_text,
            "executable_path": str(self.executable_path),
            "executable_sha256": self.executable_sha256,
            "energy_hartree": self.energy_hartree,
            "atom_count": self.atom_count,
            "molecular_charge": self.molecular_charge,
            "unpaired_electrons": self.unpaired_electrons,
            "multiplicity": self.unpaired_electrons + 1,
            "hessian_path": str(self.hessian_path),
            "hessian_sha256": _sha256(self.hessian_path),
            "hessian_antisymmetry_max": self.antisymmetry_max,
            "gradient_norm_hartree_per_bohr": self.gradient_norm_hartree_per_bohr,
            "raw_hessian_asymmetry_warning_count": (
                self.raw_hessian_asymmetry_warning_count
            ),
            "unoptimized_geometry_warning": self.unoptimized_geometry_warning,
            "scf_starts": [start.to_dict() for start in self.scf_starts],
            "version_command": _command_to_dict(self.version_result),
            "hessian_command": _command_to_dict(self.command_result),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "version_stdout_path": str(self.version_stdout_path),
            "version_stderr_path": str(self.version_stderr_path),
            "report_path": str(self.report_path),
            "geometry_optimization_performed": False,
            "hessian_interpretation": (
                "Full unweighted Cartesian Hessian in Hartree/Bohr^2, evaluated "
                "at the input geometry by numerical differentiation of analytic "
                f"{backend} gradients."
            ),
        }


@dataclass(frozen=True)
class ScfStart:
    """One way into the SCF before the Hessian, and where it ended."""

    label: str
    # The electronic temperature converged at first and cooled from; None for
    # the start made straight at the Hessian's own temperature.
    annealed_from_kelvin: float | None
    converged: bool
    energy_hartree: float | None
    work_dir: Path
    commands: tuple[CommandResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "annealed_from_kelvin": self.annealed_from_kelvin,
            "converged": self.converged,
            "energy_hartree": self.energy_hartree,
            "work_dir": str(self.work_dir),
            "commands": [_command_to_dict(command) for command in self.commands],
        }


# The Hessian's own SCF starts from the chosen state's wavefunction and has to
# end where it began. The states 2QFK's starts reached lie 1 to 45 mEh apart.
SCF_STATE_ENERGY_TOLERANCE_HARTREE = 1.0e-5


def build_gxtb_hessian_command(
    *,
    executable: str,
    xyz_name: str,
    charge: int,
    spin: int,
    accuracy: float,
    model: str = "gxtb",
    max_scf_iterations: int | None = None,
    electronic_temperature_kelvin: float | None = None,
    hessian: bool = True,
) -> list[str]:
    """Build the fixed-geometry command without any optimization run type.

    ``hessian`` false gives the single point the SCF search runs before it.
    """

    if spin < 0:
        raise GxtbHessianError("g-xTB UHF count must be non-negative.")
    if not np.isfinite(accuracy) or accuracy <= 0:
        raise GxtbHessianError("g-xTB accuracy must be positive and finite.")
    if model not in {"gfn2", "gxtb"}:
        raise GxtbHessianError(f"Unsupported xTB Hessian model: {model}")
    if max_scf_iterations is not None and max_scf_iterations < 1:
        raise GxtbHessianError("xTB maximum SCF iterations must be positive.")
    if electronic_temperature_kelvin is not None and not (
        np.isfinite(electronic_temperature_kelvin) and electronic_temperature_kelvin > 0
    ):
        raise GxtbHessianError("xTB electronic temperature must be positive and finite.")
    model_args = ["--gfn", "2"] if model == "gfn2" else ["--gxtb"]
    command = [
        executable,
        xyz_name,
        *model_args,
        "--chrg",
        str(charge),
        "--uhf",
        str(spin),
        *(["--hess"] if hessian else []),
        "--acc",
        f"{accuracy:.12g}",
    ]
    if max_scf_iterations is not None:
        command.extend(["--iterations", str(max_scf_iterations)])
    if electronic_temperature_kelvin is not None:
        command.extend(["--etemp", f"{electronic_temperature_kelvin:g}"])
    return command


def run_gxtb_hessian(
    *,
    xyz_path: str | Path,
    atom_count: int,
    charge: int,
    spin: int,
    config: McpbXtbHessianConfig,
    work_dir: str | Path,
) -> GxtbHessianResult:
    """Run a serial-safe g-xTB Hessian and validate its units and dimensions."""

    if atom_count < 1:
        raise GxtbHessianError("g-xTB Hessian requires at least one atom.")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    source_xyz = Path(xyz_path)
    if source_xyz.resolve().parent != work.resolve():
        raise GxtbHessianError(
            "g-xTB input XYZ must already be staged in its dedicated working directory."
        )
    if not source_xyz.is_file() or source_xyz.stat().st_size == 0:
        raise GxtbHessianError(f"g-xTB input XYZ is missing or empty: {source_xyz}")
    backend = "GFN2-xTB" if config.model == "gfn2" else "g-xTB"
    file_prefix = "gfn2_xtb" if config.model == "gfn2" else "gxtb"
    if sys.platform == "darwin" and config.model == "gxtb" and config.num_threads != 1:
        raise GxtbHessianError(
            "The official g-xTB macOS binary does not support reliable parallel "
            "numerical Hessians; set gxtb.num_threads: 1."
        )

    executable = _resolve_executable(config.executable, model=config.model)
    executable_sha256 = _sha256(executable)
    if (
        config.expected_executable_sha256 is not None
        and executable_sha256.lower() != config.expected_executable_sha256.lower()
    ):
        raise GxtbHessianError(
            "g-xTB executable checksum mismatch: expected "
            f"{config.expected_executable_sha256}, found {executable_sha256} for {executable}."
        )

    environment = dict(os.environ)
    thread_count = str(config.num_threads)
    environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
        }
    )
    version_result = run_command(
        [str(executable), "--version"],
        cwd=work,
        env=environment,
    )
    version_stdout_path = work / f"{file_prefix}.version.stdout.txt"
    version_stderr_path = work / f"{file_prefix}.version.stderr.txt"
    version_stdout_path.write_text(version_result.stdout, encoding="utf-8")
    version_stderr_path.write_text(version_result.stderr, encoding="utf-8")
    if version_result.returncode != 0:
        raise GxtbHessianError(
            f"g-xTB version command failed with exit code {version_result.returncode}. "
            f"See {version_stdout_path} and {version_stderr_path}."
        )
    version_text = _parse_version(version_result.stdout + "\n" + version_result.stderr)

    _remove_stale_gxtb_outputs(work)
    scf_starts: tuple[ScfStart, ...] = ()
    if config.scf_annealing_kelvin:
        scf_starts = _search_scf_states(
            executable=executable,
            source_xyz=source_xyz,
            work=work,
            charge=charge,
            spin=spin,
            config=config,
            environment=environment,
            file_prefix=file_prefix,
        )
        # xTB reads the wavefunction it finds beside the input.
        shutil.copyfile(scf_starts[0].work_dir / "xtbrestart", work / "xtbrestart")
    hessian_path = work / "hessian"
    command = build_gxtb_hessian_command(
        executable=str(executable),
        xyz_name=source_xyz.name,
        charge=charge,
        spin=spin,
        accuracy=config.accuracy,
        model=config.model,
        max_scf_iterations=config.max_scf_iterations,
        electronic_temperature_kelvin=config.electronic_temperature_kelvin,
    )
    command_result = run_command(command, cwd=work, env=environment)
    stdout_path = work / f"{file_prefix}.hessian.stdout.txt"
    stderr_path = work / f"{file_prefix}.hessian.stderr.txt"
    stdout_path.write_text(command_result.stdout, encoding="utf-8")
    stderr_path.write_text(command_result.stderr, encoding="utf-8")
    if command_result.returncode != 0:
        hint = ""
        if "SCF not converged" in command_result.stdout:
            hint = (
                " Its SCF did not converge. More cycles rarely help an SCF that "
                "oscillates; smearing the occupations does. Set "
                "mcpb.pyscf.xtb.electronic_temperature_kelvin (1000 converges a "
                "low-spin Fe(III) heme in seconds)."
                if config.electronic_temperature_kelvin is None
                else " Its SCF did not converge, even at an electronic "
                f"temperature of {config.electronic_temperature_kelvin:g} K. A "
                "high-spin heme can take a thousand cycles or more from a cold "
                "start: raise mcpb.pyscf.xtb.max_scf_iterations, and set "
                "scf_annealing_kelvin to start it from a converged hotter SCF "
                "as well (see docs/metals.md)."
            )
        raise GxtbHessianError(
            f"{backend} Hessian failed with exit code {command_result.returncode}.{hint} "
            f"See {stdout_path} and {stderr_path}."
        )
    if config.model == "gxtb" and "for g-xTB:" not in command_result.stdout:
        raise GxtbHessianError(
            "The selected executable completed but did not identify g-xTB support. "
            "Install the official g-xTB build and retain the --gxtb command flag."
        )
    if config.model == "gfn2" and "gfn2-xtb" not in command_result.stdout.lower():
        raise GxtbHessianError(
            "The selected executable completed but did not identify GFN2-xTB. "
            "Use an xTB build supporting '--gfn 2'."
        )
    combined_output = command_result.stdout + "\n" + command_result.stderr
    if "normal termination of xtb" not in combined_output.lower():
        raise GxtbHessianError(
            f"{backend} did not report normal termination despite returning exit code zero."
        )
    _validate_open_shell_metadata(
        command_result.stdout,
        expected_charge=charge,
        expected_spin=spin,
        model=config.model,
    )
    hessian = parse_gxtb_hessian(hessian_path, atom_count=atom_count)
    antisymmetry = float(np.max(np.abs(hessian - hessian.T)))
    if antisymmetry > 1.0e-5:
        raise GxtbHessianError(
            "g-xTB Cartesian Hessian is not symmetric "
            f"(maximum antisymmetry {antisymmetry:.3e})."
        )
    hessian = 0.5 * (hessian + hessian.T)
    energy = _parse_total_energy(command_result.stdout)
    if scf_starts and (
        abs(energy - float(scf_starts[0].energy_hartree))  # type: ignore[arg-type]
        > SCF_STATE_ENERGY_TOLERANCE_HARTREE
    ):
        raise GxtbHessianError(
            f"The {backend} Hessian's SCF started from the lowest state the search "
            f"found ({scf_starts[0].label}, {scf_starts[0].energy_hartree:.8f} Eh) "
            f"and ended in another ({energy:.8f} Eh). See {stdout_path}."
        )
    gradient_norm = _parse_gradient_norm(command_result.stdout)
    raw_asymmetry_warning_count = len(
        re.findall(
            r"Hessian element\s+\d+\s+\d+\s+is not symmetric",
            combined_output,
            flags=re.IGNORECASE,
        )
    )
    unoptimized_geometry_warning = (
        "hessian on incompletely optimized geometry" in combined_output.lower()
    )

    report_path = work / f"{file_prefix}_hessian.json"
    result = GxtbHessianResult(
        hessian_hartree_per_bohr2=hessian,
        model=config.model,
        energy_hartree=energy,
        atom_count=atom_count,
        molecular_charge=charge,
        unpaired_electrons=spin,
        executable_path=executable,
        executable_sha256=executable_sha256,
        release_tag=config.release_tag,
        version_text=version_text,
        version_result=version_result,
        command_result=command_result,
        hessian_path=hessian_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        version_stdout_path=version_stdout_path,
        version_stderr_path=version_stderr_path,
        report_path=report_path,
        antisymmetry_max=antisymmetry,
        gradient_norm_hartree_per_bohr=gradient_norm,
        raw_hessian_asymmetry_warning_count=raw_asymmetry_warning_count,
        unoptimized_geometry_warning=unoptimized_geometry_warning,
        scf_starts=scf_starts,
    )
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _search_scf_states(
    *,
    executable: Path,
    source_xyz: Path,
    work: Path,
    charge: int,
    spin: int,
    config: McpbXtbHessianConfig,
    environment: dict[str, str],
    file_prefix: str = "gxtb",
) -> tuple[ScfStart, ...]:
    """Converge the SCF from each start; the lowest converged one first.

    Single-threaded, so that which state a start reaches is decided by the
    start and not by how the sums were split between threads. Each single
    point's output is kept in its start's folder, as the Hessian's is beside
    them, so that a run stopped later still shows what every start reached.
    """

    serial = dict(environment)
    serial.update(
        {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    )

    def single_point(kelvin: float | None) -> list[str]:
        return build_gxtb_hessian_command(
            executable=str(executable),
            xyz_name=source_xyz.name,
            charge=charge,
            spin=spin,
            accuracy=config.accuracy,
            model=config.model,
            max_scf_iterations=config.max_scf_iterations,
            electronic_temperature_kelvin=kelvin,
            hessian=False,
        )

    starts: list[ScfStart] = []
    for hot in [None, *config.scf_annealing_kelvin]:
        label = "direct" if hot is None else f"from {hot:g} K"
        start_dir = work / ("scf_start_direct" if hot is None else f"scf_start_{hot:g}K")
        start_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_gxtb_outputs(start_dir)
        shutil.copyfile(source_xyz, start_dir / source_xyz.name)
        # Hot first, if annealed; then at the Hessian's temperature, reading
        # the wavefunction the hot SCF left.
        steps = ([hot] if hot is not None else []) + [config.electronic_temperature_kelvin]
        commands: list[CommandResult] = []
        for kelvin in steps:
            result = run_command(single_point(kelvin), cwd=start_dir, env=serial)
            commands.append(result)
            stem = f"{file_prefix}.sp_{'xtb_default' if kelvin is None else f'{kelvin:g}K'}"
            (start_dir / f"{stem}.stdout.txt").write_text(result.stdout, encoding="utf-8")
            (start_dir / f"{stem}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            if not _terminated_normally(result):
                break
        converged = len(commands) == len(steps) and _terminated_normally(commands[-1])
        energy = _parse_total_energy(commands[-1].stdout) if converged else None
        if converged and not (start_dir / "xtbrestart").is_file():
            raise GxtbHessianError(
                f"xTB converged the SCF {label} but left no wavefunction to start "
                f"the Hessian from in {start_dir}."
            )
        starts.append(
            ScfStart(
                label=label,
                annealed_from_kelvin=hot,
                converged=converged,
                energy_hartree=energy,
                work_dir=start_dir,
                commands=tuple(commands),
            )
        )
    found = sorted(
        (start for start in starts if start.converged),
        key=lambda start: float(start.energy_hartree),  # type: ignore[arg-type]
    )
    if not found:
        raise GxtbHessianError(
            "The SCF did not converge from any start ("
            + ", ".join(start.label for start in starts)
            + f"). Raise mcpb.pyscf.xtb.max_scf_iterations; see {work}/scf_start_*."
        )
    return tuple(found) + tuple(start for start in starts if not start.converged)


def _terminated_normally(result: CommandResult) -> bool:
    return result.returncode == 0 and "normal termination of xtb" in (
        result.stdout + "\n" + result.stderr
    ).lower()


def parse_gxtb_hessian(path: str | Path, *, atom_count: int) -> np.ndarray:
    """Parse xTB's full row-major ``$hessian`` matrix in atomic units."""

    hessian_path = Path(path)
    if not hessian_path.is_file() or hessian_path.stat().st_size == 0:
        raise GxtbHessianError(f"g-xTB did not produce a non-empty Hessian: {hessian_path}")
    lines = hessian_path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(
            index for index, line in enumerate(lines) if line.strip().lower() == "$hessian"
        )
    except StopIteration as exc:
        raise GxtbHessianError(
            f"g-xTB Hessian file has no $hessian section: {hessian_path}"
        ) from exc
    values: list[float] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$"):
            break
        for token in stripped.split():
            try:
                values.append(float(token.replace("D", "E").replace("d", "e")))
            except ValueError as exc:
                raise GxtbHessianError(
                    f"Invalid numeric token {token!r} in g-xTB Hessian {hessian_path}."
                ) from exc
    dimension = atom_count * 3
    expected = dimension * dimension
    if len(values) != expected:
        raise GxtbHessianError(
            f"g-xTB Hessian contains {len(values)} values; expected {expected} "
            f"for {atom_count} atoms."
        )
    matrix = np.asarray(values, dtype=float).reshape((dimension, dimension))
    if not np.all(np.isfinite(matrix)):
        raise GxtbHessianError("g-xTB Cartesian Hessian contains non-finite values.")
    return matrix


# Where to look for a g-xTB build when the manifest does not name one. The
# conda xtb package is GFN-only, so an unqualified "xtb" on PATH is usually
# the wrong binary and the run would fail later with a less obvious message.
GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE = "MDPREP_GXTB_EXECUTABLE"


def gxtb_executable_candidates(configured: str | None, model: str) -> list[str]:
    """Where to look for the binary, in order, for this Hessian model."""

    if configured is not None:
        return [configured]
    if model != "gxtb":
        return ["xtb"]
    from os import environ

    candidates: list[str] = []
    from_environment = environ.get(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE)
    if from_environment:
        candidates.append(from_environment)
    # A g-xTB build is often installed beside the GFN one rather than on PATH.
    candidates.append("g-xtb")
    candidates.append("xtb")
    return candidates


def _resolve_executable(configured: str | None, *, model: str = "gxtb") -> Path:
    candidates = gxtb_executable_candidates(configured, model)
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
        found = which_executable(candidate)
        if found is not None:
            return Path(found).resolve()
    if configured is not None:
        raise GxtbHessianError(f"g-xTB executable not found: {configured}")
    raise GxtbHessianError(
        "No g-xTB executable found. g-xTB is a separate build from the xtb "
        "that conda installs, which is GFN-only. Download it from "
        "https://github.com/grimme-lab/g-xtb/releases, then either set "
        f"{GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE} to it or give "
        "mcpb.pyscf.xtb.executable its path. To use GFN2 instead, set "
        "mcpb.pyscf.xtb.model: gfn2 -- but check the force constants, because "
        "GFN2 describes transition-metal coordination poorly."
    )


def _remove_stale_gxtb_outputs(work_dir: Path) -> None:
    """Prevent an earlier xTB restart or Hessian from affecting a rerun."""

    for name in (
        "charges",
        "energy",
        "g98.out",
        "gradient",
        "hessian",
        "vibspectrum",
        "wbo",
        "xtbhess.coord",
        "xtbhess.xyz",
        "xtbrestart",
        "xtbtopo.mol",
    ):
        (work_dir / name).unlink(missing_ok=True)


def _parse_total_energy(stdout: str) -> float:
    matches = re.findall(
        r"\btotal\s+energy\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+Eh\b",
        stdout,
        flags=re.IGNORECASE,
    )
    if not matches:
        raise GxtbHessianError("Could not parse the final g-xTB total energy.")
    value = float(matches[-1].replace("D", "E").replace("d", "e"))
    if not np.isfinite(value):
        raise GxtbHessianError("g-xTB total energy is not finite.")
    return value


def _parse_version(text: str) -> str:
    match = re.search(r"\bxtb version\s+([^\r\n]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise GxtbHessianError("Could not parse the xTB executable version.")
    return match.group(1).strip()


def _parse_gradient_norm(stdout: str) -> float | None:
    matches = re.findall(
        r"gradient norm\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+Eh/(?:a0|α)",
        stdout,
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    value = float(matches[-1].replace("D", "E").replace("d", "e"))
    return value if np.isfinite(value) else None


def _validate_open_shell_metadata(
    stdout: str,
    *,
    expected_charge: int,
    expected_spin: int,
    model: str = "gxtb",
) -> None:
    charge_match = re.search(
        r"molecular charge\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+e",
        stdout,
        flags=re.IGNORECASE,
    )
    if charge_match is None:
        charge_match = re.search(
            r"net charge\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\b",
            stdout,
            flags=re.IGNORECASE,
        )
    spin_match = re.search(
        r"number of unpaired electrons\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s+e",
        stdout,
        flags=re.IGNORECASE,
    )
    if spin_match is None:
        spin_match = re.search(
            r"unpaired electrons\s+([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\b",
            stdout,
            flags=re.IGNORECASE,
        )
    # The setup summary's own line. A run started from a converged SCF says
    # "Wavefunction read from restart file" before it.
    wavefunction_match = re.search(
        r"^\s*wavefunction\s+([A-Za-z-]+)\s*$",
        stdout,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if charge_match is None or spin_match is None:
        raise GxtbHessianError(
            "Could not validate xTB charge and unpaired-electron count."
        )
    actual_charge = float(charge_match.group(1).replace("D", "E").replace("d", "e"))
    actual_spin = float(spin_match.group(1).replace("D", "E").replace("d", "e"))
    if abs(actual_charge - expected_charge) > 1.0e-6:
        raise GxtbHessianError(
            f"g-xTB reported charge {actual_charge}, expected {expected_charge}."
        )
    if abs(actual_spin - expected_spin) > 1.0e-6:
        raise GxtbHessianError(
            f"g-xTB reported {actual_spin} unpaired electrons, expected {expected_spin}."
        )
    if (
        model == "gxtb"
        and wavefunction_match is not None
        and wavefunction_match.group(1).lower() != "unrestricted"
    ):
        raise GxtbHessianError(
            "g-xTB did not report the required unrestricted wavefunction."
        )


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
