"""xTB command construction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdprep.config.models import HistidineXtbConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command


class XtbExecutionError(RuntimeError):
    """Raised when xTB is requested but unavailable or fails."""


@dataclass(frozen=True)
class XtbRunResult:
    command_result: CommandResult
    stdout_path: Path
    stderr_path: Path


def resolve_xtb_executable(config: HistidineXtbConfig) -> str:
    configured = config.executable
    if Path(configured).exists():
        return configured
    found = which_executable(configured)
    if found:
        return found
    raise XtbExecutionError(f"xTB executable not found: {configured}")


def build_xtb_command(
    *,
    config: HistidineXtbConfig,
    xyz_path: str | Path,
    cluster_charge: int,
    executable: str | None = None,
    input_path: str | Path | None = None,
) -> list[str]:
    exe = executable or config.executable
    command = [exe, str(xyz_path)]
    extra_args = list(config.extra_args)
    if config.model == "gfn2":
        command.extend(["--gfn", "2"])
    elif "--gxtb" not in extra_args:
        command.append("--gxtb")
    if config.mode == "opt":
        command.extend(["--opt", config.opt_level])
    command.extend(["--chrg", str(cluster_charge), "--uhf", "0"])
    command.extend(["--iterations", str(config.scf_iterations)])
    if config.electronic_temperature_kelvin is not None:
        command.extend(["--etemp", str(config.electronic_temperature_kelvin)])
    if config.solvent is not None and not _extra_args_define_solvent(extra_args):
        command.extend(["--alpb", config.solvent])
    if input_path is not None:
        command.extend(["--input", str(input_path)])
    command.extend(extra_args)
    return command


def run_xtb(
    *,
    config: HistidineXtbConfig,
    xyz_path: str | Path,
    work_dir: str | Path,
    cluster_charge: int,
    stdout_path: str | Path,
    stderr_path: str | Path,
    input_path: str | Path | None = None,
) -> XtbRunResult:
    executable = resolve_xtb_executable(config)
    command = build_xtb_command(
        config=config,
        xyz_path=Path(xyz_path).name,
        cluster_charge=cluster_charge,
        executable=executable,
        input_path=Path(input_path).name if input_path is not None else None,
    )
    result = run_command(command, cwd=work_dir)
    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise XtbExecutionError(
            "xTB failed with exit code "
            f"{result.returncode}. Command: {' '.join(result.command)}. "
            f"See {stdout} and {stderr}.\n"
            f"stdout tail:\n{_tail(result.stdout)}\n"
            f"stderr tail:\n{_tail(result.stderr)}"
        )
    return XtbRunResult(command_result=result, stdout_path=stdout, stderr_path=stderr)


def _extra_args_define_solvent(extra_args: list[str]) -> bool:
    return any(arg in {"--alpb", "--gbsa"} for arg in extra_args)


def _tail(text: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    if not text:
        return "<empty>"
    lines = text.rstrip().splitlines()[-max_lines:]
    tail = "\n".join(lines)
    if len(tail) > max_chars:
        return tail[-max_chars:]
    return tail
