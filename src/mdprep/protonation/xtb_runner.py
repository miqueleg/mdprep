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
    if config.solvent is not None and not _extra_args_define_solvent(extra_args):
        command.extend(["--alpb", config.solvent])
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
) -> XtbRunResult:
    executable = resolve_xtb_executable(config)
    command = build_xtb_command(
        config=config,
        xyz_path=Path(xyz_path).name,
        cluster_charge=cluster_charge,
        executable=executable,
    )
    result = run_command(command, cwd=work_dir)
    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise XtbExecutionError(f"xTB failed with exit code {result.returncode}. See {stdout} and {stderr}.")
    return XtbRunResult(command_result=result, stdout_path=stdout, stderr_path=stderr)


def _extra_args_define_solvent(extra_args: list[str]) -> bool:
    return any(arg in {"--alpb", "--gbsa"} for arg in extra_args)

