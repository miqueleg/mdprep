"""parmchk2 wrapper for ligand frcmod generation."""

from __future__ import annotations

from pathlib import Path

from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.config.models import LigandConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import run_command


def build_parmchk2_command(
    *,
    executable: str,
    input_mol2: str | Path,
    output_frcmod: str | Path,
    ligand: LigandConfig,
) -> list[str]:
    return [
        executable,
        "-i",
        str(input_mol2),
        "-f",
        "mol2",
        "-o",
        str(output_frcmod),
        "-s",
        ligand.atom_types,
    ]


def run_parmchk2(
    *,
    ligand: LigandConfig,
    input_mol2: str | Path,
    output_frcmod: str | Path,
    work_dir: str | Path,
    executable: str = "parmchk2",
) -> AmberToolRun:
    exe = _resolve_executable(executable)
    work = Path(work_dir)
    stdout_path = work / "parmchk2_stdout.txt"
    stderr_path = work / "parmchk2_stderr.txt"
    input_path = Path(input_mol2).resolve()
    output = Path(output_frcmod).resolve()
    command = build_parmchk2_command(
        executable=exe,
        input_mol2=input_path,
        output_frcmod=output,
        ligand=ligand,
    )
    result = run_command(command, cwd=work)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AmberToolsError(
            "\n".join(
                [
                    f"parmchk2 failed with exit code {result.returncode}.",
                    f"Command: {' '.join(result.command)}",
                    f"See {stdout_path} and {stderr_path}.",
                    f"stdout tail:\n{_tail(result.stdout)}",
                    f"stderr tail:\n{_tail(result.stderr)}",
                ]
            )
        )
    if not output.exists():
        raise AmberToolsError(f"parmchk2 did not produce expected frcmod file: {output}")
    return AmberToolRun(
        command_result=result,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_path=output,
    )


def _resolve_executable(name: str) -> str:
    found = which_executable(name)
    if found:
        return found
    raise AmberToolsError(f"AmberTools executable not found: {name}")


def _tail(text: str, *, lines: int = 20) -> str:
    stripped = text.strip()
    if not stripped:
        return "<empty>"
    return "\n".join(stripped.splitlines()[-lines:])
