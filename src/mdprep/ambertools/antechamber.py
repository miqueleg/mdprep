"""Antechamber wrapper for ligand AM1-BCC parameterization."""

from __future__ import annotations

from pathlib import Path

from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.config.models import LigandConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import run_command


def build_antechamber_command(
    *,
    executable: str,
    input_pdb: str | Path,
    output_mol2: str | Path,
    residue_name: str,
    ligand: LigandConfig,
) -> list[str]:
    return [
        executable,
        "-i",
        str(input_pdb),
        "-fi",
        "pdb",
        "-o",
        str(output_mol2),
        "-fo",
        "mol2",
        "-rn",
        residue_name,
        "-nc",
        str(ligand.net_charge),
        "-m",
        str(ligand.multiplicity),
        "-c",
        "bcc",
        "-at",
        ligand.atom_types,
        "-s",
        "2",
    ]


def run_antechamber(
    *,
    ligand: LigandConfig,
    input_pdb: str | Path,
    output_mol2: str | Path,
    residue_name: str,
    work_dir: str | Path,
    executable: str = "antechamber",
) -> AmberToolRun:
    exe = _resolve_executable(executable)
    work = Path(work_dir)
    stdout_path = work / "antechamber_stdout.txt"
    stderr_path = work / "antechamber_stderr.txt"
    output = Path(output_mol2)
    command = build_antechamber_command(
        executable=exe,
        input_pdb=input_pdb,
        output_mol2=output,
        residue_name=residue_name,
        ligand=ligand,
    )
    result = run_command(command, cwd=work)
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise AmberToolsError(
            f"antechamber failed with exit code {result.returncode}. See {stdout_path} and {stderr_path}."
        )
    if not output.exists():
        raise AmberToolsError(f"antechamber did not produce expected mol2 file: {output}")
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

