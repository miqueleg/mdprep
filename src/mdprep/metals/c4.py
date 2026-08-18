"""Apply and verify Amber 12-6-4 C4 terms with ParmEd."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.metals.nonbonded import resolve_amberhome


class C4PostprocessError(ValueError):
    """Raised when a requested 12-6-4 topology cannot be produced."""


@dataclass(frozen=True)
class C4PostprocessRun:
    input_prmtop_backup: Path
    output_prmtop: Path
    script_path: Path
    stdout_path: Path
    stderr_path: Path
    atom_types: tuple[str, ...]
    water_model: str
    nonzero_c4_coefficients: int
    result: CommandResult

    def to_dict(self) -> dict[str, object]:
        return {
            "input_prmtop_backup": str(self.input_prmtop_backup),
            "output_prmtop": str(self.output_prmtop),
            "script_path": str(self.script_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "atom_types": list(self.atom_types),
            "water_model": self.water_model,
            "nonzero_c4_coefficients": self.nonzero_c4_coefficients,
            "command": list(self.result.command),
            "cwd": self.result.cwd,
            "returncode": self.result.returncode,
            "stdout": self.result.stdout,
            "stderr": self.result.stderr,
            "runtime_seconds": self.result.runtime_seconds,
        }


def apply_12_6_4_c4(
    prmtop: str | Path,
    *,
    atom_types: list[str],
    water_model: str,
    work_dir: str | Path,
    executable: str = "parmed",
) -> C4PostprocessRun:
    if not atom_types:
        raise C4PostprocessError("At least one 12-6-4 metal atom type is required.")
    exe = which_executable(executable)
    if exe is None:
        raise C4PostprocessError(
            "ParmEd executable is required to add LENNARD_JONES_CCOEF for a 12-6-4 ion model."
        )
    water = water_model.upper()
    if water not in {"TIP3P", "OPC"}:
        raise C4PostprocessError(f"Unsupported 12-6-4 water model: {water_model}")
    amberhome = resolve_amberhome()
    polfile = amberhome / "dat" / "leap" / "parm" / "lj_1264_pol.dat"
    if not polfile.is_file():
        raise C4PostprocessError(f"Amber 12-6-4 polarizability file is missing: {polfile}")

    topology = Path(prmtop)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    backup = work / f"{topology.name}.pre1264"
    generated = work / f"{topology.name}.with1264"
    shutil.copyfile(topology, backup)
    mask = _atom_type_mask(atom_types)
    script = work / "parmed.add1264.in"
    script.write_text(
        "\n".join(
            [
                f"add12_6_4 {mask} watermodel {water} polfile {polfile.resolve()}",
                f"outparm {generated.name}",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = run_command(
        [exe, "-O", "-p", str(backup.resolve()), "-i", script.name],
        cwd=work,
    )
    stdout_path = work / "parmed.add1264.stdout.txt"
    stderr_path = work / "parmed.add1264.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise C4PostprocessError(
            f"ParmEd add12_6_4 failed with exit code {result.returncode}; see {stdout_path} and {stderr_path}."
        )
    if not generated.is_file() or generated.stat().st_size == 0:
        raise C4PostprocessError(f"ParmEd did not produce 12-6-4 topology: {generated}")
    contents = generated.read_text(encoding="utf-8", errors="replace")
    if "%FLAG LENNARD_JONES_CCOEF" not in contents:
        raise C4PostprocessError(
            f"ParmEd output lacks LENNARD_JONES_CCOEF and is not a valid 12-6-4 topology: {generated}"
        )
    c4_values = _prmtop_flag_values(contents, "LENNARD_JONES_CCOEF")
    nonzero_c4 = sum(abs(value) > 0.0 for value in c4_values)
    if nonzero_c4 == 0:
        raise C4PostprocessError(
            f"ParmEd produced a LENNARD_JONES_CCOEF array with no nonzero terms for "
            f"{mask}; the requested metal atom type was not parameterized."
        )
    shutil.copyfile(generated, topology)
    return C4PostprocessRun(
        input_prmtop_backup=backup,
        output_prmtop=topology,
        script_path=script,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        atom_types=tuple(sorted(set(atom_types))),
        water_model=water,
        nonzero_c4_coefficients=nonzero_c4,
        result=result,
    )


def _atom_type_mask(atom_types: list[str]) -> str:
    return "@%" + ",".join(sorted(set(atom_types)))


def _prmtop_flag_values(contents: str, flag: str) -> list[float]:
    lines = contents.splitlines()
    marker = f"%FLAG {flag}"
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise C4PostprocessError(f"Amber topology is missing %FLAG {flag}.") from exc
    values: list[float] = []
    for line in lines[start:]:
        if line.startswith("%FLAG "):
            break
        if line.startswith("%") or not line.strip():
            continue
        for token in line.split():
            try:
                values.append(float(token.replace("D", "E")))
            except ValueError as exc:
                raise C4PostprocessError(
                    f"Malformed numeric value {token!r} in %FLAG {flag}."
                ) from exc
    if not values:
        raise C4PostprocessError(f"Amber topology %FLAG {flag} contains no values.")
    return values
