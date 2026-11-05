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


# GAFF2 atom types Amber's 12-6-4 polarizability table (lj_1264_pol.dat) does
# not list, each with the GAFF type whose polarizability it takes. The table
# predates GAFF2's ring and protonation variants, and ParmEd stops on the
# first type it cannot find -- c6, a ring CH2, is enough. Each variant is the
# same element in the same hybridisation as its GAFF parent, which is what a
# polarizability depends on.
GAFF2_POLARIZABILITY_ANALOGS = {
    "c5": "c3", "c6": "c3", "cs": "c",
    "n5": "n3", "n6": "n3", "n7": "n3", "n8": "n3", "n9": "n3",
    "np": "n3", "nq": "n3",
    "nu": "nh", "nv": "nh", "nm": "nh", "nn": "nh",
    "ns": "n", "nt": "n", "ni": "n", "nj": "n1",
    "nx": "n4", "ny": "n4", "nz": "n4", "n+": "n4", "nk": "n4", "nl": "n4",
    "op": "os", "oq": "os",
    "sp": "ss", "sq": "ss",
}


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
    polfile = complete_polarizability_table(
        polfile,
        _prmtop_flag_strings(topology.read_text(encoding="utf-8", errors="replace"), "AMBER_ATOM_TYPE"),
        work / "lj_1264_pol.dat",
    )
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


def complete_polarizability_table(
    source: Path, atom_types: list[str], destination: Path
) -> Path:
    """Amber's polarizability table, with any GAFF2 type it lacks added.

    Returns ``source`` itself when it already lists every type, so a topology
    of protein and GAFF types runs against the table as distributed.
    """

    lines = source.read_text(encoding="utf-8").splitlines()
    values = {
        line.split()[0]: line.split()[1]
        for line in lines
        if line.strip() and len(line.split()) >= 2
    }
    added = [
        f"{atom_type:<5} {values[GAFF2_POLARIZABILITY_ANALOGS[atom_type]]}    "
        f"Taken from GAFF type {GAFF2_POLARIZABILITY_ANALOGS[atom_type]} by mdprep"
        for atom_type in sorted(set(atom_types))
        if atom_type not in values
        and GAFF2_POLARIZABILITY_ANALOGS.get(atom_type) in values
    ]
    if not added:
        return source
    destination.write_text("\n".join([*lines, *added]) + "\n", encoding="utf-8")
    return destination


def _prmtop_flag_strings(contents: str, flag: str) -> list[str]:
    lines = contents.splitlines()
    marker = f"%FLAG {flag}"
    # tleap pads every line to 80 columns, flags included.
    starts = [index for index, line in enumerate(lines) if line.rstrip() == marker]
    if not starts:
        return []
    values: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("%FLAG "):
            break
        if line.startswith("%"):
            continue
        # Fixed 4-character fields (20a4), a type may contain no spaces.
        values.extend(line[index : index + 4].strip() for index in range(0, len(line), 4))
    return [value for value in values if value]


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
