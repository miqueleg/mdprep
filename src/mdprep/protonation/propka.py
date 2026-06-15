"""PropKa execution workflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.protonation.propka_parser import PropkaRecord, parse_propka_file
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.writer import write_pdb
from dataclasses import replace


class PropkaExecutionError(RuntimeError):
    """Raised when PropKa execution fails or produces no pKa file."""


@dataclass(frozen=True)
class PropkaWorkflowResult:
    executable: str
    command: tuple[str, ...]
    cwd: str
    returncode: int
    runtime_seconds: float
    stdout_path: Path
    stderr_path: Path
    output_pka_path: Path
    parsed_pkas_path: Path
    records: list[PropkaRecord]

    def to_dict(self) -> dict[str, object]:
        return {
            "executable": self.executable,
            "command": list(self.command),
            "cwd": self.cwd,
            "returncode": self.returncode,
            "runtime_seconds": self.runtime_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "output_pka_path": str(self.output_pka_path),
            "parsed_pkas_path": str(self.parsed_pkas_path),
        }


def run_propka_workflow(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    work_dir: str | Path,
) -> PropkaWorkflowResult:
    propka_dir = Path(work_dir)
    propka_dir.mkdir(parents=True, exist_ok=True)
    executable = resolve_propka_executable(manifest)
    input_pdb = propka_dir / "propka_input.pdb"
    propka_structure = _canonicalized_structure_for_propka(structure, input_pdb)
    write_pdb(propka_structure, input_pdb)

    command = [executable, input_pdb.name, *manifest.protonation.propka.extra_args]
    result = run_command(command, cwd=propka_dir)
    stdout_path = propka_dir / "propka_stdout.txt"
    stderr_path = propka_dir / "propka_stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if manifest.protonation.propka.require_success and result.returncode != 0:
        raise PropkaExecutionError(
            f"PropKa failed with exit code {result.returncode}. See {stdout_path} and {stderr_path}."
        )

    pka_path = _locate_pka_file(propka_dir)
    output_pka_path = propka_dir / "propka_output.pka"
    output_pka_path.write_text(pka_path.read_text(encoding="utf-8"), encoding="utf-8")
    records = parse_propka_file(output_pka_path)
    parsed_path = propka_dir / "parsed_pkas.csv"
    _write_parsed_csv(records, parsed_path)
    return PropkaWorkflowResult(
        executable=executable,
        command=result.command,
        cwd=result.cwd,
        returncode=result.returncode,
        runtime_seconds=result.runtime_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output_pka_path=output_pka_path,
        parsed_pkas_path=parsed_path,
        records=records,
    )


def resolve_propka_executable(manifest: ManifestConfig) -> str:
    configured = manifest.protonation.propka.executable
    if configured:
        if Path(configured).exists():
            return configured
        found = which_executable(configured)
        if found:
            return found
        raise PropkaExecutionError(f"Configured PropKa executable not found: {configured}")
    for name in manifest.protonation.propka.fallback_executables:
        found = which_executable(name)
        if found:
            return found
    searched = ", ".join(manifest.protonation.propka.fallback_executables)
    raise PropkaExecutionError(f"PropKa executable not found. Searched: {searched}")


def _canonicalized_structure_for_propka(structure: PdbStructure, path: Path) -> PdbStructure:
    atoms = [replace(atom, resname=_canonical_resname(atom.resname)) for atom in structure.atoms]
    residues: list[ResidueRecord] = []
    current_key: tuple[str, str, int, str | None] | None = None
    current_atoms: list[AtomRecord] = []
    for atom in atoms:
        key = atom.residue_key
        if current_key is not None and key != current_key:
            residues.append(_residue_from_atoms(current_key, current_atoms, len(residues)))
            current_atoms = []
        current_key = key
        current_atoms.append(atom)
    if current_key is not None:
        residues.append(_residue_from_atoms(current_key, current_atoms, len(residues)))
    return PdbStructure(
        path=path,
        atoms=atoms,
        residues=residues,
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
    )


def _residue_from_atoms(
    key: tuple[str, str, int, str | None],
    atoms: list[AtomRecord],
    index: int,
) -> ResidueRecord:
    chain_id, resname, resid, icode = key
    return ResidueRecord(
        id=ResidueId(chain_id=chain_id, resname=resname, resid=resid, icode=icode),
        atoms=atoms,
        record_names={atom.record_name for atom in atoms},
        original_index=index,
    )


def _canonical_resname(resname: str) -> str:
    return {
        "ASH": "ASP",
        "GLH": "GLU",
        "HID": "HIS",
        "HIE": "HIS",
        "HIP": "HIS",
        "CYM": "CYS",
        "CYX": "CYS",
        "LYN": "LYS",
    }.get(resname, resname)


def _locate_pka_file(propka_dir: Path) -> Path:
    candidates = sorted(path for path in propka_dir.glob("*.pka") if path.name != "propka_output.pka")
    if not candidates:
        raise PropkaExecutionError(f"PropKa did not produce a .pka file in {propka_dir}")
    return candidates[0]


def _write_parsed_csv(records: list[PropkaRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["resname", "resid", "chain_id", "pka", "raw_line"])
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())

