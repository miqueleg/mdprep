"""Execute the ASH QM/MM refinement in an explicitly selected Python environment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from mdprep.config.models import AshRefinementConfig, MacePolar1RefinementConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.refinement.selection import RefinementSelection


class AshRefinementError(RuntimeError):
    """Raised when the requested ASH QM/MM refinement cannot complete safely."""


@dataclass(frozen=True)
class AshRefinementCache:
    """Validated inputs and outputs captured before a run is resumed."""

    input_payload: dict[str, object]
    prmtop_sha256: str
    inpcrd_sha256: str
    output_text: str
    stdout: str
    stderr: str
    command_record: dict[str, object]


@dataclass(frozen=True)
class AshRefinementRun:
    command_result: CommandResult
    driver_path: Path
    input_path: Path
    output_path: Path
    stdout_path: Path
    stderr_path: Path
    command_record_path: Path
    python_executable: str
    qm_method: str
    embedding: str
    xtb_executable: str | None
    mace_model_path: str | None
    mace_model_sha256: str | None
    optimized_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    final_energy_hartree: float | None
    reused_cached_result: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command_result.command),
            "cwd": self.command_result.cwd,
            "returncode": self.command_result.returncode,
            "stdout": self.command_result.stdout,
            "stderr": self.command_result.stderr,
            "runtime_seconds": self.command_result.runtime_seconds,
            "driver_path": str(self.driver_path),
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "command_record_path": str(self.command_record_path),
            "python_executable": self.python_executable,
            "qm_method": self.qm_method,
            "embedding": self.embedding,
            "xtb_executable": self.xtb_executable,
            "mace_model_path": self.mace_model_path,
            "mace_model_sha256": self.mace_model_sha256,
            "final_energy_hartree": self.final_energy_hartree,
            "optimized_atom_count": len(self.optimized_coordinates_angstrom),
            "reused_cached_result": self.reused_cached_result,
        }


def capture_ash_refinement_cache(work_dir: str | Path) -> AshRefinementCache:
    """Capture a converged result and hashes of the exact topology it used."""

    work = Path(work_dir)
    metadata_path = work / "ash_refinement_cache.json"
    output_path = work / "ash_refinement_output.json"
    command_record_path = work / "ash_command.json"
    stdout_path = work / "ash_stdout.txt"
    stderr_path = work / "ash_stderr.txt"
    required = (metadata_path, output_path, command_record_path, stdout_path, stderr_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AshRefinementError(
            "Cannot resume ASH refinement because cached artifacts are missing: "
            f"{missing}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1:
            raise AshRefinementError(
                "Cannot resume ASH refinement because its immutable cache metadata "
                "has an unsupported schema"
            )
        payload = metadata["input_payload"]
        output_text = output_path.read_text(encoding="utf-8")
        output = json.loads(output_text)
        command_record = json.loads(command_record_path.read_text(encoding="utf-8"))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AshRefinementError("Cached ASH artifacts are malformed and cannot be resumed") from exc
    if output.get("status") != "converged":
        raise AshRefinementError(
            "Cannot resume ASH refinement because the cached output is not marked converged"
        )
    artifact_hashes = {
        "output_sha256": _sha256(output_path),
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "command_record_sha256": _sha256(command_record_path),
    }
    for label, observed in artifact_hashes.items():
        if metadata.get(label) != observed:
            raise AshRefinementError(
                "Cannot resume ASH refinement because a cached artifact no longer "
                f"matches immutable metadata: {label}"
            )
    return AshRefinementCache(
        input_payload=payload,
        prmtop_sha256=str(metadata["prmtop_scientific_sha256"]),
        inpcrd_sha256=str(metadata["inpcrd_sha256"]),
        output_text=output_text,
        stdout=stdout_path.read_text(encoding="utf-8"),
        stderr=stderr_path.read_text(encoding="utf-8"),
        command_record=command_record,
    )


def run_ash_refinement(
    *,
    prmtop_path: str | Path,
    inpcrd_path: str | Path,
    selection: RefinementSelection,
    config: AshRefinementConfig,
    work_dir: str | Path,
    qm_method: str = "gfn2_xtb",
    embedding: str = "electrostatic",
    mace_polar1: MacePolar1RefinementConfig | None = None,
    cached_run: AshRefinementCache | None = None,
) -> AshRefinementRun:
    """Run a validated ASH/OpenMM active-region QM/MM optimization."""

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    python_executable = _resolve_executable(
        config.python_executable,
        label="ASH Python executable",
    )
    if qm_method not in {"gfn2_xtb", "gxtb", "mace_polar1"}:
        raise AshRefinementError(f"Unsupported refinement QM method: {qm_method}")
    expected_embedding = "electrostatic" if qm_method == "gfn2_xtb" else "mechanical"
    if embedding != expected_embedding:
        raise AshRefinementError(
            f"Refinement method {qm_method} requires {expected_embedding} embedding"
        )
    xtb_executable: Path | None = None
    mace_model_path: Path | None = None
    mace_model_sha256: str | None = None
    if qm_method in {"gfn2_xtb", "gxtb"}:
        xtb_executable = Path(
            _resolve_executable(config.xtb_executable, label="xTB executable")
        ).resolve()
        ash_xtb = xtb_executable.parent / "xtb"
        if not ash_xtb.is_file():
            raise AshRefinementError(
                "ASH's xTB interface requires an executable named 'xtb' in its configured "
                f"directory, but none exists at {ash_xtb}"
            )
    else:
        if mace_polar1 is None:
            raise AshRefinementError(
                "MACE-POLAR-1 refinement requires explicit model settings"
            )
        mace_model_path = Path(mace_polar1.model_path).expanduser().resolve()
        if not mace_model_path.is_file() or mace_model_path.stat().st_size == 0:
            raise AshRefinementError(
                f"MACE-POLAR-1 checkpoint is missing or empty: {mace_model_path}"
            )
        mace_model_sha256 = _sha256(mace_model_path)
        if mace_model_sha256.lower() != mace_polar1.expected_model_sha256.lower():
            raise AshRefinementError(
                "MACE-POLAR-1 checkpoint checksum mismatch: expected "
                f"{mace_polar1.expected_model_sha256.lower()}, found {mace_model_sha256}"
            )
        missing_search_paths = [
            str(Path(item).expanduser())
            for item in mace_polar1.python_search_paths
            if not Path(item).expanduser().is_dir()
        ]
        if missing_search_paths:
            raise AshRefinementError(
                "MACE-POLAR-1 Python search paths do not exist or are not directories: "
                f"{missing_search_paths}"
            )

    prmtop = Path(prmtop_path).resolve()
    inpcrd = Path(inpcrd_path).resolve()
    for label, path in (("prmtop", prmtop), ("inpcrd", inpcrd)):
        if not path.is_file() or path.stat().st_size == 0:
            raise AshRefinementError(f"Provisional {label} file is missing or empty: {path}")
    if not selection.active_atom_indices:
        raise AshRefinementError("QM/MM refinement requires at least one movable atom")
    invalid_indices = sorted(
        {
            index
            for index in (*selection.qm_atom_indices, *selection.active_atom_indices)
            if index < 0 or index >= selection.topology_atom_count
        }
    )
    if invalid_indices:
        raise AshRefinementError(
            "QM or movable atom indices fall outside the provisional topology: "
            f"{invalid_indices}"
        )

    driver_path = work / "run_ash_refinement.py"
    input_path = work / "ash_refinement_input.json"
    output_path = work / "ash_refinement_output.json"
    stdout_path = work / "ash_stdout.txt"
    stderr_path = work / "ash_stderr.txt"
    command_record_path = work / "ash_command.json"
    driver_path.write_text(_ASH_DRIVER, encoding="utf-8")
    payload = {
        "prmtop_path": str(prmtop),
        "inpcrd_path": str(inpcrd),
        "qm_method": qm_method,
        "embedding": embedding,
        "xtb_directory": None if xtb_executable is None else str(xtb_executable.parent),
        "mace": None
        if mace_polar1 is None
        else {
            "model": mace_polar1.model,
            "model_path": str(mace_model_path),
            "model_sha256": mace_model_sha256,
            "device": mace_polar1.device,
            "python_search_paths": [
                str(Path(item).expanduser().resolve())
                for item in mace_polar1.python_search_paths
            ],
        },
        "qm_atom_indices": list(selection.qm_atom_indices),
        "active_atom_indices": list(selection.active_atom_indices),
        "qm_boundary_excluded_atom_indices": list(
            selection.qm_boundary_excluded_atom_indices
        ),
        "qm_charge": selection.total_qm_charge,
        "qm_multiplicity": selection.total_qm_multiplicity,
        "allow_unusual_link_boundaries": config.allow_unusual_link_boundaries,
        "max_iterations": config.max_iterations,
        "num_cores": config.num_cores,
        "platform": config.platform,
        "xtb_max_iterations": config.xtb_max_iterations,
        "electronic_temperature_kelvin": config.electronic_temperature_kelvin,
        "accuracy": config.accuracy,
    }
    input_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reused_cached_result = cached_run is not None
    if cached_run is not None:
        mismatches: list[str] = []
        if payload != cached_run.input_payload:
            mismatches.append("ASH input payload")
        if _amber_prmtop_scientific_sha256(prmtop) != cached_run.prmtop_sha256:
            mismatches.append("provisional prmtop")
        if _sha256(inpcrd) != cached_run.inpcrd_sha256:
            mismatches.append("provisional inpcrd")
        if mismatches:
            raise AshRefinementError(
                "Refusing to reuse cached ASH refinement because regenerated inputs differ: "
                + ", ".join(mismatches)
            )
        output_path.write_text(cached_run.output_text, encoding="utf-8")
        stdout_path.write_text(cached_run.stdout, encoding="utf-8")
        stderr_path.write_text(cached_run.stderr, encoding="utf-8")
        command_record_path.write_text(
            json.dumps(cached_run.command_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            command_result = CommandResult(
                command=tuple(str(item) for item in cached_run.command_record["command"]),
                cwd=str(cached_run.command_record["cwd"]),
                returncode=int(cached_run.command_record["returncode"]),
                stdout=str(cached_run.command_record["stdout"]),
                stderr=str(cached_run.command_record["stderr"]),
                runtime_seconds=float(cached_run.command_record["runtime_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AshRefinementError("Cached ASH command record is malformed") from exc
    else:
        command_result = run_command(
            [python_executable, driver_path.name, input_path.name, output_path.name],
            cwd=work,
        )
        stdout_path.write_text(command_result.stdout, encoding="utf-8")
        stderr_path.write_text(command_result.stderr, encoding="utf-8")
        command_record_path.write_text(
            json.dumps(
                {
                    "command": list(command_result.command),
                    "cwd": command_result.cwd,
                    "returncode": command_result.returncode,
                    "stdout": command_result.stdout,
                    "stderr": command_result.stderr,
                    "runtime_seconds": command_result.runtime_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if command_result.returncode != 0:
        raise AshRefinementError(
            "ASH QM/MM refinement failed with exit code "
            f"{command_result.returncode}. See {stdout_path} and {stderr_path}. "
            f"stderr tail:\n{_tail(command_result.stderr)}"
        )
    if not output_path.is_file():
        raise AshRefinementError(
            f"ASH returned successfully but did not write {output_path}"
        )
    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        raw_coordinates = output["coordinates_angstrom"]
        coordinates = tuple(
            (float(item[0]), float(item[1]), float(item[2]))
            for item in raw_coordinates
        )
    except (KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        raise AshRefinementError(
            f"ASH produced an invalid optimized-coordinate file: {output_path}"
        ) from exc
    if len(coordinates) != selection.topology_atom_count:
        raise AshRefinementError(
            "ASH optimized-coordinate count does not match the provisional topology: "
            f"{len(coordinates)} != {selection.topology_atom_count}"
        )
    if any(not isfinite(value) for coordinate in coordinates for value in coordinate):
        raise AshRefinementError("ASH optimized coordinates contain NaN or infinity")
    if output.get("qm_method") not in {None, qm_method}:
        raise AshRefinementError("ASH output QM method does not match the request")
    if output.get("embedding") not in {None, embedding}:
        raise AshRefinementError("ASH output embedding does not match the request")
    energy_value = output.get("final_energy_hartree")
    final_energy = float(energy_value) if energy_value is not None else None
    if final_energy is not None and not isfinite(final_energy):
        raise AshRefinementError("ASH final energy is NaN or infinity")
    if not reused_cached_result:
        cache_metadata = {
            "schema_version": 1,
            "input_payload": payload,
            "prmtop_scientific_sha256": _amber_prmtop_scientific_sha256(prmtop),
            "inpcrd_sha256": _sha256(inpcrd),
            "output_sha256": _sha256(output_path),
            "stdout_sha256": _sha256(stdout_path),
            "stderr_sha256": _sha256(stderr_path),
            "command_record_sha256": _sha256(command_record_path),
        }
        (work / "ash_refinement_cache.json").write_text(
            json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return AshRefinementRun(
        command_result=command_result,
        driver_path=driver_path,
        input_path=input_path,
        output_path=output_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        command_record_path=command_record_path,
        python_executable=python_executable,
        qm_method=qm_method,
        embedding=embedding,
        xtb_executable=None if xtb_executable is None else str(xtb_executable),
        mace_model_path=None if mace_model_path is None else str(mace_model_path),
        mace_model_sha256=mace_model_sha256,
        optimized_coordinates_angstrom=coordinates,
        final_energy_hartree=final_energy,
        reused_cached_result=reused_cached_result,
    )


def _resolve_executable(configured: str, *, label: str) -> str:
    path = Path(configured).expanduser()
    if path.is_file():
        return str(path.resolve())
    discovered = which_executable(configured)
    if discovered is not None:
        return discovered
    raise AshRefinementError(f"{label} not found: {configured}")


def _amber_prmtop_scientific_sha256(path: Path) -> str:
    """Hash prmtop scientific content while excluding Amber's wall-clock stamp."""

    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if lines and lines[0].startswith(b"%VERSION"):
        data = b"".join(lines[1:])
    return hashlib.sha256(data).hexdigest()


def _tail(text: str, *, lines: int = 25, characters: int = 5000) -> str:
    if not text:
        return "<empty>"
    value = "\n".join(text.rstrip().splitlines()[-lines:])
    return value[-characters:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ASH_DRIVER = r'''#!/usr/bin/env python3
"""Generated mdprep ASH driver. Inputs and outputs are JSON for auditability."""

import hashlib
import json
import sys

from ash import Fragment, OpenMMTheory, QMMMTheory, geomeTRICOptimizer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_ash_refinement.py INPUT.json OUTPUT.json")
    input_path, output_path = sys.argv[1:]
    with open(input_path, encoding="utf-8") as handle:
        data = json.load(handle)

    mace = data.get("mace")
    if mace is not None:
        for path in mace["python_search_paths"]:
            if path not in sys.path:
                sys.path.append(path)

    fragment = Fragment(
        amber_prmtopfile=data["prmtop_path"],
        amber_inpcrdfile=data["inpcrd_path"],
        conncalc=False,
    )
    mm_theory = OpenMMTheory(
        Amberfiles=True,
        amberprmtopfile=data["prmtop_path"],
        periodic=False,
        platform=data["platform"],
        numcores=data["num_cores"],
        applyconstraints_in_run=False,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
        printlevel=2,
    )
    if data["qm_method"] in {"gfn2_xtb", "gxtb"}:
        from ash import xTBTheory

        qm_theory = xTBTheory(
            xtbdir=data["xtb_directory"],
            xtbmethod="GFN2" if data["qm_method"] == "gfn2_xtb" else "G-XTB",
            runmode="inputfile",
            numcores=data["num_cores"],
            maxiter=data["xtb_max_iterations"],
            electronic_temp=data["electronic_temperature_kelvin"],
            accuracy=data["accuracy"],
            printlevel=2,
        )
    elif data["qm_method"] == "mace_polar1":
        from ash import MACETheory

        with open(mace["model_path"], "rb") as handle:
            model_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if model_sha256 != mace["model_sha256"]:
            raise RuntimeError(
                "MACE-POLAR-1 checkpoint changed after mdprep validation: "
                f"expected {mace['model_sha256']}, found {model_sha256}"
            )
        qm_theory = MACETheory(
            model_file=mace["model_path"],
            platform=mace["device"],
            default_dtype="float64",
            numcores=data["num_cores"],
            printlevel=2,
        )
    else:
        raise RuntimeError(f"Unsupported refinement QM method: {data['qm_method']}")
    qmmm_theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm_theory,
        mm_theory=mm_theory,
        qmatoms=data["qm_atom_indices"],
        embedding="Elstat" if data["embedding"] == "electrostatic" else "Mechanical",
        qm_charge=data["qm_charge"],
        qm_mult=data["qm_multiplicity"],
        unusualboundary=data["allow_unusual_link_boundaries"],
        excludeboundaryatomlist=data["qm_boundary_excluded_atom_indices"],
        numcores=data["num_cores"],
        printlevel=2,
    )
    result = geomeTRICOptimizer(
        theory=qmmm_theory,
        fragment=fragment,
        charge=data["qm_charge"],
        mult=data["qm_multiplicity"],
        coordsystem="hdlc",
        maxiter=data["max_iterations"],
        ActiveRegion=True,
        actatoms=data["active_atom_indices"],
        # ASH always writes full-system and QM-region XYZ trajectories for an
        # active-region optimization. Its optional PDB writer depends on an MD
        # ``simulation`` object that OpenMMTheory does not create for QM/MM
        # energy/gradient calls, so it must remain disabled here.
        MM_PDB_traj_write=False,
        result_write_to_disk=True,
        printlevel=2,
    )
    energy = getattr(result, "energy", None)
    payload = {
        "status": "converged",
        "qm_method": data["qm_method"],
        "embedding": data["embedding"],
        "coordinates_angstrom": fragment.coords.tolist(),
        "final_energy_hartree": None if energy is None else float(energy),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
'''
