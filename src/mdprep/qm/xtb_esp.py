"""GFN2 electrostatic potential on a grid, with optional MM embedding.

MCPB's large model exists to be fitted to an ESP, and the fit is what sets
the charges the finished system carries. PySCF does that at DFT cost: for the
BesD site, an open-shell Fe(II) large model at B3LYP/6-31G* with a few
thousand embedding charges. GFN2 answers the same question in seconds, which
is the difference between a method a student can run and one they cannot.

What this is not: a RESP calculation at HF/6-31G*. GFN2 is a semi-empirical
valence density, and charges fitted to its ESP inherit that. Every result
carries the warning, and ``esp_backend: pyscf`` remains the default.

xTB reads the grid and the point charges in bohr, and writes the potential
beside the grid it was given. Both are checked rather than assumed: a grid
silently reordered or re-scaled would still produce a plausible fit.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mdprep.external.discovery import which_executable
from mdprep.external.runner import run_command
from mdprep.protonation.xtb_parser import XtbParseError, parse_xtb_energy_text
from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM
from mdprep.structure.pdb import ATOMIC_NUMBER


class XtbEspError(ValueError):
    """Raised when no usable GFN2 density ESP can be produced."""


# An MM site is a point charge, not an atom. xTB would otherwise give it the
# chemical hardness of some default element and let it screen the QM density.
POINT_CHARGE_HARDNESS = 1.0e16


@dataclass(frozen=True)
class XtbEspResult:
    esp_hartree_per_e: np.ndarray
    atomic_esp_hartree_per_e: np.ndarray
    atomic_charges: np.ndarray
    energy_hartree: float
    metadata: dict
    warnings: list[str]


def write_embedding(
    charges: Sequence[float],
    coordinates_angstrom: np.ndarray,
    path: str | Path,
) -> None:
    """Write xTB's ``pcharge`` file, positions in bohr."""

    values = np.asarray(charges, dtype=float)
    points = np.asarray(coordinates_angstrom, dtype=float)
    if values.ndim != 1 or points.shape != (len(values), 3):
        raise XtbEspError(
            "MM embedding needs one finite charge and one position per site; "
            f"got {values.shape} charges and {points.shape} positions."
        )
    if not (np.isfinite(values).all() and np.isfinite(points).all()):
        raise XtbEspError("MM embedding charges and positions must all be finite.")
    rows = [str(len(values))]
    rows.extend(
        f"{charge:.16g} {x:.16g} {y:.16g} {z:.16g} {POINT_CHARGE_HARDNESS:.16g}"
        for charge, (x, y, z) in zip(values, points * BOHR_PER_ANGSTROM)
    )
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_esp(path: str | Path, grid_angstrom: np.ndarray) -> np.ndarray:
    """Read ``xtb_esp.dat``, checking it describes the grid that was asked for."""

    try:
        data = np.loadtxt(path, ndmin=2)
    except (OSError, ValueError) as exc:
        raise XtbEspError(f"xTB wrote no readable ESP at {path}: {exc}") from exc
    grid = np.asarray(grid_angstrom, dtype=float) * BOHR_PER_ANGSTROM
    if data.shape != (len(grid), 4) or not np.isfinite(data).all():
        raise XtbEspError(
            "xTB's ESP output must carry one finite x/y/z/potential row per "
            f"requested grid point; got shape {data.shape} for {len(grid)} points."
        )
    if not np.allclose(data[:, :3], grid, rtol=1e-8, atol=1e-8):
        raise XtbEspError(
            "The grid xTB reported back is not the one it was given, so the "
            "potentials cannot be matched to grid points. Refusing the fit."
        )
    return data[:, 3]


def _atomic_esp_excluding_self(
    coordinates_angstrom: np.ndarray,
    charges: np.ndarray,
) -> np.ndarray:
    """Potential at each atom from every other atom's GFN2 charge.

    The density ESP diverges at a nucleus, so the value at an atom's own
    position has to come from somewhere else. MCPB.py parses these numbers out
    of the Gaussian log and then discards them -- only the fit-centre values
    reach the RESP input -- so this exists to put a defensible number in the
    file rather than a placeholder.
    """

    points = np.asarray(coordinates_angstrom, dtype=float) * BOHR_PER_ANGSTROM
    values = np.zeros(len(points), dtype=float)
    for index, origin in enumerate(points):
        separations = np.linalg.norm(points - origin[None, :], axis=1)
        other = np.arange(len(points)) != index
        values[index] = float(np.sum(charges[other] / separations[other]))
    return values


def run_xtb_esp(
    *,
    elements: Sequence[str],
    coordinates_angstrom: np.ndarray,
    grid_coordinates_angstrom: np.ndarray,
    charge: int,
    spin: int,
    work_dir: str | Path,
    executable: str = "xtb",
    max_cycle: int = 500,
    electronic_temperature_kelvin: float | None = None,
    solvent: str | None = None,
    num_threads: int = 1,
    omp_stacksize: str = "1G",
    mm_charges: Sequence[float] | None = None,
    mm_coordinates_angstrom: np.ndarray | None = None,
) -> XtbEspResult:
    """Evaluate the GFN2 density ESP on a grid, optionally MM-embedded."""

    import os

    binary = which_executable(executable)
    if binary is None:
        raise XtbEspError(f"xTB executable not found: {executable}")

    xyz = np.asarray(coordinates_angstrom, dtype=float)
    grid = np.asarray(grid_coordinates_angstrom, dtype=float)
    if xyz.shape != (len(elements), 3):
        raise XtbEspError(
            f"Expected one position per element; got {xyz.shape} for "
            f"{len(elements)} elements."
        )
    if grid.ndim != 2 or grid.shape[1] != 3 or not len(grid):
        raise XtbEspError("The ESP grid must be a non-empty array of xyz points.")
    if not (np.isfinite(xyz).all() and np.isfinite(grid).all()):
        raise XtbEspError("QM and grid coordinates must all be finite.")

    try:
        atomic_numbers = [ATOMIC_NUMBER[element.strip().upper()] for element in elements]
    except KeyError as exc:
        raise XtbEspError(f"Unrecognised element in the QM region: {exc}") from exc
    electrons = sum(atomic_numbers) - charge
    if spin < 0 or electrons <= 0 or spin > electrons or (electrons - spin) % 2:
        raise XtbEspError(
            f"A charge of {charge:+d} leaves {electrons} electrons, which cannot "
            f"carry {spin} unpaired. Check the declared charge and multiplicity."
        )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    xyz_path = work / "large_model.xyz"
    xyz_path.write_text(
        f"{len(elements)}\nmdprep MCPB large model\n"
        + "".join(
            f"{element} {x:.12f} {y:.12f} {z:.12f}\n"
            for element, (x, y, z) in zip(elements, xyz)
        ),
        encoding="utf-8",
    )
    np.savetxt(work / "esp_coord", grid * BOHR_PER_ANGSTROM, fmt="%.16g")

    control = "$write\n esp=true\n gridfile=esp_coord\n$end\n"
    embedded = mm_charges is not None and len(mm_charges) > 0
    mm_count = 0 if mm_charges is None else len(mm_charges)
    if embedded:
        if mm_coordinates_angstrom is None:
            raise XtbEspError("MM charges were supplied without their positions.")
        write_embedding(mm_charges, mm_coordinates_angstrom, work / "pcharge")
        control += "$embedding\n input=pcharge\n es=true\n$end\n"
    (work / "control.inp").write_text(control, encoding="utf-8")

    command = [
        binary,
        xyz_path.name,
        "--gfn", "2",
        "--chrg", str(charge),
        "--uhf", str(spin),
        "--iterations", str(max_cycle),
        "--esp",
        "--input", "control.inp",
    ]
    if electronic_temperature_kelvin is not None:
        command.extend(["--etemp", str(electronic_temperature_kelvin)])
    if solvent is not None:
        command.extend(["--alpb", solvent])

    environment = dict(os.environ)
    threads = str(num_threads)
    environment.update(
        {
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "OMP_STACKSIZE": omp_stacksize,
        }
    )
    result = run_command(command, cwd=work, env=environment)
    (work / "xtb_stdout.txt").write_text(result.stdout, encoding="utf-8")
    (work / "xtb_stderr.txt").write_text(result.stderr, encoding="utf-8")
    (work / "command.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    if result.returncode != 0 or (work / "NOT_CONVERGED").exists():
        raise XtbEspError(
            f"The GFN2 ESP calculation failed; see {work / 'xtb_stdout.txt'}."
        )
    if embedded and not re.search(r"PC potential\s+true", result.stdout, re.I):
        raise XtbEspError(
            "xTB did not report applying the point charges, so the density is "
            "unpolarised. Fitting charges to it would silently give gas-phase "
            f"values; see {work / 'xtb_stdout.txt'}."
        )

    esp = parse_esp(work / "xtb_esp.dat", grid)
    charges_path = work / "charges"
    if not charges_path.is_file():
        raise XtbEspError(f"xTB wrote no atomic charges at {charges_path}.")
    atomic_charges = np.asarray(
        [float(value) for value in charges_path.read_text().split()], dtype=float
    )
    if len(atomic_charges) != len(elements):
        raise XtbEspError(
            f"xTB reported {len(atomic_charges)} atomic charges for "
            f"{len(elements)} atoms."
        )
    try:
        energy = parse_xtb_energy_text(result.stdout)
    except XtbParseError as exc:
        raise XtbEspError(
            f"The GFN2 ESP calculation reported no total energy; see "
            f"{work / 'xtb_stdout.txt'}."
        ) from exc

    return XtbEspResult(
        esp_hartree_per_e=esp,
        atomic_esp_hartree_per_e=_atomic_esp_excluding_self(xyz, atomic_charges),
        atomic_charges=atomic_charges,
        energy_hartree=energy,
        metadata={
            "backend": "xtb",
            "method": "GFN2-xTB",
            "basis": "GFN2 valence basis",
            "charge": charge,
            "spin": spin,
            "multiplicity": spin + 1,
            "electron_count": electrons,
            "energy_hartree": energy,
            "grid_point_count": int(len(grid)),
            "mm_point_charge_count": int(mm_count),
            "electrostatic_embedding_applied": bool(embedded),
            "solvent": solvent,
            "command": list(result.command),
            "version": next(
                (line.strip() for line in result.stdout.splitlines() if "xtb version" in line),
                "unknown",
            ),
        },
        warnings=[
            "Charges were fitted to a GFN2 semi-empirical valence density, not "
            "to an HF/6-31G* density. They are not RESP charges in the usual "
            "sense; use esp_backend: pyscf when that matters."
        ],
    )
