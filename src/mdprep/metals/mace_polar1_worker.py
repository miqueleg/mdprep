"""Isolated MACE-POLAR-1 inference worker invoked through the external runner.

This module deliberately has no imports from mdprep. It is executed by the
Python interpreter configured for MACE-POLAR-1, which may be a dedicated
environment separate from mdprep's own Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any


EV_PER_HARTREE = 27.211386245988
ANGSTROM_PER_BOHR = 0.529177210903
MINIMUM_MACE_VERSION = (0, 3, 16)


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, str | None]:
    return {
        "python": sys.version.splitlines()[0],
        "mace_torch": _version("mace-torch"),
        "torch": _version("torch"),
        "ase": _version("ase"),
        "graph_longrange": _version("graph-longrange"),
    }


def _require_supported_mace_version() -> str:
    version = _version("mace-torch")
    if version is None:
        raise RuntimeError("mace-torch distribution metadata is unavailable")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    parsed = tuple(int(value) for value in match.groups()) if match else None
    if parsed is None or parsed < MINIMUM_MACE_VERSION:
        raise RuntimeError(
            "MACE-POLAR-1 Hessians require mace-torch 0.3.16 or newer; "
            f"found {version}"
        )
    return version


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cartesian_matrix(raw: object, atom_count: int) -> Any:
    import numpy as np

    values = np.asarray(raw, dtype=float)
    dimension = atom_count * 3
    if values.shape == (dimension, dimension):
        matrix = values
    elif values.shape == (dimension, atom_count, 3):
        matrix = values.reshape((dimension, dimension))
    elif values.shape == (atom_count, 3, atom_count, 3):
        matrix = values.reshape((dimension, dimension))
    elif values.shape == (atom_count, atom_count, 3, 3):
        matrix = values.transpose(0, 2, 1, 3).reshape((dimension, dimension))
    else:
        raise ValueError(
            f"MACE Hessian shape {values.shape} is incompatible with "
            f"{atom_count} atoms"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("MACE-POLAR-1 Hessian contains non-finite values")
    return matrix


def _deterministic_direction(dimension: int) -> Any:
    import numpy as np

    indices = np.arange(1, dimension + 1, dtype=float)
    direction = np.sin(indices) + 0.5 * np.cos(indices * 0.37)
    direction /= np.linalg.norm(direction)
    return direction


def _finite_difference_check(
    atoms: Any,
    hessian_ev_per_angstrom2: object,
    *,
    step_angstrom: float,
) -> dict[str, float]:
    import numpy as np

    atom_count = len(atoms)
    direction = _deterministic_direction(atom_count * 3)
    original = np.asarray(atoms.get_positions(), dtype=float)
    displaced = direction.reshape((atom_count, 3)) * step_angstrom
    try:
        atoms.set_positions(original + displaced)
        force_plus = np.asarray(atoms.get_forces(), dtype=float).reshape(-1)
        atoms.set_positions(original - displaced)
        force_minus = np.asarray(atoms.get_forces(), dtype=float).reshape(-1)
    finally:
        atoms.set_positions(original)
    finite_difference = -(force_plus - force_minus) / (2.0 * step_angstrom)
    analytic = np.asarray(hessian_ev_per_angstrom2, dtype=float) @ direction
    difference = analytic - finite_difference
    denominator = max(
        float(np.linalg.norm(analytic)),
        float(np.linalg.norm(finite_difference)),
        1.0e-12,
    )
    return {
        "step_angstrom": step_angstrom,
        "analytic_hessian_vector_norm_ev_per_angstrom2": float(
            np.linalg.norm(analytic)
        ),
        "finite_difference_hessian_vector_norm_ev_per_angstrom2": float(
            np.linalg.norm(finite_difference)
        ),
        "difference_norm_ev_per_angstrom2": float(np.linalg.norm(difference)),
        "relative_error": float(np.linalg.norm(difference) / denominator),
    }


def _run(args: argparse.Namespace) -> None:
    import numpy as np
    from ase.io import read
    from mace.calculators import mace_polar
    from mace.calculators.foundations_models import download_mace_polar_checkpoint

    _require_supported_mace_version()

    input_path = Path(args.xyz).resolve()
    output_hessian = Path(args.output_hessian).resolve()
    output_report = Path(args.output_report).resolve()
    checkpoint = Path(download_mace_polar_checkpoint(args.model)).resolve()
    model_sha256 = _sha256(checkpoint)
    if args.expected_model_sha256 is not None and (
        model_sha256.lower() != args.expected_model_sha256.lower()
    ):
        raise ValueError(
            "MACE-POLAR-1 checkpoint checksum mismatch: expected "
            f"{args.expected_model_sha256}, found {model_sha256}"
        )

    atoms = read(input_path, format="xyz")
    if len(atoms) != args.atom_count:
        raise ValueError(
            f"XYZ contains {len(atoms)} atoms; expected {args.atom_count}"
        )
    atoms.info["charge"] = args.charge
    # MACE-POLAR-1 defines spin as the spin multiplicity (unpaired electrons + 1).
    atoms.info["spin"] = args.multiplicity
    atoms.info["external_field"] = [0.0, 0.0, 0.0]

    started = perf_counter()
    calculator = mace_polar(
        model=str(checkpoint),
        device=args.device,
        default_dtype="float64",
    )
    supported_atomic_numbers = sorted(
        int(value) for value in calculator.z_table.zs
    )
    unsupported = sorted(
        set(int(value) for value in atoms.numbers) - set(supported_atomic_numbers)
    )
    if unsupported:
        raise ValueError(
            f"MACE-POLAR-1 checkpoint does not support atomic numbers {unsupported}"
        )
    atoms.calc = calculator
    energy_ev = float(atoms.get_potential_energy())
    forces_ev_per_angstrom = np.asarray(atoms.get_forces(), dtype=float)
    raw_hessian = calculator.get_hessian(atoms=atoms)
    hessian_ev_per_angstrom2 = _cartesian_matrix(raw_hessian, len(atoms))
    finite_difference = _finite_difference_check(
        atoms,
        hessian_ev_per_angstrom2,
        step_angstrom=args.finite_difference_step_angstrom,
    )
    if finite_difference["relative_error"] > args.max_hessian_vector_relative_error:
        raise ValueError(
            "MACE-POLAR-1 analytic Hessian failed its deterministic central-force "
            "difference check: relative error "
            f"{finite_difference['relative_error']:.6g} exceeds "
            f"{args.max_hessian_vector_relative_error:.6g}"
        )

    hessian_hartree_per_bohr2 = (
        hessian_ev_per_angstrom2 * ANGSTROM_PER_BOHR**2 / EV_PER_HARTREE
    )
    antisymmetry = float(
        np.max(np.abs(hessian_hartree_per_bohr2 - hessian_hartree_per_bohr2.T))
    )
    np.save(output_hessian, hessian_hartree_per_bohr2)
    report = {
        "status": "complete",
        "backend": "MACE-POLAR-1",
        "model": args.model,
        "model_path": str(checkpoint),
        "model_sha256": model_sha256,
        "versions": _versions(),
        "device": args.device,
        "default_dtype": "float64",
        "atom_count": len(atoms),
        "atomic_numbers": [int(value) for value in atoms.numbers],
        "supported_atomic_numbers": supported_atomic_numbers,
        "molecular_charge": args.charge,
        "multiplicity": args.multiplicity,
        "spin_input_semantics": "spin multiplicity (unpaired electrons + 1)",
        "external_field": [0.0, 0.0, 0.0],
        "energy_ev": energy_ev,
        "energy_hartree": energy_ev / EV_PER_HARTREE,
        "force_norm_ev_per_angstrom": float(np.linalg.norm(forces_ev_per_angstrom)),
        "gradient_norm_hartree_per_bohr": float(
            np.linalg.norm(forces_ev_per_angstrom)
            * ANGSTROM_PER_BOHR
            / EV_PER_HARTREE
        ),
        "raw_hessian_shape": list(np.asarray(raw_hessian).shape),
        "hessian_shape": list(hessian_hartree_per_bohr2.shape),
        "source_hessian_units": "eV/angstrom^2",
        "output_hessian_units": "hartree/bohr^2",
        "hessian_antisymmetry_max_hartree_per_bohr2": antisymmetry,
        "hessian_sha256": _sha256(output_hessian),
        "finite_difference_validation": finite_difference,
        "runtime_seconds": perf_counter() - started,
        "geometry_optimization_performed": False,
    }
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--xyz")
    parser.add_argument("--atom-count", type=int)
    parser.add_argument("--charge", type=int)
    parser.add_argument("--multiplicity", type=int)
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--finite-difference-step-angstrom", type=float)
    parser.add_argument("--max-hessian-vector-relative-error", type=float)
    parser.add_argument("--output-hessian")
    parser.add_argument("--output-report")
    args = parser.parse_args()
    if args.probe:
        from mace.calculators import mace_polar  # noqa: F401

        _require_supported_mace_version()
        print(
            json.dumps(
                {"status": "available", "versions": _versions()},
                sort_keys=True,
            )
        )
        return
    required = (
        "xyz",
        "atom_count",
        "charge",
        "multiplicity",
        "model",
        "device",
        "finite_difference_step_angstrom",
        "max_hessian_vector_relative_error",
        "output_hessian",
        "output_report",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    _run(args)


if __name__ == "__main__":
    main()
