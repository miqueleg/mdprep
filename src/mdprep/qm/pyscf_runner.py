"""Lazy PySCF runner for ligand RESP/QMMESP charge derivation."""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PySCFRunnerError(RuntimeError):
    """Raised when PySCF cannot run the requested QM calculation."""


@dataclass(frozen=True)
class PySCFResult:
    mol: object
    mf: object
    method: str
    basis: str
    charge: int
    spin: int
    multiplicity: int
    electron_count: int
    energy_hartree: float
    converged: bool
    cycles: int | None
    stdout: str
    stderr: str
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "basis": self.basis,
            "charge": self.charge,
            "spin": self.spin,
            "multiplicity": self.multiplicity,
            "electron_count": self.electron_count,
            "energy_hartree": self.energy_hartree,
            "converged": self.converged,
            "cycles": self.cycles,
            "warnings": self.warnings,
        }


def pyscf_available() -> bool:
    try:
        import pyscf  # noqa: F401
    except Exception:
        return False
    return True


def pyscf_version() -> str:
    try:
        import pyscf
    except Exception:
        return "not available"
    return getattr(pyscf, "__version__", "unknown")


def scf_class_name(method: str, spin: int) -> str:
    if method.upper() == "HF":
        return "UHF" if spin > 0 else "RHF"
    return "UKS" if spin > 0 else "RKS"


def run_pyscf_scf(
    *,
    elements: list[str],
    coordinates: np.ndarray,
    charge: int,
    spin: int,
    method: str,
    basis: str,
    max_cycle: int,
    conv_tol: float,
    mm_charges: np.ndarray | None = None,
    mm_coordinates: np.ndarray | None = None,
    work_dir: str | Path | None = None,
) -> PySCFResult:
    try:
        from pyscf import dft, gto, qmmm, scf
    except Exception as exc:
        raise PySCFRunnerError(
            "PySCF is required for gas_resp_pyscf and qmmesp_pyscf charge methods."
        ) from exc

    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(elements), 3):
        raise PySCFRunnerError("QM coordinate array shape does not match element list.")
    atom_spec = [(element, tuple(coord)) for element, coord in zip(elements, coords, strict=True)]
    stdout = io.StringIO()
    stderr = io.StringIO()
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.unit = "Angstrom"
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = basis
    mol.verbose = 4
    mol.stdout = stdout
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            mol.build()
        except Exception as exc:
            raise PySCFRunnerError(f"PySCF molecule build failed: {exc}") from exc

    if method.upper() == "HF":
        mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
    else:
        try:
            mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
            mf.xc = method
        except Exception as exc:
            raise PySCFRunnerError(f"PySCF DFT setup failed for method {method!r}; use method: HF.") from exc
    mf.stdout = stdout
    mf.max_cycle = int(max_cycle)
    mf.conv_tol = float(conv_tol)
    if mm_charges is not None and len(mm_charges):
        if mm_coordinates is None:
            raise PySCFRunnerError("MM coordinates are required when MM charges are provided.")
        try:
            mf = qmmm.mm_charge(mf, np.asarray(mm_coordinates, dtype=float), np.asarray(mm_charges, dtype=float), unit="Angstrom")
            mf.stdout = stdout
        except Exception as exc:
            raise PySCFRunnerError(f"PySCF point-charge embedding could not be applied: {exc}") from exc

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        energy = mf.kernel()
    converged = bool(getattr(mf, "converged", False))
    if not converged:
        raise PySCFRunnerError("PySCF SCF did not converge.")
    cycles = getattr(getattr(mf, "scf_summary", {}), "get", lambda key, default=None: default)("cycles", None)
    return PySCFResult(
        mol=mol,
        mf=mf,
        method=method,
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        multiplicity=int(spin) + 1,
        electron_count=int(mol.nelectron),
        energy_hartree=float(energy),
        converged=converged,
        cycles=cycles,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        warnings=[],
    )
