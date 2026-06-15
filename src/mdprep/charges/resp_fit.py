"""Native constrained ESP/RESP-like charge fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RespFitError(ValueError):
    """Raised when ESP/RESP fitting cannot be completed safely."""


@dataclass(frozen=True)
class RespFitResult:
    charges: np.ndarray
    charge_sum_before_correction: float
    charge_correction_applied: float
    charge_sum_final: float
    rms_error: float
    relative_rms_error: float | None
    max_error: float
    iterations: int
    converged: bool
    fitting_mode: str
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "charges": [float(value) for value in self.charges],
            "charge_sum_before_correction": self.charge_sum_before_correction,
            "charge_correction_applied": self.charge_correction_applied,
            "charge_sum_final": self.charge_sum_final,
            "rms_error": self.rms_error,
            "relative_rms_error": self.relative_rms_error,
            "max_error": self.max_error,
            "iterations": self.iterations,
            "converged": self.converged,
            "fitting_mode": self.fitting_mode,
            "warnings": self.warnings,
        }


def fit_resp_charges(
    *,
    atom_coordinates: np.ndarray,
    grid_coordinates: np.ndarray,
    esp_values: np.ndarray,
    total_charge: float,
    restraint: str = "resp",
    restraint_a: float = 0.0005,
    restraint_b: float = 0.1,
    max_iter: int = 25,
    convergence: float = 1.0e-6,
) -> RespFitResult:
    atoms = np.asarray(atom_coordinates, dtype=float)
    grid = np.asarray(grid_coordinates, dtype=float)
    esp = np.asarray(esp_values, dtype=float)
    if atoms.ndim != 2 or atoms.shape[1] != 3:
        raise RespFitError("Atom coordinates must have shape (n_atoms, 3).")
    if grid.ndim != 2 or grid.shape[1] != 3:
        raise RespFitError("Grid coordinates must have shape (n_points, 3).")
    if esp.shape != (len(grid),):
        raise RespFitError("ESP values must have one value per grid point.")
    if len(grid) < len(atoms) + 1:
        raise RespFitError("Too few ESP points for constrained charge fitting.")

    design = _design_matrix(atoms, grid)
    warnings = ["Equivalent-atom constraints are not implemented in mdprep v0.1; fitted charges are atom-specific."]
    regularization = np.zeros(len(atoms), dtype=float)
    charges = np.zeros(len(atoms), dtype=float)
    converged = False
    iterations = 1
    mode = "constrained_esp"
    if restraint == "resp":
        mode = "native_resp_like"
        for iteration in range(1, max_iter + 1):
            new_charges = _solve_constrained(design, esp, total_charge, regularization)
            delta = float(np.max(np.abs(new_charges - charges))) if len(charges) else 0.0
            charges = new_charges
            regularization = restraint_a / np.sqrt(charges * charges + restraint_b * restraint_b)
            iterations = iteration
            if delta <= convergence:
                converged = True
                break
        if not converged:
            warnings.append("RESP-like fit reached max_iter before convergence.")
    elif restraint == "none":
        charges = _solve_constrained(design, esp, total_charge, regularization)
        converged = True
    else:
        raise RespFitError(f"Unsupported restraint mode: {restraint}")

    before = float(np.sum(charges))
    residual = total_charge - before
    correction = 0.0
    if abs(residual) > 1.0e-10:
        index = int(np.argmax(np.abs(charges)))
        charges[index] += residual
        correction = residual
    predicted = design @ charges
    errors = predicted - esp
    rms = float(np.sqrt(np.mean(errors * errors)))
    esp_rms = float(np.sqrt(np.mean(esp * esp)))
    rel = rms / esp_rms if esp_rms > 0 else None
    return RespFitResult(
        charges=charges,
        charge_sum_before_correction=before,
        charge_correction_applied=correction,
        charge_sum_final=float(np.sum(charges)),
        rms_error=rms,
        relative_rms_error=rel,
        max_error=float(np.max(np.abs(errors))),
        iterations=iterations,
        converged=converged,
        fitting_mode=mode,
        warnings=warnings,
    )


def _design_matrix(atom_coordinates: np.ndarray, grid_coordinates: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(grid_coordinates[:, None, :] - atom_coordinates[None, :, :], axis=2)
    if np.any(distances < 1.0e-8):
        raise RespFitError("ESP grid contains a point on an atom center.")
    return 1.0 / distances


def _solve_constrained(
    design: np.ndarray,
    esp: np.ndarray,
    total_charge: float,
    regularization: np.ndarray,
) -> np.ndarray:
    lhs = design.T @ design + np.diag(regularization)
    rhs = design.T @ esp
    ones = np.ones((len(rhs), 1))
    augmented = np.block([[lhs, ones], [ones.T, np.zeros((1, 1))]])
    target = np.concatenate([rhs, np.asarray([total_charge], dtype=float)])
    try:
        solution = np.linalg.solve(augmented, target)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    charges = solution[:-1]
    if not np.all(np.isfinite(charges)):
        raise RespFitError("Charge fit produced non-finite charges.")
    return charges
