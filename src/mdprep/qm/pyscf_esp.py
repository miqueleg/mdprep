"""ESP evaluation from a PySCF density matrix."""

from __future__ import annotations

import numpy as np


BOHR_PER_ANGSTROM = 1.8897261254578281


class PySCFEspError(ValueError):
    """Raised when ESP evaluation fails."""


def evaluate_ligand_esp(
    *,
    mol: object,
    mf: object,
    grid_coordinates_angstrom: np.ndarray,
) -> np.ndarray:
    points = np.asarray(grid_coordinates_angstrom, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PySCFEspError("Grid coordinates must have shape (n_points, 3).")
    atom_coords_bohr = np.asarray(mol.atom_coords(), dtype=float)
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)
    dm = mf.make_rdm1()
    if isinstance(dm, tuple) or (hasattr(dm, "ndim") and dm.ndim == 3):
        dm_total = np.asarray(dm[0]) + np.asarray(dm[1])
    else:
        dm_total = np.asarray(dm)

    values = []
    for point_angstrom in points:
        point_bohr = point_angstrom * BOHR_PER_ANGSTROM
        distances = np.linalg.norm(atom_coords_bohr - point_bohr, axis=1)
        if np.any(distances < 1.0e-10):
            raise PySCFEspError("ESP grid point is on a nucleus.")
        nuclear = float(np.sum(atom_charges / distances))
        mol.set_rinv_origin(point_bohr)
        rinv = mol.intor("int1e_rinv")
        electronic = -float(np.einsum("ij,ij->", dm_total, rinv))
        values.append(nuclear + electronic)
    return np.asarray(values, dtype=float)


def write_esp_values(values: np.ndarray, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{float(value):.12e}\n")
