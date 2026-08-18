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
    batch_size: int = 256,
) -> np.ndarray:
    points = np.asarray(grid_coordinates_angstrom, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise PySCFEspError("Grid coordinates must have shape (n_points, 3).")
    if len(points) == 0:
        raise PySCFEspError("At least one ESP grid point is required.")
    if batch_size < 1:
        raise PySCFEspError("ESP evaluation batch_size must be positive.")
    atom_coords_bohr = np.asarray(mol.atom_coords(), dtype=float)
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)
    dm = mf.make_rdm1()
    if isinstance(dm, tuple) or (hasattr(dm, "ndim") and dm.ndim == 3):
        dm_total = np.asarray(dm[0]) + np.asarray(dm[1])
    else:
        dm_total = np.asarray(dm)

    points_bohr = points * BOHR_PER_ANGSTROM
    distances = np.linalg.norm(points_bohr[:, None, :] - atom_coords_bohr[None, :, :], axis=2)
    if np.any(distances < 1.0e-10):
        raise PySCFEspError("ESP grid point is on a nucleus.")
    nuclear = np.sum(atom_charges[None, :] / distances, axis=1)
    electronic = np.empty(len(points_bohr), dtype=float)
    for start in range(0, len(points_bohr), batch_size):
        stop = min(start + batch_size, len(points_bohr))
        batch = points_bohr[start:stop]
        try:
            # PySCF's batched 1/r integral avoids one integral-engine setup per
            # ESP point and returns (n_grid, n_ao, n_ao).
            rinv = np.asarray(mol.intor("int1e_grids", grids=batch), dtype=float)
            if rinv.shape[0] != len(batch):
                raise ValueError("unexpected int1e_grids result shape")
            electronic[start:stop] = -np.einsum("pij,ij->p", rinv, dm_total, optimize=True)
        except (TypeError, ValueError, RuntimeError):
            # Older supported PySCF builds may not expose int1e_grids.  This is
            # the same integral evaluated point-by-point, not a charge-model
            # fallback.
            for offset, point_bohr in enumerate(batch, start=start):
                mol.set_rinv_origin(point_bohr)
                rinv_point = mol.intor("int1e_rinv")
                electronic[offset] = -float(np.einsum("ij,ij->", dm_total, rinv_point))
    values = np.asarray(nuclear + electronic, dtype=float)
    if not np.all(np.isfinite(values)):
        raise PySCFEspError("PySCF ESP evaluation produced non-finite values.")
    return values


def write_esp_values(values: np.ndarray, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{float(value):.12e}\n")
