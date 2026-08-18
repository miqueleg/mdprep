"""Deterministic layered Merz-Kollman ESP grid generation."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from pathlib import Path

import numpy as np


class EspGridError(ValueError):
    """Raised when an ESP grid cannot be generated safely."""


VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "FE": 2.00,
    "ZN": 2.10,
    "MG": 1.73,
    "NA": 2.27,
    "K": 2.75,
    "CA": 2.31,
}


@dataclass(frozen=True)
class EspGrid:
    points: np.ndarray
    atom_indices: list[int]
    shell_scales: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "point_count": int(len(self.points)),
            "atom_indices": self.atom_indices,
            "shell_scales": self.shell_scales,
        }


def generate_merz_kollman_grid(
    *,
    elements: list[str],
    coordinates: np.ndarray,
    vdw_scale_factors: list[float],
    point_density_per_square_angstrom: float,
    exclude_inside_vdw_scale: float,
    max_points: int,
) -> EspGrid:
    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(elements), 3):
        raise EspGridError("Coordinate array shape does not match element list.")
    if not elements:
        raise EspGridError("Cannot generate an ESP grid for zero atoms.")
    if not np.all(np.isfinite(coords)):
        raise EspGridError("ESP-grid coordinates contain non-finite values.")
    if not vdw_scale_factors or any(
        not np.isfinite(scale) or scale <= 0 for scale in vdw_scale_factors
    ):
        raise EspGridError("ESP-grid vdW scale factors must be finite, positive, and non-empty.")
    if not np.isfinite(point_density_per_square_angstrom) or point_density_per_square_angstrom <= 0:
        raise EspGridError("ESP-grid surface-point density must be finite and positive.")
    if not np.isfinite(exclude_inside_vdw_scale) or exclude_inside_vdw_scale <= 0:
        raise EspGridError("ESP-grid exclusion scale must be finite and positive.")
    if max_points < len(elements) + 1:
        raise EspGridError("ESP-grid max_points is too small for the number of fitted atoms.")

    points: list[np.ndarray] = []
    atom_indices: list[int] = []
    shell_scales: list[float] = []
    radii = np.asarray([vdw_radius(element) for element in elements], dtype=float)
    for atom_index, center in enumerate(coords):
        for scale in vdw_scale_factors:
            shell_radius = radii[atom_index] * scale
            point_count = max(
                6,
                int(round(4.0 * pi * shell_radius * shell_radius * point_density_per_square_angstrom)),
            )
            unit_sphere = _fibonacci_sphere(point_count)
            for unit_point in unit_sphere:
                point = center + unit_point * shell_radius
                if _inside_other_atom(
                    point,
                    coords,
                    radii,
                    parent_index=atom_index,
                    exclude_scale=exclude_inside_vdw_scale,
                ):
                    continue
                points.append(point)
                atom_indices.append(atom_index)
                shell_scales.append(scale)
    if not points:
        raise EspGridError("No ESP grid points survived vdW exclusion.")

    deduped_points, deduped_atoms, deduped_scales = _dedupe(points, atom_indices, shell_scales)
    if len(deduped_points) < len(elements) + 1:
        raise EspGridError("Too few ESP grid points were generated for charge fitting.")
    if len(deduped_points) > max_points:
        raise EspGridError(
            f"Merz-Kollman grid contains {len(deduped_points)} points, exceeding "
            f"grid.max_points={max_points}; refusing to thin the RESP sampling grid."
        )
    return EspGrid(
        points=np.asarray(deduped_points, dtype=float),
        atom_indices=deduped_atoms,
        shell_scales=deduped_scales,
    )


def vdw_radius(element: str) -> float:
    key = element.strip().upper()
    if key not in VDW_RADII:
        raise EspGridError(f"No vdW radius is configured for element {element!r}.")
    return VDW_RADII[key]


def write_grid_xyz(grid: EspGrid, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(len(grid.points)), "mdprep ESP grid"]
    for point in grid.points:
        lines.append(f"X {point[0]:.8f} {point[1]:.8f} {point[2]:.8f}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_grid_xyz(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    data = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) == 4:
            data.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return np.asarray(data, dtype=float)


def _fibonacci_sphere(count: int) -> np.ndarray:
    points = []
    golden_angle = pi * (3.0 - sqrt(5.0))
    for index in range(count):
        y = 1.0 - (2.0 * index + 1.0) / count
        radius = sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * index
        points.append([cos(theta) * radius, y, sin(theta) * radius])
    return np.asarray(points, dtype=float)


def _inside_other_atom(
    point: np.ndarray,
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    parent_index: int,
    exclude_scale: float,
) -> bool:
    for index, center in enumerate(coords):
        if index == parent_index:
            continue
        if np.linalg.norm(point - center) < radii[index] * exclude_scale:
            return True
    return False


def _dedupe(
    points: list[np.ndarray],
    atom_indices: list[int],
    shell_scales: list[float],
) -> tuple[list[np.ndarray], list[int], list[float]]:
    seen: set[tuple[int, int, int]] = set()
    deduped_points: list[np.ndarray] = []
    deduped_atoms: list[int] = []
    deduped_scales: list[float] = []
    for point, atom_index, scale in zip(points, atom_indices, shell_scales, strict=True):
        key = tuple(int(round(value * 10000.0)) for value in point)
        if key in seen:
            continue
        seen.add(key)
        deduped_points.append(point)
        deduped_atoms.append(atom_index)
        deduped_scales.append(scale)
    return deduped_points, deduped_atoms, deduped_scales
