"""Reading the DCD trajectory OpenMM writes, with numpy alone.

A DCD file is Fortran unformatted binary: every record is framed by its
length in bytes, before and after. A header (``CORD`` and twenty control
integers), a title, the atom count; then, per frame, an optional unit-cell
record of six doubles followed by the x, y and z coordinates as float32.

OpenMM stores the cell as ``a, cos(gamma), b, cos(beta), cos(alpha), c`` in
angstrom; older CHARMM writers put the angles in degrees, which is told apart
by magnitude.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DcdError(ValueError):
    """Raised when a file is not a DCD trajectory this reader understands."""


@dataclass
class DcdTrajectory:
    positions: np.ndarray  # (frames, atoms, 3), angstrom
    cells: np.ndarray | None  # (frames, 3, 3), one cell vector per row, angstrom
    timestep_ps: float | None  # simulated time between saved frames, if recorded
    first_step: int
    step_interval: int


def cell_vectors(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Cell vectors (rows) from lengths and angles in degrees, a along x."""

    alpha, beta, gamma = np.radians([alpha, beta, gamma])
    bx, by = b * np.cos(gamma), b * np.sin(gamma)
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    vectors = np.array([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])
    vectors[np.abs(vectors) < 1e-9] = 0.0
    return vectors


def _angle(value: float) -> float:
    """Degrees from a stored angle, which may be its cosine."""

    if -1.0 <= value <= 1.0:
        return float(np.degrees(np.arccos(value)))
    return float(value)


def read_dcd(path: str | Path) -> DcdTrajectory:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise DcdError(f"{Path(path).name} is too short to be a DCD file")
    for endian in ("<", ">"):
        if struct.unpack(endian + "i", data[:4])[0] == 84 and data[4:8] == b"CORD":
            break
    else:
        raise DcdError(f"{Path(path).name} does not start like a DCD file")

    offset = 0

    def record() -> bytes:
        nonlocal offset
        (size,) = struct.unpack_from(endian + "i", data, offset)
        body = data[offset + 4 : offset + 4 + size]
        (closing,) = struct.unpack_from(endian + "i", data, offset + 4 + size)
        if closing != size:
            raise DcdError(f"{Path(path).name} is damaged at byte {offset}")
        offset += size + 8
        return body

    header = record()
    control = struct.unpack(endian + "20i", header[4:84])
    first_step, step_interval = control[1], control[2]
    timestep = struct.unpack(endian + "f", header[4 + 9 * 4 : 4 + 10 * 4])[0]
    has_cell = control[10] != 0
    if control[8] != 0:
        raise DcdError("DCD files with fixed atoms are not supported")
    record()  # title
    (natoms,) = struct.unpack(endian + "i", record())

    positions = []
    cells = []
    coordinate_bytes = 4 * natoms
    while offset < len(data):
        if has_cell:
            a, cos_gamma, b, cos_beta, cos_alpha, c = struct.unpack(endian + "6d", record())
            cells.append(
                cell_vectors(a, b, c, _angle(cos_alpha), _angle(cos_beta), _angle(cos_gamma))
            )
        xyz = []
        for _ in range(3):
            body = record()
            if len(body) != coordinate_bytes:
                raise DcdError(f"{Path(path).name}: a frame has the wrong number of atoms")
            xyz.append(np.frombuffer(body, dtype=endian + "f4"))
        positions.append(np.stack(xyz, axis=1))
    # AKMA time units: 1 unit is 1/20.455 ps. OpenMM writes the step size so.
    timestep_ps = timestep / 20.455 * step_interval if timestep > 0 else None
    return DcdTrajectory(
        positions=np.array(positions, dtype=float).reshape(-1, natoms, 3),
        cells=np.array(cells) if cells else None,
        timestep_ps=timestep_ps,
        first_step=first_step,
        step_interval=step_interval,
    )
