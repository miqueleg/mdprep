"""The GFN2 ESP used for MCPB's large model."""

from pathlib import Path

import numpy as np
import pytest

from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM
from mdprep.qm.xtb_esp import (
    POINT_CHARGE_HARDNESS,
    XtbEspError,
    parse_esp,
    write_embedding,
)


def test_embedding_positions_are_written_in_bohr(tmp_path: Path):
    """xTB reads pcharge positions in bohr, and says so nowhere."""

    path = tmp_path / "pcharge"
    write_embedding([0.5, -0.5], np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), path)
    lines = path.read_text().splitlines()

    assert lines[0] == "2"
    first = lines[1].split()
    assert float(first[0]) == pytest.approx(0.5)
    assert float(first[1]) == pytest.approx(1.0 * BOHR_PER_ANGSTROM)
    assert float(first[1]) != pytest.approx(1.0)


def test_an_mm_site_is_a_point_charge_not_an_atom(tmp_path: Path):
    """Without a large hardness xTB gives each site a default element's
    chemical hardness and lets it screen the QM density."""

    path = tmp_path / "pcharge"
    write_embedding([1.0], np.array([[1.0, 2.0, 3.0]]), path)

    assert float(path.read_text().splitlines()[1].split()[4]) == POINT_CHARGE_HARDNESS


def test_mismatched_charges_and_positions_are_refused(tmp_path: Path):
    with pytest.raises(XtbEspError, match="one finite charge and one position"):
        write_embedding([1.0, 2.0], np.array([[0.0, 0.0, 0.0]]), tmp_path / "p")


def test_non_finite_embedding_is_refused(tmp_path: Path):
    with pytest.raises(XtbEspError, match="finite"):
        write_embedding([float("nan")], np.array([[0.0, 0.0, 0.0]]), tmp_path / "p")


def _grid() -> np.ndarray:
    return np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])


def _esp_file(path: Path, grid: np.ndarray, values) -> None:
    rows = np.hstack([grid * BOHR_PER_ANGSTROM, np.asarray(values)[:, None]])
    np.savetxt(path, rows)


def test_an_esp_matching_the_requested_grid_is_accepted(tmp_path: Path):
    grid = _grid()
    path = tmp_path / "xtb_esp.dat"
    _esp_file(path, grid, [0.1, -0.2, 0.3])

    assert parse_esp(path, grid) == pytest.approx([0.1, -0.2, 0.3])


def test_a_reordered_grid_is_refused(tmp_path: Path):
    """The potentials would otherwise be attached to the wrong points, and the
    fit that follows would look entirely normal."""

    grid = _grid()
    path = tmp_path / "xtb_esp.dat"
    _esp_file(path, grid[::-1], [0.1, -0.2, 0.3])

    with pytest.raises(XtbEspError, match="not the one it was given"):
        parse_esp(path, grid)


def test_a_grid_in_the_wrong_units_is_refused(tmp_path: Path):
    grid = _grid()
    path = tmp_path / "xtb_esp.dat"
    np.savetxt(path, np.hstack([grid, np.array([[0.1], [-0.2], [0.3]])]))

    with pytest.raises(XtbEspError, match="not the one it was given"):
        parse_esp(path, grid)


def test_a_truncated_esp_is_refused(tmp_path: Path):
    grid = _grid()
    path = tmp_path / "xtb_esp.dat"
    _esp_file(path, grid[:2], [0.1, -0.2])

    with pytest.raises(XtbEspError, match="one finite .* row per"):
        parse_esp(path, grid)
