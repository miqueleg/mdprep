import numpy as np
import pytest

from mdprep.charges.resp_fit import RespFitError, fit_resp_charges


def synthetic_esp(atom_coords, grid_coords, charges):
    design = 1.0 / np.linalg.norm(grid_coords[:, None, :] - atom_coords[None, :, :], axis=2)
    return design @ charges


def test_constrained_fit_enforces_total_charge_exactly():
    atoms = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    grid = np.asarray([[0.0, 3.0, 0.0], [2.0, 3.0, 0.0], [1.0, -3.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    target = np.asarray([-0.4, 0.4], dtype=float)

    result = fit_resp_charges(
        atom_coordinates=atoms,
        grid_coordinates=grid,
        esp_values=synthetic_esp(atoms, grid, target),
        total_charge=0.0,
        restraint="none",
    )

    assert result.charge_sum_final == pytest.approx(0.0, abs=1.0e-10)
    assert result.charges == pytest.approx(target, abs=1.0e-8)


def test_resp_like_restraint_reduces_charge_magnitude():
    atoms = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    grid = np.asarray([[0.0, 3.0, 0.0], [2.0, 3.0, 0.0], [1.0, -3.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
    target = np.asarray([-0.8, 0.8], dtype=float)
    esp = synthetic_esp(atoms, grid, target)

    unrestrained = fit_resp_charges(
        atom_coordinates=atoms,
        grid_coordinates=grid,
        esp_values=esp,
        total_charge=0.0,
        restraint="none",
    )
    restrained = fit_resp_charges(
        atom_coordinates=atoms,
        grid_coordinates=grid,
        esp_values=esp,
        total_charge=0.0,
        restraint="resp",
        restraint_a=0.2,
        restraint_b=0.1,
    )

    assert max(abs(value) for value in restrained.charges) < max(abs(value) for value in unrestrained.charges)
    assert restrained.rms_error >= 0.0
    assert restrained.warnings


def test_too_few_esp_points_fails_clearly():
    with pytest.raises(RespFitError, match="Too few ESP points"):
        fit_resp_charges(
            atom_coordinates=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            grid_coordinates=np.asarray([[0.0, 2.0, 0.0]], dtype=float),
            esp_values=np.asarray([0.0], dtype=float),
            total_charge=0.0,
        )


def test_grid_point_on_atom_fails_clearly():
    with pytest.raises(RespFitError, match="atom center"):
        fit_resp_charges(
            atom_coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            grid_coordinates=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            esp_values=np.asarray([0.0, 0.0], dtype=float),
            total_charge=0.0,
        )
