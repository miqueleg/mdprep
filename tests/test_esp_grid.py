import numpy as np
import pytest

from mdprep.charges.esp_grid import EspGridError, generate_connolly_grid, read_grid_xyz, vdw_radius, write_grid_xyz


def test_grid_generation_is_deterministic():
    coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=float)
    first = generate_connolly_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4, 1.6],
        points_per_atom_per_shell=12,
        exclude_inside_vdw_scale=1.1,
        max_points=1000,
    )
    second = generate_connolly_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4, 1.6],
        points_per_atom_per_shell=12,
        exclude_inside_vdw_scale=1.1,
        max_points=1000,
    )

    assert np.allclose(first.points, second.points)
    assert first.atom_indices == second.atom_indices


def test_grid_points_are_outside_other_atom_exclusion_radius():
    coords = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    grid = generate_connolly_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4],
        points_per_atom_per_shell=16,
        exclude_inside_vdw_scale=1.1,
        max_points=1000,
    )

    radii = [vdw_radius("C"), vdw_radius("O")]
    for point, parent in zip(grid.points, grid.atom_indices, strict=True):
        for index, center in enumerate(coords):
            if index == parent:
                continue
            assert np.linalg.norm(point - center) >= radii[index] * 1.1


def test_grid_max_points_is_respected():
    coords = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    grid = generate_connolly_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4, 1.6, 1.8],
        points_per_atom_per_shell=40,
        exclude_inside_vdw_scale=1.1,
        max_points=25,
    )

    assert len(grid.points) == 25


def test_too_few_points_fails_clearly():
    with pytest.raises(EspGridError, match="Too few ESP grid points"):
        generate_connolly_grid(
            elements=["C"],
            coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            vdw_scale_factors=[1.4],
            points_per_atom_per_shell=1,
            exclude_inside_vdw_scale=1.1,
            max_points=10,
        )


def test_grid_xyz_writer_round_trips(tmp_path):
    coords = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    grid = generate_connolly_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4],
        points_per_atom_per_shell=8,
        exclude_inside_vdw_scale=1.1,
        max_points=100,
    )
    path = tmp_path / "grid.xyz"

    write_grid_xyz(grid, path)

    assert np.allclose(read_grid_xyz(path), grid.points)
