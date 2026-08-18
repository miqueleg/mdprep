from math import pi

import numpy as np
import pytest

from mdprep.charges.esp_grid import (
    EspGridError,
    generate_merz_kollman_grid,
    read_grid_xyz,
    vdw_radius,
    write_grid_xyz,
)


def test_grid_generation_is_deterministic():
    coords = np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=float)
    first = generate_merz_kollman_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4, 1.6],
        point_density_per_square_angstrom=0.5,
        exclude_inside_vdw_scale=1.4,
        max_points=1000,
    )
    second = generate_merz_kollman_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4, 1.6],
        point_density_per_square_angstrom=0.5,
        exclude_inside_vdw_scale=1.4,
        max_points=1000,
    )

    assert np.allclose(first.points, second.points)
    assert first.atom_indices == second.atom_indices


def test_grid_points_are_outside_other_atom_exclusion_radius():
    coords = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    grid = generate_merz_kollman_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4],
        point_density_per_square_angstrom=1.0,
        exclude_inside_vdw_scale=1.4,
        max_points=1000,
    )

    radii = [vdw_radius("C"), vdw_radius("O")]
    for point, parent in zip(grid.points, grid.atom_indices, strict=True):
        for index, center in enumerate(coords):
            if index == parent:
                continue
            assert np.linalg.norm(point - center) >= radii[index] * 1.4


def test_grid_is_not_silently_thinned_to_max_points():
    coords = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(EspGridError, match="refusing to thin"):
        generate_merz_kollman_grid(
            elements=["C", "O"],
            coordinates=coords,
            vdw_scale_factors=[1.4, 1.6, 1.8],
            point_density_per_square_angstrom=1.0,
            exclude_inside_vdw_scale=1.4,
            max_points=25,
        )


def test_nonpositive_surface_density_fails_clearly():
    with pytest.raises(EspGridError, match="density must be finite and positive"):
        generate_merz_kollman_grid(
            elements=["C"],
            coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            vdw_scale_factors=[1.4],
            point_density_per_square_angstrom=0.0,
            exclude_inside_vdw_scale=1.4,
            max_points=10,
        )


def test_isolated_atom_uses_requested_surface_density_on_each_layer():
    scales = [1.4, 1.6, 1.8, 2.0]
    density = 1.0
    grid = generate_merz_kollman_grid(
        elements=["C"],
        coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
        vdw_scale_factors=scales,
        point_density_per_square_angstrom=density,
        exclude_inside_vdw_scale=1.4,
        max_points=1000,
    )

    expected = sum(max(6, round(4.0 * pi * (vdw_radius("C") * scale) ** 2 * density)) for scale in scales)
    assert len(grid.points) == expected
    assert grid.shell_scales.count(1.4) == round(4.0 * pi * (vdw_radius("C") * 1.4) ** 2)


def test_grid_xyz_writer_round_trips(tmp_path):
    coords = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    grid = generate_merz_kollman_grid(
        elements=["C", "O"],
        coordinates=coords,
        vdw_scale_factors=[1.4],
        point_density_per_square_angstrom=0.25,
        exclude_inside_vdw_scale=1.4,
        max_points=100,
    )
    path = tmp_path / "grid.xyz"

    write_grid_xyz(grid, path)

    assert np.allclose(read_grid_xyz(path), grid.points)
