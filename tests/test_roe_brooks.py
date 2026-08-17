import numpy as np
import pytest

from mdprep.md.roe_brooks import (
    RoeBrooksError,
    density_plateau_check,
    platform_candidates,
)


def test_auto_platform_selection_prefers_gpu_then_cpu():
    assert platform_candidates("auto", ["Reference", "CPU", "OpenCL", "CUDA"]) == [
        "CUDA",
        "OpenCL",
        "CPU",
        "Reference",
    ]


def test_explicit_unavailable_platform_fails_clearly():
    with pytest.raises(RoeBrooksError, match="unavailable"):
        platform_candidates("CUDA", ["CPU", "Reference"])


def test_density_plateau_accepts_flat_window():
    times = np.arange(0.0, 301.0, 1.0)
    densities = np.full(times.shape, 1.025)

    converged, info = density_plateau_check(
        times,
        densities,
        window_ps=300.0,
        report_interval_ps=1.0,
        slope_threshold=1.0e-6,
        mean_difference_threshold=0.02,
        minimum_points=50,
    )

    assert converged
    assert abs(float(info["slope_g_ml_ps"])) < 1.0e-12


def test_density_plateau_rejects_drifting_window():
    times = np.arange(0.0, 301.0, 1.0)
    densities = 1.0 + times * 1.0e-4

    converged, info = density_plateau_check(
        times,
        densities,
        window_ps=300.0,
        report_interval_ps=1.0,
        slope_threshold=1.0e-6,
        mean_difference_threshold=0.02,
        minimum_points=50,
    )

    assert not converged
    assert float(info["slope_g_ml_ps"]) > 1.0e-6
