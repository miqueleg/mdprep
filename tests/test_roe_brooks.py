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


# --------------------------------------------------------------------------
# The equilibration log
# --------------------------------------------------------------------------


def test_degrees_of_freedom_are_counted_like_openmm_counts_them():
    from mdprep.md.roe_brooks import degrees_of_freedom

    # Three waters, rigid: 9 atoms, 9 constraints, centre of mass removed.
    masses = [15.999, 1.008, 1.008] * 3
    assert degrees_of_freedom(masses, 9, True) == 27 - 9 - 3
    # A massless virtual site has no kinetic degrees of freedom.
    assert degrees_of_freedom([15.999, 0.0], 0, False) == 3


def test_temperature_and_density_come_out_in_the_units_the_log_promises():
    from mdprep.md.roe_brooks import density_g_ml, temperature_kelvin

    # 1000 degrees of freedom at 300 K hold 0.5 * 1000 * R * 300 kJ/mol.
    kinetic = 0.5 * 1000 * 0.00831446261815324 * 300.0
    assert temperature_kelvin(kinetic, 1000) == pytest.approx(300.0)
    assert temperature_kelvin(10.0, 0) == 0.0
    # A litre of water: 1000 g in 1e24 nm^3 is 1 g/mL.
    assert density_g_ml(1000.0 * 6.02214076e23, 1.0e24) == pytest.approx(1.0)
