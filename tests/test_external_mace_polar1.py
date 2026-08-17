from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mdprep.config.models import McpbMacePolar1HessianConfig
from mdprep.metals.mace_polar1_hessian import run_mace_polar1_hessian


@pytest.mark.external
@pytest.mark.mace
def test_real_mace_polar1_fixed_geometry_hessian(tmp_path):
    configured = os.environ.get("MDPREP_MACE_POLAR1_PYTHON")
    if not configured:
        pytest.skip(
            "set MDPREP_MACE_POLAR1_PYTHON to a Python environment containing "
            "MACE-POLAR-1 dependencies"
        )
    python = Path(configured).expanduser().absolute()
    if not python.is_file():
        pytest.skip(f"configured MACE-POLAR-1 Python is unavailable: {python}")
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nfixed MACE-POLAR-1 test\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.758602  0.000000  0.504284\n"
        "H -0.758602  0.000000  0.504284\n",
        encoding="utf-8",
    )

    result = run_mace_polar1_hessian(
        xyz_path=xyz,
        atom_count=3,
        charge=0,
        multiplicity=1,
        config=McpbMacePolar1HessianConfig(
            python_executable=str(python),
            model=os.environ.get("MDPREP_MACE_POLAR1_MODEL", "polar-1-m"),
            device=os.environ.get("MDPREP_MACE_POLAR1_DEVICE", "cpu"),
            accept_model_license=True,
        ),
        work_dir=tmp_path,
    )

    assert result.hessian_hartree_per_bohr2.shape == (9, 9)
    assert np.all(np.isfinite(result.hessian_hartree_per_bohr2))
    assert result.antisymmetry_max <= 1.0e-8
    assert result.finite_difference_relative_error <= 0.01
    assert result.command_result.runtime_seconds > 0
