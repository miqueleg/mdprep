from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mdprep.config.models import McpbGxtbHessianConfig, McpbXtbHessianConfig
from mdprep.metals.gxtb_hessian import run_gxtb_hessian


@pytest.mark.external
@pytest.mark.xtb
def test_real_gxtb_fixed_geometry_hessian(tmp_path):
    configured = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not configured:
        pytest.skip(
            "set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary"
        )
    executable = Path(configured).expanduser().resolve()
    if not executable.is_file():
        pytest.skip(f"configured g-xTB executable is unavailable: {executable}")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nfixed g-xTB test\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.758602  0.000000  0.504284\n"
        "H -0.758602  0.000000  0.504284\n",
        encoding="utf-8",
    )

    result = run_gxtb_hessian(
        xyz_path=xyz,
        atom_count=3,
        charge=0,
        spin=0,
        config=McpbGxtbHessianConfig(
            executable=str(executable),
            expected_executable_sha256=digest,
            accuracy=0.2,
            num_threads=1,
        ),
        work_dir=tmp_path,
    )

    assert result.hessian_hartree_per_bohr2.shape == (9, 9)
    assert np.all(np.isfinite(result.hessian_hartree_per_bohr2))
    assert result.antisymmetry_max <= 1.0e-5
    assert result.command_result.runtime_seconds > 0


@pytest.mark.external
@pytest.mark.xtb
def test_real_gfn2_fixed_geometry_hessian(tmp_path):
    configured = os.environ.get("MDPREP_XTB_EXECUTABLE") or shutil.which("xtb")
    if not configured:
        pytest.skip("xTB executable is unavailable")
    executable = Path(configured).expanduser().resolve()
    if not executable.is_file():
        pytest.skip(f"configured xTB executable is unavailable: {executable}")
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    xyz = tmp_path / "water.xyz"
    xyz.write_text(
        "3\nfixed GFN2-xTB test\n"
        "O  0.000000  0.000000  0.000000\n"
        "H  0.758602  0.000000  0.504284\n"
        "H -0.758602  0.000000  0.504284\n",
        encoding="utf-8",
    )

    result = run_gxtb_hessian(
        xyz_path=xyz,
        atom_count=3,
        charge=0,
        spin=0,
        config=McpbXtbHessianConfig(
            model="gfn2",
            executable=str(executable),
            expected_executable_sha256=digest,
            accuracy=0.2,
            num_threads=1,
        ),
        work_dir=tmp_path,
    )

    assert result.model == "gfn2"
    assert result.hessian_hartree_per_bohr2.shape == (9, 9)
    assert np.all(np.isfinite(result.hessian_hartree_per_bohr2))
    assert result.antisymmetry_max <= 1.0e-5
    assert result.command_result.runtime_seconds > 0


@pytest.mark.external
def test_the_pinned_gxtb_release_still_exists_with_the_checksum_we_claim():
    """The Colab setup cell downloads this exact file and verifies it.

    If the release is retagged, deleted, or the asset rebuilt, every student
    running a metal site gets a checksum failure in the setup cell, and the
    only place that is visible is here. Fetches the published .sha256, which
    is a few bytes, rather than the 40 MB archive.
    """

    import urllib.error
    import urllib.request

    from mdprep.workflows.runtime import (
        GXTB_ARCHIVE,
        GXTB_RELEASE_URL,
        GXTB_SHA256,
    )

    url = f"{GXTB_RELEASE_URL}/{GXTB_ARCHIVE}.sha256"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            published = response.read().decode("utf-8").split()[0]
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"cannot reach the g-xTB release: {exc}")

    assert published == GXTB_SHA256, (
        f"The g-xTB release asset changed. {url} now publishes {published}, "
        f"but mdprep pins {GXTB_SHA256}. Every Colab metal run will fail its "
        "checksum until this is updated."
    )


DHP_SMALL_MODEL = Path(__file__).parent / "data" / "metals" / "dhp_2qfk_small_model.xyz"


@pytest.mark.external
@pytest.mark.xtb
@pytest.mark.slow
def test_the_scf_search_finds_the_lower_state_of_a_high_spin_heme(tmp_path):
    """2QFK's aquo Fe(III), spin 6: straight at 1000 K a single-threaded SCF
    settles about 27 kcal/mol above the states the annealed starts reach.
    Two and a half minutes on one thread."""

    from mdprep.metals.gxtb_hessian import _search_scf_states

    configured = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not configured or not Path(configured).expanduser().is_file():
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")
    xyz = tmp_path / DHP_SMALL_MODEL.name
    shutil.copyfile(DHP_SMALL_MODEL, xyz)
    config = McpbXtbHessianConfig(
        executable=str(Path(configured).expanduser()),
        electronic_temperature_kelvin=1000.0,
        max_scf_iterations=3000,
        scf_annealing_kelvin=[2000.0, 3000.0, 5000.0],
    )

    starts = _search_scf_states(
        executable=Path(configured).expanduser().resolve(),
        source_xyz=xyz,
        work=tmp_path,
        charge=-1,
        spin=5,
        config=config,
        environment=dict(os.environ),
    )

    lowest = starts[0]
    (direct,) = [start for start in starts if start.label == "direct"]
    assert lowest.label != "direct"
    assert direct.converged
    assert direct.energy_hartree - lowest.energy_hartree > 0.03
    assert (tmp_path / "scf_start_5000K" / "xtbrestart").is_file()
