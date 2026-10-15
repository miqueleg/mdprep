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
