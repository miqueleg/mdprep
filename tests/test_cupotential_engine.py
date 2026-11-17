"""molecular_dynamics.engine: cupotential, without NVIDIA's plugin installed.

The plugin is proprietary, CUDA-only and 600 MB, so nothing here needs it:
the manifest rules, the System edits (against a stand-in for the plugin's
classes), the refusal when it is missing, and the setup that installs it in
Colab, run with a fake nvidia-smi, getconf and micromamba.
test_external_cupotential.py runs the real one where it is installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest
from pydantic import ValidationError

from mdprep.config.models import MolecularDynamicsConfig
from mdprep.md import cupotential
from mdprep.workflows.runtime import (
    CUPOTENTIAL_ENVIRONMENT,
    CUPOTENTIAL_PLUGIN_SPEC,
    cupotential_environment_variables,
    cupotential_library_directory,
    cupotential_run_command,
    cupotential_setup_commands,
)

PRODUCTION = {
    "steps": 1000,
    "timestep_fs": 2.0,
    "trajectory_interval_steps": 100,
    "state_interval_steps": 100,
    "checkpoint_interval_steps": 1000,
}


def md(**settings):
    return MolecularDynamicsConfig.model_validate(
        {"enabled": True, "production": PRODUCTION, **settings}
    )


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_openmm_runs_the_md_unless_the_manifest_asks_otherwise():
    assert md().engine == "openmm"
    assert md(engine="cupotential").engine == "cupotential"


@pytest.mark.parametrize("platform", ["OpenCL", "CPU", "Reference", "HIP", "Metal"])
def test_cupotential_is_refused_on_a_platform_that_cannot_run_it(platform):
    with pytest.raises(ValidationError, match="CUDA platform only"):
        md(engine="cupotential", platform=platform)


def test_cupotential_is_refused_in_double_precision():
    with pytest.raises(ValidationError, match="single or mixed"):
        md(engine="cupotential", precision="double")


@pytest.mark.parametrize("platform", ["auto", "CUDA"])
def test_cupotential_is_accepted_where_cuda_can_be_used(platform):
    assert md(engine="cupotential", platform=platform, precision="single").platform == platform


# --------------------------------------------------------------------------
# A missing plugin is said, not worked around
# --------------------------------------------------------------------------


def test_a_missing_plugin_says_what_to_install_and_under_which_licence(monkeypatch):
    monkeypatch.setitem(sys.modules, "nv_openmm_plugin", None)

    with pytest.raises(cupotential.CuPotentialUnavailable) as raised:
        cupotential.load_plugin()

    message = str(raised.value)
    assert "pip install nv-openmm-plugin" in message
    assert cupotential.LICENCE_URL in message
    assert "engine: openmm" in message


def test_a_plugin_whose_libraries_did_not_load_says_so(monkeypatch):
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("libOpenMMNvidiaCUDA.so could not be loaded")

    monkeypatch.setitem(sys.modules, "nv_openmm_plugin", Broken())

    with pytest.raises(cupotential.CuPotentialUnavailable, match="did not load"):
        cupotential.load_plugin()


# --------------------------------------------------------------------------
# The System edits, against a stand-in for the plugin
# --------------------------------------------------------------------------


class FakePlugin:
    """Records what it was asked to build; returns ordinary OpenMM forces."""

    def __init__(self, mm):
        self.mm = mm
        self.nonbonded_sources = []
        self.barostats = []

    def NonbondedForce(self, source):
        self.nonbonded_sources.append(source)
        return self.mm.CustomExternalForce("0")

    def MonteCarloBarostat(self, pressure, temperature, frequency, seed=0):
        self.barostats.append((pressure, temperature, frequency, seed))
        return self.mm.CustomExternalForce("0")


@pytest.fixture
def mm():
    return pytest.importorskip("openmm")


def system_with(mm, *, tolerance=5e-4, barostat=True, nonbonded=1):
    from openmm import unit

    system = mm.System()
    for _ in range(3):
        system.addParticle(1.0)
    for _ in range(nonbonded):
        force = mm.NonbondedForce()
        force.setNonbondedMethod(mm.NonbondedForce.PME)
        force.setEwaldErrorTolerance(tolerance)
        for _ in range(3):
            force.addParticle(0.0, 0.3, 0.5)
        system.addForce(force)
    system.addForce(mm.CMMotionRemover())
    if barostat:
        force = mm.MonteCarloBarostat(1.0 * unit.atmosphere, 300 * unit.kelvin, 100)
        force.setRandomNumberSeed(4242)
        system.addForce(force)
    return system


def kinds(system):
    return [type(system.getForce(index)).__name__ for index in range(system.getNumForces())]


def test_the_nonbonded_force_is_replaced_by_the_plugins_copy_of_it(mm):
    plugin = FakePlugin(mm)
    system = system_with(mm)

    cupotential.use_cupotential_forces(system, plugin)

    assert "NonbondedForce" not in kinds(system)
    [source] = plugin.nonbonded_sources
    assert isinstance(source, mm.NonbondedForce)


def test_a_system_built_at_another_tolerance_is_refused(mm):
    """The plugin accepts 5e-4 only; mdprep's own runs use 1e-4."""

    with pytest.raises(cupotential.CuPotentialUnavailable, match="5e-4|0.0005"):
        cupotential.use_cupotential_forces(system_with(mm, tolerance=1e-4), FakePlugin(mm))


def test_two_nonbonded_forces_are_refused(mm):
    with pytest.raises(cupotential.CuPotentialUnavailable, match="exactly one"):
        cupotential.use_cupotential_forces(system_with(mm, nonbonded=2), FakePlugin(mm))


def test_the_dynamics_lose_the_com_remover_and_keep_their_barostat(mm):
    """The plugin's integrator removes centre-of-mass motion itself, and its
    barostat replaces OpenMM's with the same pressure, frequency and seed."""

    from openmm import unit

    plugin = FakePlugin(mm)
    system = system_with(mm)

    cupotential.use_cupotential_dynamics(system, plugin)

    assert "CMMotionRemover" not in kinds(system)
    assert "MonteCarloBarostat" not in kinds(system)
    [(pressure, temperature, frequency, seed)] = plugin.barostats
    # The plugin reads pressure in bar.
    assert pressure.value_in_unit(unit.bar) == pytest.approx(1.01325)
    assert temperature.value_in_unit(unit.kelvin) == pytest.approx(300)
    assert (frequency, seed) == (100, 4242)


def test_dynamics_without_a_barostat_get_none(mm):
    plugin = FakePlugin(mm)
    system = system_with(mm, barostat=False)

    cupotential.use_cupotential_dynamics(system, plugin)

    assert plugin.barostats == []
    assert "CMMotionRemover" not in kinds(system)


# --------------------------------------------------------------------------
# Installing it in Colab
# --------------------------------------------------------------------------


def run_setup(tmp_path, *, driver_cuda="13.0", glibc="2.39"):
    """The setup lines in bash, with fake nvidia-smi, getconf and micromamba."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    (bin_dir / "micromamba").write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$*" >> {log}\nexit 0\n', encoding="utf-8"
    )
    (bin_dir / "getconf").write_text(f'#!/bin/bash\necho "glibc {glibc}"\n', encoding="utf-8")
    if driver_cuda is not None:
        (bin_dir / "nvidia-smi").write_text(
            "#!/bin/bash\n"
            f'echo "| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: {driver_cuda} |"\n',
            encoding="utf-8",
        )
    for path in bin_dir.iterdir():
        path.chmod(0o755)
    # Only these on the PATH, so a real nvidia-smi on the test machine is not found.
    for tool in ("awk", "sed", "sort"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    commands = cupotential_setup_commands(mdprep_spec='"mdprep==9.9.9"', prefix=str(tmp_path))
    script = "\n".join(commands).replace(f"{tmp_path}/bin/micromamba", str(bin_dir / "micromamba"))
    result = subprocess.run(
        ["/bin/bash", "-c", script],
        env={"PATH": str(bin_dir), "HOME": os.environ.get("HOME", "/tmp")},
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result, calls


def test_the_plugin_gets_an_environment_of_its_own_with_the_same_mdprep(tmp_path):
    result, calls = run_setup(tmp_path, driver_cuda="13.4")

    assert result.returncode == 0, result.stderr
    create, install = calls
    assert create.startswith(f"create -y -q -n {CUPOTENTIAL_ENVIRONMENT}")
    assert "python=3.12" in create
    assert install.startswith(f"run -n {CUPOTENTIAL_ENVIRONMENT} pip install")
    assert "mdprep==9.9.9" in install and CUPOTENTIAL_PLUGIN_SPEC in install


@pytest.mark.parametrize("driver_cuda, nvrtc", [("13.0", "13.0"), ("13.2", "13.2")])
def test_a_cuda_13_driver_older_than_the_plugin_gets_its_own_nvrtc(tmp_path, driver_cuda, nvrtc):
    """The wheel brings NVRTC 13.4, whose kernels an older driver refuses."""

    result, calls = run_setup(tmp_path, driver_cuda=driver_cuda)

    assert result.returncode == 0, result.stderr
    assert calls[-1].endswith(f"nvidia-cuda-nvrtc=={nvrtc}.*")
    assert "--no-deps" in calls[-1]
    assert f"supports CUDA {driver_cuda}" in result.stdout


def test_a_driver_new_enough_keeps_the_plugins_nvrtc(tmp_path):
    result, calls = run_setup(tmp_path, driver_cuda="13.5")

    assert result.returncode == 0, result.stderr
    assert not any("nvidia-cuda-nvrtc" in call for call in calls)


def test_a_cuda_12_driver_stops_before_anything_is_installed(tmp_path):
    result, calls = run_setup(tmp_path, driver_cuda="12.4")

    assert result.returncode == 1
    assert calls == []
    assert "CUDA 13" in result.stderr and "engine openmm" in result.stderr


def test_a_machine_without_a_gpu_stops_before_anything_is_installed(tmp_path):
    result, calls = run_setup(tmp_path, driver_cuda=None)

    assert result.returncode == 1
    assert calls == []
    assert "NVIDIA GPU" in result.stderr


def test_a_glibc_older_than_the_wheel_stops_before_anything_is_installed(tmp_path):
    """Colab's Ubuntu 22.04 runtimes have glibc 2.35; the wheel needs 2.38."""

    result, calls = run_setup(tmp_path, glibc="2.35")

    assert result.returncode == 1
    assert calls == []
    assert "glibc 2.38" in result.stderr and "2.35" in result.stderr


def test_the_md_runs_with_the_plugins_cuda_libraries_first():
    """Colab's /usr/local/cuda is CUDA 12: found first, it is the wrong NVRTC."""

    variables = cupotential_environment_variables("/content", {"LD_LIBRARY_PATH": "/usr/lib64-nvidia"})

    assert variables["LD_LIBRARY_PATH"] == (
        f"{cupotential_library_directory('/content')}:/usr/lib64-nvidia"
    )
    assert cupotential_library_directory("/content").endswith(
        f"envs/{CUPOTENTIAL_ENVIRONMENT}/lib/python3.12/site-packages/nvidia/cu13/lib"
    )
    assert variables["PYTHONNOUSERSITE"] == "1"


def test_the_run_command_uses_the_plugins_environment():
    assert cupotential_run_command("system.md.yaml", prefix="/content") == [
        "/content/bin/micromamba", "run", "-n", CUPOTENTIAL_ENVIRONMENT,
        "mdprep", "run-md", "system.md.yaml",
    ]
