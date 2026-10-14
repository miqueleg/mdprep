"""Choosing an OpenMM platform when the fast ones are not usable.

OpenMM ships the CUDA plugin wherever it was built with CUDA, so a machine
with no GPU -- a CI runner, a cluster login node -- still lists CUDA among its
platforms and only fails when a context is created on one. The force-field
refinement is the default backend, so picking a platform by name and hoping
turns that into an unhandled CUDA_ERROR_NO_DEVICE partway through a run.

These exercise the selection itself with stand-in objects, so they run
everywhere rather than only where OpenMM is installed.
"""

import pytest

from mdprep.config.models import OpenmmRefinementConfig
from mdprep.refinement.openmm_minimizer import (
    OpenmmRefinementError,
    _build_simulation,
    platform_candidates,
)



def test_auto_prefers_accelerators_but_only_among_those_present():
    assert platform_candidates("auto", ["Reference", "CPU", "CUDA", "OpenCL"]) == [
        "CUDA",
        "OpenCL",
        "CPU",
        "Reference",
    ]
    assert platform_candidates("auto", ["Reference", "CPU"]) == ["CPU", "Reference"]


def test_an_explicit_platform_is_never_quietly_replaced():
    assert platform_candidates("CPU", ["CPU", "CUDA"]) == ["CPU"]
    with pytest.raises(OpenmmRefinementError, match="unavailable"):
        platform_candidates("CUDA", ["CPU", "Reference"])


def test_a_registered_but_unusable_platform_falls_through_to_the_next():
    """OpenMM ships the CUDA plugin wherever it was built with CUDA, so a
    machine with no GPU still lists CUDA and only fails when a context is
    created on it. Selecting by name and hoping gives CUDA_ERROR_NO_DEVICE.
    """

    class _FakePlatform:
        def __init__(self, name: str) -> None:
            self._name = name

        def getName(self) -> str:
            return self._name

    names = ["Reference", "CPU", "CUDA"]

    class _FakeMM:
        class Platform:
            @staticmethod
            def getNumPlatforms() -> int:
                return len(names)

            @staticmethod
            def getPlatform(index: int) -> _FakePlatform:
                return _FakePlatform(names[index])

            @staticmethod
            def getPlatformByName(name: str) -> _FakePlatform:
                return _FakePlatform(name)

        @staticmethod
        def VerletIntegrator(step):  # noqa: N802 - mirrors OpenMM's name
            return object()

    attempted: list[str] = []

    def fake_simulation(topology, system, integrator, platform):
        attempted.append(platform.getName())
        if platform.getName() == "CUDA":
            raise Exception("Error initializing CUDA: CUDA_ERROR_NO_DEVICE (100)")
        return f"simulation on {platform.getName()}"

    class _FakeUnit:
        femtoseconds = 1.0

    simulation, platform = _build_simulation(
        _FakeMM,
        _FakeUnit,
        fake_simulation,
        topology=None,
        system=None,
        config=OpenmmRefinementConfig(),
    )

    assert attempted == ["CUDA", "CPU"]
    assert platform.getName() == "CPU"
    assert simulation == "simulation on CPU"


def test_an_explicitly_requested_platform_that_fails_is_reported_not_replaced():
    class _FakePlatform:
        def __init__(self, name: str) -> None:
            self._name = name

        def getName(self) -> str:
            return self._name

    class _FakeMM:
        class Platform:
            @staticmethod
            def getNumPlatforms() -> int:
                return 2

            @staticmethod
            def getPlatform(index: int) -> _FakePlatform:
                return _FakePlatform(["CPU", "CUDA"][index])

            @staticmethod
            def getPlatformByName(name: str) -> _FakePlatform:
                return _FakePlatform(name)

        @staticmethod
        def VerletIntegrator(step):  # noqa: N802
            return object()

    def always_fails(topology, system, integrator, platform):
        raise Exception("CUDA_ERROR_NO_DEVICE (100)")

    class _FakeUnit:
        femtoseconds = 1.0

    with pytest.raises(OpenmmRefinementError, match="CUDA.*failed"):
        _build_simulation(
            _FakeMM,
            _FakeUnit,
            always_fails,
            topology=None,
            system=None,
            config=OpenmmRefinementConfig(platform="CUDA"),
        )
