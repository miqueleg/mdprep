import pytest
from pydantic import ValidationError

from mdprep.config.models import ManifestConfig


def base_manifest() -> dict:
    return {
        "project": {
            "name": "test",
            "input_structure": "input.pdb",
            "output_dir": "prepared/test",
        },
        "structure": {},
        "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
        "protonation": {"method": "manual_only"},
        "disulfides": {},
        "ligands": [],
        "metals": [],
        "solvation": {},
        "validation": {},
    }


def enabled_md() -> dict:
    return {
        "enabled": True,
        "platform": "auto",
        "production": {
            "steps": 50_000,
            "trajectory_interval_steps": 10_000,
            "state_interval_steps": 5_000,
            "checkpoint_interval_steps": 25_000,
        },
    }


def test_enabled_md_requires_explicit_production_steps():
    data = base_manifest()
    data["molecular_dynamics"] = {"enabled": True}

    with pytest.raises(ValidationError, match="requires an explicit production block"):
        ManifestConfig.model_validate(data)


def test_manifest_accepts_roe_brooks_production_step_controls():
    data = base_manifest()
    data["molecular_dynamics"] = enabled_md()

    manifest = ManifestConfig.model_validate(data)

    assert manifest.molecular_dynamics.production is not None
    assert manifest.molecular_dynamics.production.steps == 50_000
    assert manifest.molecular_dynamics.production.timestep_fs == 2.0
    assert manifest.molecular_dynamics.platform == "auto"


def test_production_reporting_interval_cannot_exceed_run():
    data = base_manifest()
    settings = enabled_md()
    settings["production"]["trajectory_interval_steps"] = 50_001
    data["molecular_dynamics"] = settings

    with pytest.raises(ValidationError, match="trajectory_interval_steps"):
        ManifestConfig.model_validate(data)


def test_density_window_must_fit_inside_maximum_duration():
    data = base_manifest()
    settings = enabled_md()
    settings["density"] = {"window_ps": 300.0, "maximum_duration_ns": 0.2}
    data["molecular_dynamics"] = settings

    with pytest.raises(ValidationError, match="window_ps may not exceed"):
        ManifestConfig.model_validate(data)


def test_device_index_is_rejected_for_explicit_cpu():
    data = base_manifest()
    settings = enabled_md()
    settings.update({"platform": "CPU", "device_index": 0})
    data["molecular_dynamics"] = settings

    with pytest.raises(ValidationError, match="device_index"):
        ManifestConfig.model_validate(data)
