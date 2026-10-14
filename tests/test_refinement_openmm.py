"""Force-field hydrogen refinement.

The refinement stage only ever moves hydrogens, so the quantum method the
QM/MM backend spends on that is answering a question a force field already
answers. These tests pin the two properties that make the cheap backend safe
to default to: it moves hydrogens, and it moves nothing else.

The charges used are provisional in both backends. Whatever QMMESP or MCPB
calculation the manifest configures still runs afterwards and its charges are
what reach the final build.
"""

from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig, OpenmmRefinementConfig
from mdprep.refinement.openmm_minimizer import (
    HEAVY_ATOM_TOLERANCE_ANGSTROM,
    OpenmmRefinementError,
    run_openmm_refinement,
)


AMBER = Path("tests/data/amber")
PRMTOP = AMBER / "ace_ala_gly_nme.prmtop"
INPCRD = AMBER / "ace_ala_gly_nme.inpcrd"

openmm = pytest.importorskip("openmm", reason="OpenMM is required for this backend")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return run_openmm_refinement(
        prmtop_path=PRMTOP,
        inpcrd_path=INPCRD,
        config=OpenmmRefinementConfig(platform="CPU"),
        work_dir=tmp_path_factory.mktemp("openmm_refinement"),
    )


def test_every_heavy_atom_stays_exactly_where_it_was(run):
    """The experimental coordinates are the one thing that must not change."""
    assert run.max_heavy_atom_shift_angstrom <= HEAVY_ATOM_TOLERANCE_ANGSTROM
    assert run.held_heavy_atoms == 14


def test_the_hydrogens_actually_move(run):
    """A refinement that changes nothing is not a refinement."""
    assert run.hydrogen_count == 15
    assert run.hydrogen_rms_displacement_angstrom > 0.0


def test_the_energy_goes_down(run):
    assert run.final_energy_kj_mol < run.initial_energy_kj_mol


def test_it_returns_a_coordinate_for_every_atom(run):
    """The caller transfers these straight onto the structure."""
    assert len(run.optimized_coordinates_angstrom) == run.atom_count == 29
    assert all(len(position) == 3 for position in run.optimized_coordinates_angstrom)


def test_it_is_fast_enough_to_be_the_default(run):
    """The whole point is that this is cheap next to GFN2 QM/MM."""
    assert run.seconds < 30.0


def test_the_run_is_recorded(run):
    import json

    assert run.report_path.is_file()
    report = json.loads(run.report_path.read_text(encoding="utf-8"))
    assert report["backend"] == "openmm"
    assert report["held_heavy_atoms"] == 14


def test_an_unavailable_platform_fails_clearly(tmp_path, monkeypatch):
    """An explicit platform must not silently fall back to another one."""
    import openmm as mm

    monkeypatch.setattr(mm.Platform, "getNumPlatforms", staticmethod(lambda: 0))

    with pytest.raises(OpenmmRefinementError, match="unavailable"):
        run_openmm_refinement(
            prmtop_path=PRMTOP,
            inpcrd_path=INPCRD,
            config=OpenmmRefinementConfig(platform="CUDA"),
            work_dir=tmp_path,
        )


def test_auto_selects_a_platform_that_exists(run):
    import openmm as mm

    available = {
        mm.Platform.getPlatform(i).getName()
        for i in range(mm.Platform.getNumPlatforms())
    }
    assert run.platform in available


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def manifest_with(refinement: dict) -> dict:
    return {
        "project": {"name": "t", "input_structure": "in.pdb", "output_dir": "out"},
        "structure": {"keep_crystal_waters": True},
        "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
        "protonation": {"method": "manual_only"},
        "disulfides": {},
        "ligands": [
            {
                "id": "lig",
                "selector": {"chain": "A", "resname": "LIG", "resid": 1, "icode": None},
                "net_charge": 0,
                # A QM region requires this explicitly.
                "multiplicity": 1,
                "atom_types": "gaff2",
                "charge_method": "am1bcc",
            }
        ],
        "metals": [],
        "refinement": refinement,
        "solvation": {},
        "validation": {},
    }


def test_the_force_field_backend_is_the_default():
    """GFN2 QM/MM for hydrogen placement is an expensive default."""
    manifest = ManifestConfig.model_validate(
        manifest_with({"enabled": True, "qm_components": {"ligands": ["lig"]}})
    )

    assert manifest.refinement.backend == "openmm"


def test_the_default_backend_needs_no_ash_settings():
    manifest = ManifestConfig.model_validate(
        manifest_with({"enabled": True, "qm_components": {"ligands": ["lig"]}})
    )

    assert manifest.refinement.ash is None


def test_the_force_field_backend_refuses_to_move_heavy_atoms():
    """It only knows how to move hydrogens, so it must not accept otherwise."""
    with pytest.raises(ValueError, match="only moves hydrogens"):
        ManifestConfig.model_validate(
            manifest_with(
                {
                    "enabled": True,
                    "qm_components": {"ligands": ["lig"]},
                    "movable_atoms": "all_active_region",
                }
            )
        )


def test_the_ash_backend_still_requires_its_settings():
    with pytest.raises(ValueError, match="requires refinement.ash"):
        ManifestConfig.model_validate(
            manifest_with(
                {
                    "enabled": True,
                    "backend": "ash",
                    "qm_components": {"ligands": ["lig"]},
                }
            )
        )


def test_the_ash_backend_is_still_available():
    manifest = ManifestConfig.model_validate(
        manifest_with(
            {
                "enabled": True,
                "backend": "ash",
                "qm_components": {"ligands": ["lig"]},
                "ash": {"python_executable": "python"},
            }
        )
    )

    assert manifest.refinement.backend == "ash"

