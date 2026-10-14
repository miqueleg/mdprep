"""The refinement report, for both backends.

The Markdown writer used to read report["ash"] unconditionally, so the
force-field backend -- which is the default -- raised KeyError after a
successful refinement. Nothing caught it: the only test that reached this
function monkeypatched it away.
"""

import json
from pathlib import Path

import pytest

from mdprep.refinement.report import write_refinement_reports


class _Result:
    """The writer's only contract with the workflow is to_report_dict()."""

    def __init__(self, report: dict) -> None:
        self._report = report

    def to_report_dict(self) -> dict:
        return self._report


def _shared_report() -> dict:
    return {
        "status": "complete",
        "provisional_ligand_charge_method": "am1bcc",
        "provisional_ligands": [],
        "provisional_system": {},
        "coordinate_transfer": {},
        "active_water_residue_count": 20,
        "refined_hydrogenated_pdb_path": "/tmp/refined.pdb",
        "warnings": ["a warning worth reading"],
        "selection": {
            "qm_atom_indices_zero_based": list(range(50)),
            "active_atom_indices_zero_based": list(range(286)),
            "movable_atom_policy": "active_region_hydrogens",
            "active_region_cutoff_angstrom": 4.0,
            "active_water_cutoff_angstrom": 8.0,
            "total_qm_charge": -1,
            "total_qm_multiplicity": 5,
            "charge_components": [
                {"label": "ligand:a_akg_304:A:AKG304", "charge": -2, "kind": "ligand"},
                {"label": "metal:a_fe2_302_site:A:FE2302", "charge": 2, "kind": "metal"},
                {
                    "label": "coordinating_ion:a_fe2_302_site:A:CL303",
                    "charge": -1,
                    "kind": "metal",
                },
            ],
        },
    }


def openmm_report() -> dict:
    """The force-field backend's report, as run_refinement_stage builds it."""

    return {
        **_shared_report(),
        "backend": "openmm",
        "moved": "all hydrogens; every heavy atom held fixed",
        "openmm": {
            "backend": "openmm",
            "seconds": 1.11,
            "atom_count": 4651,
            "hydrogen_count": 2423,
            "held_heavy_atoms": 2228,
            "max_heavy_atom_shift_angstrom": 0.0,
            "hydrogen_rms_displacement_angstrom": 0.5022,
            "hydrogen_max_displacement_angstrom": 1.8995,
            "initial_energy_kj_mol": -5379.0,
            "final_energy_kj_mol": -40213.7,
            "platform": "CUDA",
        },
    }


def ash_report() -> dict:
    return {
        **_shared_report(),
        "backend": "ash",
        "qm_method": "gfn2_xtb",
        "embedding": "electrostatic",
        "ash": {
            "final_energy_hartree": -1234.5,
            "command": ["python", "ash_refinement.py"],
            "cwd": "/tmp/ash",
            "returncode": 0,
            "runtime_seconds": 42.0,
        },
    }


@pytest.mark.parametrize("builder", [openmm_report, ash_report], ids=["openmm", "ash"])
def test_both_backends_write_a_report(tmp_path: Path, builder):
    report = write_refinement_reports(
        _Result(builder()),
        json_path=tmp_path / "refinement_report.json",
        markdown_path=tmp_path / "refinement_report.md",
    )

    written = json.loads((tmp_path / "refinement_report.json").read_text(encoding="utf-8"))
    assert written == report

    markdown = (tmp_path / "refinement_report.md").read_text(encoding="utf-8")
    # The charge derivation is the part a reviewer checks against the manifest.
    assert "coordinating_ion:a_fe2_302_site:A:CL303: -1 (metal)" in markdown
    assert "Total QM charge: -1" in markdown
    assert "a warning worth reading" in markdown


def test_the_force_field_report_states_what_moved_and_what_did_not():
    """A refinement that silently moved a heavy atom would be a serious error,
    so the report says the shift explicitly rather than implying it."""

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_refinement_reports(
            _Result(openmm_report()),
            json_path=root / "r.json",
            markdown_path=root / "r.md",
        )
        markdown = (root / "r.md").read_text(encoding="utf-8")

    assert "Backend: OpenMM force field" in markdown
    assert "no quantum step" in markdown
    assert "Largest heavy-atom shift (Å): 0.0" in markdown
    assert "Heavy atoms held: 2228" in markdown
    assert "Hydrogens refined: 2423" in markdown
    # Nothing from the QM/MM backend should leak into it.
    assert "ASH" not in markdown
    assert "External command" not in markdown


def test_the_qmmm_report_still_records_the_external_command():
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_refinement_reports(
            _Result(ash_report()),
            json_path=root / "r.json",
            markdown_path=root / "r.md",
        )
        markdown = (root / "r.md").read_text(encoding="utf-8")

    assert "Backend: ASH QM/MM" in markdown
    assert "QM method: gfn2_xtb" in markdown
    assert "Final energy (Eh): -1234.5" in markdown
    assert "Command: `python ash_refinement.py`" in markdown
    assert "Return code: 0" in markdown
