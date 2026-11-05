"""SyrB2's iron fitted where the provisional force field puts it, built where it was.

The dimer of examples/18, prepared through the real toolchain with the
builders' geometry source: chain A's site minimised with a 12-6-4 iron before
MCPB.py fits it with a g-xTB Hessian, the fit's parameters then built onto
the original coordinates, and chain B built from the fit.

Skips cleanly without the tools; a few minutes with them.
"""

from __future__ import annotations

import importlib.util
import json
import os
from math import dist
from pathlib import Path

import pytest
import yaml

from mdprep.external.discovery import which_executable

pytestmark = [
    pytest.mark.external,
    pytest.mark.ambertools,
    pytest.mark.tleap,
    pytest.mark.pyscf,
    pytest.mark.openmm,
    pytest.mark.slow,
]

EXAMPLE = Path("examples/18_syrb2_dimer_shared_mcpb.yaml")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    missing = [
        name
        for name in ("tleap", "MCPB.py", "antechamber", "parmchk2", "propka3", "xtb", "parmed")
        if which_executable(name) is None
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} required for the bonded preparation")
    for module in ("pyscf", "openmm", "pdbfixer", "rdkit", "parmed"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for the bonded preparation")
    gxtb = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not gxtb or which_executable(gxtb) is None:
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")

    from mdprep.workflows.prepare import prepare_system

    work = tmp_path_factory.mktemp("site_minimized")
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["project"]["input_structure"] = str(
        (EXAMPLE.parent / data["project"]["input_structure"]).resolve()
    )
    data["project"]["output_dir"] = str(work / "prepared")
    for site in data["metals"]:
        if site["model"] == "bonded_mcpb":
            site["mcpb"]["provisional_nonbonded_parameter_set"] = "12_6_4"
            if "pyscf" in site["mcpb"]:
                site["mcpb"]["pyscf"]["geometry_source"] = "ff_minimized_site"
    manifest = work / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    prepare_system(manifest, stop_after="tleap")
    return work / "prepared"


@pytest.fixture(scope="module")
def site(prepared) -> dict:
    report = json.loads((prepared / "reports" / "metal_report.json").read_text())
    return report["bonded_mcpb_site"]


def test_the_site_was_minimised_within_its_limits(site):
    minimized = site["site_minimization"]

    assert minimized["problems"] == []
    assert minimized["residues_minimized"] > 1
    assert minimized["minimization"]["final_energy_kj_mol"] < minimized["minimization"][
        "initial_energy_kj_mol"
    ]
    # The iron had the 12-6-4 term it was minimised with.
    assert minimized["c4"]["nonzero_c4_coefficients"] > 0


def test_the_fit_was_taken_at_the_minimised_site(site):
    assert site["complete"]
    assert site["pyscf"]["small_model"]["geometry_status"] == "ff_minimized_site"


def test_the_structure_built_is_the_original_one(site, prepared):
    """The fitted structure is given its original coordinates back: what tleap
    builds is the input, not the minimised site."""

    from mdprep.structure.pdb import read_pdb

    built = read_pdb(site["restored_final_pdb_path"])
    before = read_pdb(
        next((prepared / "metals" / "a_fe2_900_site" / "hydrogenation").glob("*.hydrogenated.pdb"))
    )
    original = {atom.serial: atom for atom in before.atoms}
    worst = max(
        dist((atom.x, atom.y, atom.z), (original[atom.serial].x, original[atom.serial].y, original[atom.serial].z))
        for atom in built.atoms
    )
    assert worst < 0.002
    assert (prepared / "tleap" / "system.prmtop").is_file() or any(prepared.glob("**/*.prmtop"))
