"""CYP119 (PDB 1IO7) through the real toolchain: a heme iron as a bonded site.

The whole preparation of examples/19 -- protonation with the cysteine as a
thiolate, the force-field relaxation, the porphyrin's provisional charges,
MCPB.py with a g-xTB Hessian smeared at 1000 K and a GFN2 ESP, tleap -- on
chain A. The iron is read out of the heme as the structure is read, and comes
back bonded to the porphyrin's four nitrogens, the cysteine and the water.

Skips cleanly without the tools; about eight minutes with them, most of it
AM1-BCC charges for the porphyrin.
"""

from __future__ import annotations

import importlib.util
import json
import os
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

EXAMPLE = Path("examples/19_cyp119_heme_mcpb.yaml")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    missing = [
        name
        for name in ("tleap", "MCPB.py", "antechamber", "parmchk2", "propka3", "xtb")
        if which_executable(name) is None
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} required for the heme preparation")
    for module in ("pyscf", "openmm", "pdbfixer", "rdkit"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for the heme preparation")
    gxtb = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not gxtb or which_executable(gxtb) is None:
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")

    from mdprep.workflows.prepare import prepare_system

    work = tmp_path_factory.mktemp("heme")
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["project"]["input_structure"] = str(
        (EXAMPLE.parent / data["project"]["input_structure"]).resolve()
    )
    data["project"]["output_dir"] = str(work / "prepared")
    manifest = work / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    prepare_system(manifest, stop_after="tleap")
    return work / "prepared"


def test_the_iron_is_bonded_to_the_porphyrin_the_cysteine_and_the_water(prepared):
    from mdprep.analysis import charges
    from mdprep.analysis.prmtop import read_prmtop

    topology = read_prmtop(prepared / "final" / "system.prmtop")
    (site,) = charges.metal_sites(topology)

    assert len(site["bonds"]) == 6
    # Fe(III) + porphyrin(4-) + thiolate(1-) + water.
    assert site["total"] == pytest.approx(-2.0, abs=1e-3)
    names = {residue.name for residue in topology.residues}
    assert {"FE1", "HM1", "CM1", "WT1"} <= names
    assert "HEM" not in names and "FE" not in names


def test_the_smearing_reached_the_hessian(prepared):
    stdout = next(prepared.glob("metals/*/mcpb/pyscf_small/*.hessian.stdout.txt")).read_text()

    assert "--etemp 1000" in stdout
    assert "SCF not converged" not in stdout


def test_the_split_is_reported_and_the_system_validates(prepared):
    structure = json.loads((prepared / "reports" / "structure_report.json").read_text())
    assert any("A:HEM1401 holds a metal atom" in warning for warning in structure["warnings"])

    report = json.loads((prepared / "reports" / "validation_report.json").read_text())
    assert report["errors"] == []
    assert report["parmed"]["status"] == "ok"
    assert report["openmm"]["status"] == "ok"
    assert [check["ligand_id"] for check in report["ligand_presence_checks"]] == ["a_hem_1401"]
    assert all(check["ok"] for check in report["ligand_presence_checks"])
