"""Both copies of SyrB2 (PDB 2FCT) through the real toolchain: one fit, two sites.

The whole preparation -- protonation, QM/MM refinement, MCPB.py with a g-xTB
Hessian and a GFN2 ESP, tleap -- on the dimer of examples/18. Chain A's iron
site is fitted; chain B's is built from the fit. Chain B numbers its iron,
cofactor, chloride and water differently from chain A, so a copy made by
residue number would bond the wrong atoms or none.

Skips cleanly without the tools; about a minute and a half with them.
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

EXAMPLE = Path("examples/18_syrb2_dimer_shared_mcpb.yaml")
TEMPLATES = {"FE1", "AG1", "HD1", "HD2", "WT1", "CL1"}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    missing = [
        name
        for name in ("tleap", "MCPB.py", "antechamber", "parmchk2", "propka3", "xtb")
        if which_executable(name) is None
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} required for the dimer preparation")
    for module in ("pyscf", "openmm", "pdbfixer", "rdkit"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for the dimer preparation")
    gxtb = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not gxtb or which_executable(gxtb) is None:
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")

    from mdprep.workflows.prepare import prepare_system

    work = tmp_path_factory.mktemp("dimer")
    data = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    data["project"]["input_structure"] = str(
        (EXAMPLE.parent / data["project"]["input_structure"]).resolve()
    )
    data["project"]["output_dir"] = str(work / "prepared")
    manifest = work / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    prepare_system(manifest, stop_after="tleap")
    return work / "prepared"


def test_chain_b_is_built_from_chain_a_fit(prepared):
    report = json.loads((prepared / "reports" / "metal_report.json").read_text())
    site = report["bonded_mcpb_site"]
    [copy] = site["copies"]

    assert copy["site_id"] == "b_fe2_899_site"
    renamed = {(r["original_resname"], r["resid"]): r["final_resname"] for r in copy["residue_renames"]}
    assert renamed[("FE2", 899)] == "FE1"
    assert renamed[("AKG", 898)] == "AG1"
    assert set(renamed.values()) == TEMPLATES
    # Only chain A was fitted: one MCPB.py run of four steps.
    assert len(site["runs"]) == 4


def test_both_irons_are_bonded_and_carry_the_same_parameters(prepared):
    from mdprep.analysis import charges
    from mdprep.analysis.prmtop import read_prmtop

    topology = read_prmtop(prepared / "final" / "system.prmtop")
    sites = charges.metal_sites(topology)

    assert len(sites) == 2
    assert [len(site["bonds"]) for site in sites] == [6, 6]
    assert [site["total"] for site in sites] == [pytest.approx(-1.0, abs=1e-3)] * 2
    templates: dict[str, list[tuple]] = {}
    for residue in topology.residues:
        if residue.name in TEMPLATES:
            atoms = topology.atoms[residue.start : residue.stop]
            templates.setdefault(residue.name, []).append(
                tuple((atom.name, atom.type, round(atom.charge, 6)) for atom in atoms)
            )
    assert set(templates) == TEMPLATES
    assert all(len(copies) == 2 and copies[0] == copies[1] for copies in templates.values())


def test_the_dimer_validates(prepared):
    report = json.loads((prepared / "reports" / "validation_report.json").read_text())

    assert report["errors"] == []
    assert report["parmed"]["status"] == "ok"
    assert report["openmm"]["status"] == "ok"
    assert [check["ligand_id"] for check in report["ligand_presence_checks"]] == [
        "a_akg_897",
        "b_akg_898",
    ]
    assert all(check["ok"] for check in report["ligand_presence_checks"])
