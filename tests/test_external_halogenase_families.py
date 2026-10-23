"""Two halogenating enzymes through the real toolchain.

* examples/20, an unspecific peroxygenase (5OXU): five glycans and fourteen
  crystallisation additives left out, a heme-thiolate iron bonded with
  MCPB.py, and a magnesium on the heme's propionate kept nonbonded -- which
  the QM/MM refinement of the iron's site has to treat as MM environment.
* examples/21, the flavin-dependent halogenase PrnA (2AR8): FAD from its
  proposed SMILES, the chlorinated product from a typed one.

Skip cleanly without the tools; the peroxygenase takes about twenty minutes,
most of it AM1-BCC charges for the heme, and PrnA about ten, most of it
FAD's.
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
    pytest.mark.openmm,
    pytest.mark.slow,
]

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _require(executables, modules) -> None:
    missing = [name for name in executables if which_executable(name) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} required")
    for module in modules:
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required")


def _prepare(example: str, work: Path) -> Path:
    from mdprep.workflows.prepare import prepare_system

    data = yaml.safe_load((EXAMPLES / example).read_text(encoding="utf-8"))
    data["project"]["input_structure"] = str(
        (EXAMPLES / data["project"]["input_structure"]).resolve()
    )
    data["project"]["output_dir"] = str(work / "prepared")
    manifest = work / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    prepare_system(manifest, stop_after="tleap")
    return work / "prepared"


@pytest.fixture(scope="module")
def peroxygenase(tmp_path_factory) -> Path:
    _require(
        ("tleap", "MCPB.py", "antechamber", "parmchk2", "propka3", "xtb"),
        ("pyscf", "openmm", "rdkit"),
    )
    gxtb = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not gxtb or which_executable(gxtb) is None:
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")
    return _prepare("20_upo_5oxu_heme_glycans.yaml", tmp_path_factory.mktemp("upo"))


@pytest.fixture(scope="module")
def flavin_halogenase(tmp_path_factory) -> Path:
    _require(("tleap", "antechamber", "parmchk2", "propka3", "xtb"), ("openmm", "rdkit"))
    return _prepare("21_prna_fad_halogenase.yaml", tmp_path_factory.mktemp("prna"))


def test_the_glycans_and_additives_are_left_out_and_reported(peroxygenase):
    from mdprep.analysis.prmtop import read_prmtop

    structure = json.loads((peroxygenase / "reports" / "structure_report.json").read_text())
    removed = structure["residues_removed"]
    assert len(removed) == 19
    assert {item["resname"] for item in removed} == {"NAG", "PO4", "GOL", "MOH"}
    names = {residue.name for residue in read_prmtop(peroxygenase / "final" / "system.prmtop").residues}
    assert not names & {"NAG", "PO4", "GOL", "MOH"}


def test_the_iron_is_bonded_and_the_magnesium_is_not(peroxygenase):
    from mdprep.analysis import charges
    from mdprep.analysis.prmtop import read_prmtop

    topology = read_prmtop(peroxygenase / "final" / "system.prmtop")
    iron, magnesium = sorted(charges.metal_sites(topology), key=lambda site: -len(site["bonds"]))

    assert len(iron["bonds"]) == 6
    # Fe(III) + porphyrin(4-) + thiolate(1-) + water.
    assert iron["total"] == pytest.approx(-2.0, abs=1e-3)
    # A free ion, with no bond to anything.
    assert magnesium["bonds"] == [] and magnesium["total"] == pytest.approx(2.0, abs=1e-3)


def test_the_peroxygenase_validates(peroxygenase):
    report = json.loads((peroxygenase / "reports" / "validation_report.json").read_text())

    assert report["errors"] == []
    assert report["parmed"]["status"] == "ok"
    assert report["openmm"]["status"] == "ok"


def test_fad_is_built_from_its_proposed_smiles(flavin_halogenase):
    from mdprep.analysis import charges
    from mdprep.analysis.prmtop import read_prmtop

    topology = read_prmtop(flavin_halogenase / "final" / "system.prmtop")
    by_name = {row.name: row for row in charges.residue_charges(topology)}

    assert by_name["FAD"].charge == pytest.approx(-2.0, abs=1e-3)
    assert by_name["CTE"].charge == pytest.approx(0.0, abs=1e-3)


def test_the_flavin_halogenase_validates(flavin_halogenase):
    report = json.loads((flavin_halogenase / "reports" / "validation_report.json").read_text())

    assert report["errors"] == []
    assert report["openmm"]["status"] == "ok"
    assert sorted(check["ligand_id"] for check in report["ligand_presence_checks"]) == [
        "a_cte_650", "a_fad_600",
    ]
