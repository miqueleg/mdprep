"""The web builder on a site with a water on the metal: SyrB2 (PDB 2FCT).

The iron holds two histidines, alpha-ketoglutarate through two oxygens, a
chloride and a water; a free chloride and a detergent sit elsewhere. A
student clicking through the page has to come out with the water as an MCPB
bond, the bound chloride as a coordinating ion, the free one kept as a
crystal ion, and the model charge and QM spin proposed from what was already
answered.

Slow, like the other reference case: a real deposition through Pyodide.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
STRUCTURE = REPO / "examples" / "tutorials" / "syrb2_2fct" / "2FCT.pdb"

# Only what the engine will not default or propose. The model charge and the
# QM region's spin are left to the page's proposals on purpose.
ANSWERS = {
    "structure.chains": "A",
    # The detergent is declined below, so it has to be allowed to go.
    "structure.remove_unknown_heterogens": "true",
    "structure.repair.add_missing_heavy_atoms": "true",
    "ligand.a_dsu_896.include": "false",
    "ligand.a_akg_897.net_charge": "-2",
    "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
    "metal.a_fe2_900_site.confirm_donors": "true",
    "metal.a_fe2_900_site.multiplicity": "5",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")

    subprocess.run([sys.executable, str(WEB / "build.py")], check=True, capture_output=True, cwd=REPO)
    answers = tmp_path_factory.mktemp("answers") / "answers.json"
    answers.write_text(json.dumps(ANSWERS), encoding="utf-8")
    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(STRUCTURE), chrome, str(answers)],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def manifest(report) -> dict:
    assert report["ok"], report.get("failure")
    return yaml.safe_load(report["manifest"])


@pytest.fixture(scope="module")
def site(manifest) -> dict:
    return next(item for item in manifest["metals"] if item["id"] == "a_fe2_900_site")


def test_the_page_finishes_without_errors_or_unanswered_questions(report):
    assert report["errors"] == []
    assert report["unanswered"] == []


def test_the_crystal_water_and_crystal_ion_questions_are_asked(report):
    asked = {step["id"] for step in report["asked"]}

    assert "structure.keep_crystal_waters" in asked
    assert "structure.keep_crystal_ions" in asked
    # A chloride is never offered as a ligand.
    assert not any(step and step.startswith("ligand.a_cl_") for step in asked)


def test_the_water_on_the_iron_becomes_an_mcpb_bond(site):
    coordinators = {
        (bond["coordinator"]["resname"], bond["coordinator"]["resid"], bond["coordinator"]["atom_name"])
        for bond in site["mcpb"]["bonds"]
    }

    assert ("HOH", 1072, "O") in coordinators
    assert len(coordinators) == 6


def test_the_bound_chloride_is_a_coordinating_ion_and_the_free_one_stays(site, manifest):
    assert [(ion["selector"]["resid"], ion["element"]) for ion in site["coordinating_ions"]] == [(901, "Cl")]
    assert manifest["structure"]["keep_crystal_ions"] is True
    assert manifest["structure"]["keep_crystal_waters"] is True


def test_the_model_charge_and_spin_were_proposed_and_accepted(report, site, manifest):
    """Neither was typed: the page filled them in from the answers given."""

    steps = {step["id"]: step for step in report["asked"] if step.get("id")}
    assert steps["metal.a_fe2_900_site.model_charge"]["prefilled"] == "-1"
    assert steps["refinement.total_qm_multiplicity"]["prefilled"] == "5"
    assert site["mcpb"]["small_model_charge"] == -1
    assert manifest["refinement"]["total_qm_multiplicity"] == 5


def test_the_detergent_is_left_out_and_the_water_model_is_tip3p(manifest):
    assert [entry["id"] for entry in manifest["ligands"]] == ["a_akg_897"]
    assert manifest["protein"]["water_model"] == "TIP3P"


def test_the_manifest_is_one_mdprep_accepts(manifest, tmp_path):
    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(STRUCTURE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")

    ManifestConfig.model_validate(candidate)
