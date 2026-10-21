"""The web builder on an unspecific peroxygenase (PDB 5OXU).

A glycoprotein crystallised from phosphate, glycerol and methanol, with a
heme-thiolate iron and a magnesium on the heme's propionate. A student is
asked once about the five N-glycans and once about the fourteen additives,
both proposed for removal, and never about a glycerol's SMILES; the
magnesium is proposed nonbonded; the heme's SMILES and the site's charge are
proposed as for a P450.

Slow, like the other reference cases: a real deposition through Pyodide.
"""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
STRUCTURE = REPO / "examples" / "tutorials" / "upo_5oxu" / "5OXU.pdb"

# Only what cannot be proposed.
ANSWERS = {
    "structure.remove_unknown_heterogens": "false",
    "metal.a_fe_401_site.charge": "3",
    "metal.a_fe_401_site.confirm_donors": "true",
    "metal.a_fe_401_site.donor_state.a_cys_36": "CYM",
    "refinement.total_qm_multiplicity": "2",
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
def steps(report) -> dict:
    return {step["id"]: step for step in report["asked"] if step.get("id")}


@pytest.fixture(scope="module")
def manifest(report) -> dict:
    assert report["ok"], report.get("failure")
    return yaml.safe_load(report["manifest"])


def test_the_page_finishes_without_errors_or_unanswered_questions(report):
    assert report["errors"] == []
    assert report["unanswered"] == []


def test_the_glycans_and_the_additives_are_one_question_each(steps):
    assert steps["structure.remove_glycans"]["selected"] == "true"
    assert steps["structure.remove_additives"]["selected"] == "true"
    # None of the glycerols, methanols or the phosphate is asked about.
    assert not [key for key in steps if key.startswith("ligand.") and "_hem_" not in key]


def test_the_magnesium_is_proposed_nonbonded(steps):
    assert steps["metal.a_mg_402_site.model"]["selected"] == "nonbonded"
    assert "metal.a_mg_402_site.confirm_donors" not in steps


def test_the_heme_site_is_proposed_as_for_a_p450(steps):
    assert steps["ligand.a_hem_401.net_charge"]["prefilled"] == "-4"
    assert steps["metal.a_fe_401_site.model_charge"]["prefilled"] == "-2"


def test_the_manifest_removes_them_and_builds_the_rest(manifest, tmp_path):
    removed = manifest["structure"]["remove_residues"]
    assert len(removed) == 19
    assert {item["selector"]["resname"] for item in removed} == {"NAG", "PO4", "GOL", "MOH"}
    assert [site["model"] for site in manifest["metals"]] == ["bonded_mcpb", "nonbonded"]

    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(STRUCTURE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")
    ManifestConfig.model_validate(candidate)
