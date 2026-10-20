"""The web builder on a dimer: both copies of SyrB2 (PDB 2FCT).

A student who keeps both chains is asked once whether the copies share their
parameters, answers chain A's questions, and gets a manifest in which chain
B's cofactor and iron site are built from chain A's -- with chain B's own
residue numbers, which are not chain A's.

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
STRUCTURE = REPO / "examples" / "tutorials" / "syrb2_2fct" / "2FCT.pdb"

# Chain A's answers only; sharing is the default and is left to it.
ANSWERS = {
    "structure.chains": "all",
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


def test_the_page_finishes_without_errors_or_unanswered_questions(report):
    assert report["errors"] == []
    assert report["unanswered"] == []


def test_the_copies_question_is_asked_and_chain_b_is_not(report):
    asked = [step["id"] for step in report["asked"] if step.get("id")]

    assert "structure.same_parametrization" in asked
    assert not any(step.startswith(("ligand.b_", "metal.b_")) for step in asked)
    steps = {step["id"]: step for step in report["asked"] if step.get("id")}
    assert steps["metal.a_fe2_900_site.model_charge"]["prefilled"] == "-1"


def test_chain_b_is_built_from_chain_a_with_its_own_numbers(manifest):
    akg = next(item for item in manifest["ligands"] if item["id"] == "b_akg_898")
    site = next(item for item in manifest["metals"] if item["id"] == "b_fe2_899_site")

    assert akg["parameters_from"] == "a_akg_897"
    assert site["parameters_from"] == "a_fe2_900_site"
    assert site["ions"][0]["selector"]["resid"] == 899
    assert [bond["coordinator"]["resid"] for bond in site["mcpb"]["bonds"]] == [
        898, 116, 898, 235, 1019, 902
    ]


def test_only_chain_a_is_refined(manifest):
    assert manifest["refinement"]["qm_components"]["metal_sites"] == ["a_fe2_900_site"]
    assert manifest["refinement"]["total_qm_multiplicity"] == 5


def test_the_manifest_is_one_mdprep_accepts(manifest, tmp_path):
    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(STRUCTURE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")

    ManifestConfig.model_validate(candidate)
