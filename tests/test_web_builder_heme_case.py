"""The web builder on a cytochrome P450: CYP119 (PDB 1IO7), chain A.

A student loading a P450 sees the heme's iron as a metal site of its own, is
offered the porphyrin's SMILES and charge rather than asked to type them, is
asked the iron's oxidation state, the cysteine's protonation and the QM
region's spin, and
gets a manifest with a bonded iron site whose donors are the four porphyrin
nitrogens, the cysteine and the water.

Slow, like the other reference cases: a real deposition through Pyodide.
"""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from mdprep.ligands.cofactors import KNOWN_METAL_FREE_COFACTORS
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
STRUCTURE = REPO / "examples" / "tutorials" / "cyp119_1io7" / "1IO7.pdb"
HEME_SMILES = KNOWN_METAL_FREE_COFACTORS["HEM"].smiles

# Only what cannot be proposed; the SMILES, the net charge, the model charge
# and the spin of the QM region are accepted as the page offers them.
ANSWERS = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": "false",
    "structure.repair.add_missing_heavy_atoms": "true",
    "metal.a_fe_1401_site.charge": "3",
    "metal.a_fe_1401_site.confirm_donors": "true",
    "metal.a_fe_1401_site.donor_state.a_cys_317": "CYM",
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


def test_the_porphyrin_is_offered_rather_than_typed(steps):
    smiles = steps["ligand.a_hem_1401.smiles"]
    assert smiles["prefilled"] == HEME_SMILES
    assert "both propionates ionised" in smiles["note"]
    assert steps["ligand.a_hem_1401.net_charge"]["prefilled"] == "-4"


def test_the_iron_and_its_cysteine_are_asked_and_the_sum_follows(steps):
    assert steps["metal.a_fe_1401_site.charge"]["prefilled"] in ("", None)
    charge = steps["metal.a_fe_1401_site.model_charge"]
    assert charge["prefilled"] == "-2"
    assert "FE +3 HEM1401 -4 CYM317 -1 HOH1402 +0 = -2" in charge["note"]


def test_the_iron_is_bonded_to_the_porphyrin_the_cysteine_and_the_water(manifest):
    (site,) = manifest["metals"]
    assert site["ions"][0]["selector"]["resname"] == "FE"
    assert sorted(
        (bond["coordinator"]["resname"], bond["coordinator"]["atom_name"])
        for bond in site["mcpb"]["bonds"]
    ) == sorted(
        [("HEM", name) for name in ("NA", "NB", "NC", "ND")] + [("CYS", "SG"), ("HOH", "O")]
    )
    assert site["mcpb"]["small_model_charge"] == -2
    assert site["mcpb"]["pyscf"]["xtb"]["electronic_temperature_kelvin"] == 1000.0
    (ligand,) = manifest["ligands"]
    assert (ligand["smiles"], ligand["net_charge"], ligand["charge_method"]) == (
        HEME_SMILES, -4, "mcpb_resp_pyscf"
    )


def test_the_manifest_is_one_mdprep_accepts(manifest, tmp_path):
    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(STRUCTURE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")

    ManifestConfig.model_validate(candidate)
