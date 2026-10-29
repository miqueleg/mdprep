"""The web builder on a substrate docked too close to an iron with no water.

The SyrB2 iron site with the water on the iron replaced by a docked ethanol
(see tests/data/metal_sites/README.md): its oxygen is 2.24 A from the iron,
so mdprep offers it as one of the iron's bonds. The page shows every residue
in range as a box, all ticked; a student who knows the substrate is only
close unticks it, gives it charges of its own, and gets a site fitted without
it, with MCPB.py's bonding cutoff below it.

Slow, like the other cases: the real page, through Pyodide.
"""

import json
import subprocess
import sys

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
STRUCTURE = REPO / "tests" / "data" / "metal_sites" / "syrb2_site_docked_substrate.pdb"
CHECKLIST = "metal.a_fe2_900_site.confirm_donors"
EVERY_BOX = "A:AKG897,A:HIS116,A:HIS235,A:SUB950,A:CL901"
WITHOUT_SUBSTRATE = "A:AKG897,A:HIS116,A:HIS235,A:CL901"

# Only what the engine will not default or propose: the model charge is left
# to the page's proposal on purpose.
ANSWERS = {
    "structure.remove_unknown_heterogens": "true",
    "structure.repair.add_missing_heavy_atoms": "true",
    "ligand.a_akg_897.net_charge": "-2",
    "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
    "ligand.a_sub_950.net_charge": "0",
    "ligand.a_sub_950.smiles": "CCO",
    "ligand.a_sub_950.charge_method": "am1bcc",
    CHECKLIST: WITHOUT_SUBSTRATE,
    "refinement.total_qm_multiplicity": "5",
}


def _drive(tmp_path_factory, answers: dict) -> dict:
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")
    subprocess.run([sys.executable, str(WEB / "build.py")], check=True, capture_output=True, cwd=REPO)
    path = tmp_path_factory.mktemp("answers") / "answers.json"
    path.write_text(json.dumps(answers), encoding="utf-8")
    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(STRUCTURE), chrome, str(path)],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
        )
    if result.returncode != 0 and not result.stdout.strip():
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return _drive(tmp_path_factory, ANSWERS)


@pytest.fixture(scope="module")
def site(report) -> dict:
    assert report["ok"], report.get("failure")
    manifest = yaml.safe_load(report["manifest"])
    ManifestConfig.model_validate(manifest)
    return next(item for item in manifest["metals"] if item["id"] == "a_fe2_900_site")


def test_the_page_finishes_without_errors_or_unanswered_questions(report):
    assert report["errors"] == []
    assert report["unanswered"] == []


def test_every_residue_in_range_is_shown_ticked(report):
    asked = next(item for item in report["asked"] if item["id"] == CHECKLIST)

    assert asked["selected"] == EVERY_BOX


def test_the_unticked_substrate_is_not_bonded(site):
    bonded = {bond["coordinator"]["resname"] for bond in site["mcpb"]["bonds"]}

    assert "SUB" not in bonded
    assert bonded == {"AKG", "HIS", "CL"}


def test_mcpb_cutoff_is_below_the_substrate(site):
    assert site["mcpb"]["cutoff_angstrom"] == 2.19


def test_the_model_charge_proposed_leaves_the_substrate_out(site):
    # Fe(II) +2, AKG -2, chloride -1, two neutral histidines.
    assert site["mcpb"]["small_model_charge"] == -1


def test_unticking_a_substrate_still_on_the_metal_fit_is_stopped_at_the_checklist(tmp_path_factory):
    """Its charges were to come from a fit that no longer includes it."""
    answers = dict(ANSWERS)
    del answers["ligand.a_sub_950.charge_method"]

    stopped = _drive(tmp_path_factory, answers)

    stuck = [item for item in stopped["unanswered"] if item.startswith(f"stuck on {CHECKLIST}")]
    assert stuck, stopped["unanswered"]
    assert "such as AM1-BCC" in stuck[0]
