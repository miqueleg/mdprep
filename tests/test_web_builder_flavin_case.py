"""The web builder on a flavin-dependent halogenase, and on a selenomethionine.

PrnA (PDB 2AR8) holds FAD and its chlorinated product. A student is offered
FAD's SMILES and charge rather than asked to type them, and types only the
product's. The fluorinase fragment (1RQR) has a selenomethionine in its
chain: one question, proposed yes, and a manifest that replaces it.

Slow, like the other reference cases: real depositions through Pyodide.
"""

import copy
import json
import subprocess
import sys

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from mdprep.ligands.cofactors import KNOWN_ORGANIC_COFACTORS
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
PRNA = REPO / "examples" / "tutorials" / "prna_2ar8" / "2AR8.pdb"
SELENOMETHIONINE = REPO / "tests" / "data" / "heterogens" / "selenomethionine_1rqr.pdb"
PRODUCT_SMILES = "[NH3+]C(Cc1c[nH]c2c(Cl)cccc12)C(=O)[O-]"


def _drive(tmp_path_factory, structure, answers: dict) -> dict:
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
            ["node", str(DRIVER), url, str(structure), chrome, str(path)],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def prna(tmp_path_factory):
    return _drive(
        tmp_path_factory,
        PRNA,
        {
            "structure.remove_unknown_heterogens": "false",
            "ligand.a_cte_650.net_charge": "0",
            "ligand.a_cte_650.smiles": PRODUCT_SMILES,
        },
    )


@pytest.fixture(scope="module")
def selenomethionine(tmp_path_factory):
    return _drive(
        tmp_path_factory,
        SELENOMETHIONINE,
        {"structure.repair.add_missing_heavy_atoms": "false"},
    )


def _steps(report) -> dict:
    return {step["id"]: step for step in report["asked"] if step.get("id")}


def test_the_flavin_case_finishes_without_errors(prna):
    assert prna["errors"] == []
    assert prna["unanswered"] == []


def test_fad_is_offered_rather_than_typed(prna):
    steps = _steps(prna)

    assert steps["ligand.a_fad_600.smiles"]["prefilled"] == KNOWN_ORGANIC_COFACTORS["FAD"].smiles
    assert "diphosphate fully ionised" in steps["ligand.a_fad_600.smiles"]["note"]
    assert steps["ligand.a_fad_600.net_charge"]["prefilled"] == "-2"
    # Nothing proposes a product's chemistry.
    assert steps["ligand.a_cte_650.smiles"]["prefilled"] in ("", None)


def test_the_flavin_manifest_is_one_mdprep_accepts(prna, tmp_path):
    assert prna["ok"], prna.get("failure")
    manifest = yaml.safe_load(prna["manifest"])
    fad = next(ligand for ligand in manifest["ligands"] if ligand["id"] == "a_fad_600")
    assert (fad["smiles"], fad["net_charge"]) == (KNOWN_ORGANIC_COFACTORS["FAD"].smiles, -2)

    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(PRNA)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")
    ManifestConfig.model_validate(candidate)


def test_a_selenomethionine_is_one_question_proposed_yes(selenomethionine):
    assert selenomethionine["errors"] == []
    assert selenomethionine["unanswered"] == []
    steps = _steps(selenomethionine)

    assert steps["structure.replace_modified_residues"]["selected"] == "true"
    assert not [key for key in steps if key.startswith("ligand.")]


def test_the_manifest_replaces_it(selenomethionine):
    manifest = yaml.safe_load(selenomethionine["manifest"])

    assert manifest["structure"]["repair"]["replace_modified_residues"] == [
        {"selector": {"chain": "A", "resname": "MSE", "resid": 14, "icode": None}, "parent": "MET"}
    ]
