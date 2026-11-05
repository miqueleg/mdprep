"""The web builder on a dimer whose second copy is named apart from the first.

SyrB2 (PDB 2FCT) with chain B's alpha-ketoglutarate renamed AKX and its iron
FE, as a dimer assembled by hand names its second copy's heterogens. The
copies still pair -- by atoms and place, not by name -- so a student who says
the copies are alike gets one bonded fit, reused in chain B, and chain B's
cofactor built as AKG with chain A's parameters. The page says so beside the
manifest, which records the rename.

Slow, like the other reference cases: a real deposition through Pyodide.
"""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from mdprep.structure.models import PdbStructure, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb
from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
DEPOSITION = REPO / "examples" / "tutorials" / "syrb2_2fct" / "2FCT.pdb"

ANSWERS = {
    "structure.chains": "all",
    "structure.remove_unknown_heterogens": "true",
    "structure.repair.add_missing_heavy_atoms": "true",
    "ligand.a_dsu_896.include": "false",
    "ligand.a_akg_897.net_charge": "-2",
    "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
    "metal.a_fe2_900_site.confirm_donors": "true",
    "refinement.total_qm_multiplicity": "5",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")

    deposition = read_pdb(DEPOSITION)
    names = {("B", 898): "AKX", ("B", 899): "FE"}
    atoms = [
        replace(atom, resname=names[(atom.chain_id, atom.resid)], original_line="")
        if (atom.chain_id, atom.resid) in names
        else atom
        for atom in deposition.atoms
    ]
    folder = tmp_path_factory.mktemp("named_apart")
    structure = folder / "2FCT_named_apart.pdb"
    write_pdb(
        PdbStructure(path=structure, atoms=atoms, residues=build_residues(atoms), model_count=1),
        structure,
    )

    subprocess.run([sys.executable, str(WEB / "build.py")], check=True, capture_output=True, cwd=REPO)
    answers = folder / "answers.json"
    answers.write_text(json.dumps(ANSWERS), encoding="utf-8")
    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(structure), chrome, str(answers)],
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


def test_the_copy_is_built_under_the_first_ones_name(manifest):
    ManifestConfig.model_validate(manifest)
    ligand = next(item for item in manifest["ligands"] if item["id"] == "b_akx_898")
    assert ligand["parameters_from"] == "a_akg_897"
    assert ligand["selector"]["resname"] == "AKG"
    (rename,) = manifest["structure"]["rename_residues"]
    assert rename["selector"] == {"chain": "B", "resname": "AKX", "resid": 898}
    assert rename["resname"] == "AKG"


def test_one_bonded_fit_serves_both_copies(manifest):
    sites = {site["id"]: site for site in manifest["metals"]}
    assert sites["a_fe2_900_site"]["model"] == "bonded_mcpb"
    assert sites["b_fe_899_site"]["parameters_from"] == "a_fe2_900_site"


def test_the_page_says_what_it_renamed(report):
    assert "B:AKX898 → AKG" in report["warnings"]
    assert "counterpart of A:AKG897 in another copy of the protein" in report["warnings"]


def test_the_page_says_which_waters_it_left_out(report, manifest):
    """2FCT's A:HOH1034 sits 2.14 A from a carbonyl oxygen: left out, and
    said so beside the manifest."""

    waters = [
        item
        for item in manifest["structure"]["remove_residues"]
        if item["reason"].startswith("crystal water overlapping")
    ]
    assert {"chain": "A", "resname": "HOH", "resid": 1034} in [
        {key: item["selector"][key] for key in ("chain", "resname", "resid")} for item in waters
    ]
    said = " ".join(report["warnings"].split())
    assert f"{len(waters)} crystal water" in said
    assert "left out for sitting on another atom" in said
