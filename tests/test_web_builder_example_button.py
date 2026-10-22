"""The web builder's one-click example, for a student with no structure of
their own: SyrB2 (PDB 2FCT), fetched from RCSB like any typed ID.

The browser's request to RCSB is answered from the vendored deposition, so
this needs Chrome but no network. Slow, like the other browser tests: the page
boots Pyodide.
"""

import json
import subprocess
import sys

import pytest

from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "example_button.mjs"
DEPOSITION = REPO / "examples" / "tutorials" / "syrb2_2fct" / "2FCT.pdb"


@pytest.fixture(scope="module")
def report():
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")
    subprocess.run([sys.executable, str(WEB / "build.py")], check=True, capture_output=True, cwd=REPO)
    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(DEPOSITION), str(chrome)],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=600,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    payload = json.loads(result.stdout)
    if payload.get("launchFailed"):
        pytest.skip(payload.get("failure", "Chrome would not start"))
    assert payload.get("ok"), payload.get("failure")
    return payload


def test_the_button_names_the_example(report):
    assert report["buttonText"] == "Load SyrB2 (PDB 2FCT)"


def test_it_fetches_2fct_from_rcsb_as_a_typed_id_would(report):
    assert report["rcsbRequests"] == ["https://files.rcsb.org/download/2FCT.pdb"]
    assert report["loaded"]["pdbId"] == "2FCT"
    assert report["loaded"]["structureName"] == "2FCT.pdb"
    assert report["loaded"]["idField"] == "2FCT"


def test_the_questions_follow(report):
    assert report["loaded"]["questionShown"]
    assert report["loaded"]["decisions"] > 0
    assert not report["errors"], report["errors"]
