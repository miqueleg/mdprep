"""The web builder's 3D view when the browser's graphics give way.

At a workshop one laptop drew a large protein's ligands but not its cartoon,
and drew it whole after a few retries. The cartoon is the one large thing the
viewer draws, so it is what a browser short of graphics memory drops. The
viewer now gives back the last viewer's context when a structure is loaded
again, rebuilds itself when the browser resets its context, draws the protein
as lines when the card runs out of memory, and after three rebuilds stops and
says what to do. Each is forced here in a real browser.

Slow, like the other browser cases: a real structure through Pyodide.
"""

import json
import shutil
import subprocess
import sys

import pytest

from tests.test_web_builder_browser import REPO, WEB, find_chrome, has_puppeteer, serve

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "viewer_recovery.mjs"
STRUCTURE = REPO / "examples" / "tutorials" / "spoiiaa_1h4x" / "1H4X.pdb"


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")
    subprocess.run([sys.executable, str(WEB / "build.py")], check=True, capture_output=True, cwd=REPO)
    # A second name, so the page sees a new file and loads it again.
    again = tmp_path_factory.mktemp("again") / "1H4X_again.pdb"
    shutil.copyfile(STRUCTURE, again)
    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(STRUCTURE), str(again), chrome],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    data = json.loads(result.stdout)
    if data.get("launchFailed"):
        pytest.skip(data["failure"])
    assert data["ok"], data.get("failure")
    return data


def test_the_structure_is_drawn_as_a_cartoon(report):
    loaded = report["loaded"]

    assert (loaded["canvases"], loaded["lost"], loaded["light"]) == (1, False, False)
    assert "cartoon" in loaded["proteinStyle"]
    assert loaded["geometries"] > 0


def test_loading_again_deletes_the_last_viewers_buffers(report):
    """3Dmol left them to garbage collection: 80 MB for an octamer."""

    before = report["lastViewerBefore"]

    assert before["drawn"] >= 1
    assert report["lastViewerGeometries"] == before["geometries"] - before["drawn"]
    assert (report["reloaded"]["canvases"], report["reloaded"]["lost"]) == (1, False)


def test_a_reset_context_is_rebuilt_on_a_new_one(report):
    """3Dmol's own handler listens on the page's canvas, which is never lost."""

    reset = report["afterReset"]

    assert report["newContext"] is True
    assert (reset["recoveries"], reset["lost"], reset["light"]) == (1, False, False)
    assert "cartoon" in reset["proteinStyle"]


def test_out_of_memory_draws_the_protein_as_lines(report):
    memory = report["afterMemory"]

    assert (memory["recoveries"], memory["lost"], memory["light"]) == (2, False, True)
    assert memory["proteinStyle"] == "line"
    assert "drawn as lines" in memory["caption"]


def test_a_browser_that_keeps_losing_it_is_told_and_the_questions_go_on(report):
    gave_up = report["gaveUp"]

    assert (gave_up["hasViewer"], gave_up["recoveries"]) == (False, 4)
    assert "could not keep the 3D view" in gave_up["caption"]
    assert report["questionsGoOn"] >= 2


def test_redraw_brings_the_viewer_back(report):
    redrawn = report["redrawn"]

    assert (redrawn["hasViewer"], redrawn["lost"], redrawn["recoveries"]) == (True, False, 0)
    assert report["errors"] == []
