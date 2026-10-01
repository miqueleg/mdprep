"""End-to-end check of the web input builder in a real browser.

The page runs mdprep under Pyodide, so the things that can break it -- a wheel
micropip will not parse, a Pyodide package that stopped shipping, a decision
field the page reads under a name the engine no longer uses -- are invisible to
every other test in this suite. This one loads the actual page, uploads a
structure, answers the questions, and validates the manifest that falls out.

It is slow (Pyodide downloads its runtime on first load) and needs Node,
Chromium and puppeteer-core, so it is marked external and slow and skips
cleanly when any of those is missing.
"""

import contextlib
import json
import shutil
import socket
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from mdprep.config.models import ManifestConfig


pytestmark = [pytest.mark.external, pytest.mark.slow]

REPO = Path(__file__).resolve().parent.parent
WEB = REPO / "web"
DRIVER = REPO / "tests" / "browser" / "smoke.mjs"
STRUCTURE = REPO / "tests" / "data" / "protein_two_ligands.pdb"

CHROME_CANDIDATES = (
    "chromium-browser",
    "chromium",
    "google-chrome",
    "google-chrome-stable",
)


def find_chrome() -> str | None:
    for name in CHROME_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def has_puppeteer() -> bool:
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(
        ["node", "-e", "require.resolve('puppeteer-core')"],
        capture_output=True,
        cwd=REPO,
    )
    return probe.returncode == 0


@contextlib.contextmanager
def serve(directory: Path):
    """Serve `directory` on a free port for the duration of the test."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    handler = partial(QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        server.server_close()


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D102 - keep pytest output readable
        pass


@pytest.fixture(scope="module")
def report():
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")

    # The page installs the wheel at runtime, so it has to exist first.
    subprocess.run(
        [sys.executable, str(WEB / "build.py")],
        check=True,
        capture_output=True,
        cwd=REPO,
    )

    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(STRUCTURE), chrome],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=600,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def test_the_page_runs_without_errors(report):
    assert report["ok"], report.get("failure")
    assert report["errors"] == []


def test_pyodide_loads_the_real_mdprep_wheel(report):
    """If this reports a version, micropip accepted the wheel build.py made."""
    assert report["status"].startswith("Ready (mdprep ")


def test_the_browser_gets_the_same_plan_the_cli_would(report):
    from mdprep.config.decisions import plan_decisions

    expected = plan_decisions(STRUCTURE)

    assert report["plan"]["decisions"] == len(expected.decisions)
    assert report["plan"]["blocking"] == len(expected.blocking())
    assert report["plan"]["ligands"] == expected.findings["ligands"]


def test_the_structure_viewer_renders(report):
    assert report["canvas"] is not None
    assert report["canvas"]["width"] > 0
    assert report["canvas"]["height"] > 0


def test_each_question_frames_what_it_is_about(report):
    captions = {step["text"]: step["caption"] for step in report["steps"]}

    assert captions["Formal net charge of B:SUB501"] == "Highlighted: SUB501"
    assert captions["Formal net charge of C:COF601"] == "Highlighted: COF601"
    assert "SUB501" in captions[
        "Remove heterogens that are not configured as ligands or metal sites?"
    ]
    assert captions["Protein force field"] == "Showing the whole structure."


def test_the_generated_manifest_validates(report):
    manifest = yaml.safe_load(report["manifest"])

    ManifestConfig.model_validate(manifest)
    assert [entry["id"] for entry in manifest["ligands"]] == ["sub_501", "cof_601"]
    assert manifest["ligands"][0]["selector"]["resname"] == "SUB"


def test_the_manifest_records_that_values_were_entered_not_inferred(report):
    assert "not inferred" in report["manifest"]


def test_the_upload_path_tells_colab_to_unpack_the_bundle(report):
    cell = report["colabUpload"]

    assert "files.upload()" in cell
    assert "mdprep-bundle.zip" in cell
    assert "mdprep prepare system.yaml" in cell


def test_the_rcsb_path_refetches_by_id_instead(report):
    cell = report["colabRcsb"]

    assert "files.rcsb.org/download/1AKI.pdb" in cell
    assert "files.upload()" not in cell


def test_the_colab_cell_does_not_promise_a_pypi_release_that_lacks_the_fixes(report):
    """0.2.0 is the published wheel and predates the stage-2 NaN fix."""
    assert "git+https://github.com/miqueleg/mdprep" in report["colabUpload"]


def test_the_page_says_ambertools_is_needed(report):
    """`pip install mdprep` cannot provide tleap, so the page must not imply it."""
    note = report["toolchainNote"]

    assert "AmberTools" in note
    assert "tleap" in note


def test_the_bundle_writer_runs_in_the_browser(report):
    assert report["bundleNames"] > 0
