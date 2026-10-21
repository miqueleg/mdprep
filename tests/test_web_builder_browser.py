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
import tempfile
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
# A metal site is the case the toolchain is hardest for, because it needs a
# build of xtb that conda-forge does not carry.
METAL_STRUCTURE = REPO / "tests" / "data" / "protein_histidine_zinc.pdb"
# A spin multiplicity cannot be 0, which is the driver's fallback. The QM
# region's own multiplicity is asked separately from the ion's: the region is
# one quantum system, and its spin is not a function of its parts.
METAL_ANSWERS = {
    "refinement.total_qm_multiplicity": "1",
}

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

    return drive(chrome, STRUCTURE)


def drive(chrome, structure, answers=None):
    command = [str(DRIVER), "", str(structure), chrome]
    with tempfile.TemporaryDirectory() as directory:
        if answers is not None:
            answers_path = Path(directory) / "answers.json"
            answers_path.write_text(json.dumps(answers), encoding="utf-8")
            command.append(str(answers_path))
        with serve(WEB) as url:
            command[1] = url
            result = subprocess.run(
                ["node", *command],
                capture_output=True,
                text=True,
                cwd=REPO,
                timeout=600,
            )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    payload = json.loads(result.stdout)
    # Chrome occasionally will not start on a CI runner -- "Timed out waiting
    # for the WS endpoint" -- after the driver has already retried. That says
    # nothing about the page, and failing on it blocks the Pages deployment
    # for a reason no commit can fix. Anything the page did once it loaded is
    # still a failure.
    if payload.get("launchFailed"):
        pytest.skip(payload.get("failure", "Chrome would not start"))
    return payload


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

    # With the chain: in an oligomer, AKG1 names a residue in every copy.
    assert captions["Formal net charge of B:SUB501"] == "Highlighted: B:SUB501"
    assert captions["Formal net charge of C:COF601"] == "Highlighted: C:COF601"
    assert "SUB501" in captions[
        "Remove heterogens that are not configured as ligands or metal sites?"
    ]
    assert captions["Protein force field"] == "Showing the whole structure."


def test_the_generated_manifest_validates(report):
    manifest = yaml.safe_load(report["manifest"])

    ManifestConfig.model_validate(manifest)
    assert [entry["id"] for entry in manifest["ligands"]] == ["b_sub_501", "c_cof_601"]
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
    assert "git+https://github.com/miqueleg/mdprep" in report["colabSetup"]


def test_the_setup_cell_actually_installs_the_toolchain(report):
    """pip cannot provide tleap, but conda-forge can, so the cell installs it."""
    setup = report["colabSetup"]

    assert "micromamba" in setup
    assert "-c conda-forge" in setup
    assert "ambertools" in setup
    assert "pip install -q" in setup


def test_the_setup_cell_only_installs_what_this_manifest_uses(report):
    """Both ligands default to AM1-BCC and protonation is manual, so no QM."""
    setup = report["colabSetup"]

    assert "pyscf" not in setup
    assert "xtb" not in setup
    assert "propka" not in setup


def test_the_page_warns_that_the_install_is_slow(report):
    note = report["toolchainNote"]

    assert "conda-forge" in note
    assert "once" in note


def test_the_run_cell_invokes_mdprep_inside_that_environment(report):
    """mdprep is not on the notebook's Python, so a bare call would not find it."""
    cell = report["colabUpload"]

    assert "micromamba run -n mdprep-runtime mdprep prepare system.yaml" in cell


def test_the_bundle_writer_runs_in_the_browser(report):
    assert report["bundleNames"] > 0


# --------------------------------------------------------------------------
# A metal site, end to end
#
# This is the case the generated setup cell is hardest for: a bonded MCPB site
# defaults to a g-xTB Hessian, and conda-forge carries only the GFN-only build
# of xtb. Driving a metal structure through the real page is the only check
# that a student who picks the defaults gets a Colab cell that can finish.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def metal_report():
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")
    subprocess.run(
        [sys.executable, str(WEB / "build.py")],
        check=True,
        capture_output=True,
        cwd=REPO,
    )
    return drive(chrome, METAL_STRUCTURE, answers=METAL_ANSWERS)


def test_a_metal_structure_reaches_a_manifest(metal_report):
    assert metal_report["ok"], metal_report.get("failure")
    assert metal_report["errors"] == []


def test_the_metal_manifest_validates(metal_report):
    manifest = yaml.safe_load(metal_report["manifest"])

    ManifestConfig.model_validate(manifest)
    assert manifest["metals"], "the metal site was dropped from the manifest"


def test_the_setup_cell_installs_the_gxtb_build_conda_does_not_have(metal_report):
    from mdprep.workflows.runtime import GXTB_SHA256

    setup = metal_report["colabSetup"]

    assert "grimme-lab/g-xtb" in setup
    # Checked before it is unpacked: this binary produces the force constants.
    assert GXTB_SHA256 in setup
    assert setup.index(GXTB_SHA256) < setup.index("tar -xJf")


def test_the_run_cell_points_mdprep_at_that_build(metal_report):
    """`micromamba run` starts a fresh environment, so the export in the
    setup cell does not survive into the run cell."""

    assert "MDPREP_GXTB_EXECUTABLE=" in metal_report["colabUpload"]


def test_the_metal_setup_cell_provides_exactly_one_xtb(metal_report):
    """The ESP and the histidine tautomers need GFN2, and the g-xTB build
    does GFN2 too, so conda's xtb would be a second copy of the same thing."""

    from mdprep.workflows.runtime import gxtb_bin_directory

    setup = metal_report["colabSetup"]

    assert f"export PATH={gxtb_bin_directory()}:$PATH" in setup
    create = next(line for line in setup.splitlines() if "micromamba create" in line)
    assert " xtb" not in create


def test_the_note_explains_the_extra_download(metal_report):
    note = metal_report["toolchainNote"]

    assert "g-xTB" in note
    assert "checksum" in note


def test_every_highlighted_target_actually_highlights_something(metal_report):
    """The viewer drew the uploaded file while the questions asked about the
    names mdprep chose, so for a renamed ligand or metal the selectors
    matched nothing -- including on "are these the atoms coordinating this
    metal?", the one question that asks the student to confirm what they can
    see. The caption said "Highlighted: ..." regardless."""

    highlights = metal_report["highlights"]

    assert highlights, "no question highlighted a target at all"
    empty = [h for h in highlights if h["atoms"] == 0]
    assert not empty, f"these highlighted nothing: {empty[:5]}"
