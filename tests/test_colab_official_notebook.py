"""The official Colab notebook: prepare, equilibrate, and look at every step.

It is executed end to end before a release (see CONTRIBUTING.md); these
check what can be checked without Colab: that it is valid, that it drives
mdprep's own entry points rather than re-implementing them, and that it
installs the same mdprep as the web builder.
"""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path("examples/mdprep_colab.ipynb")
POINTER = Path("examples/mdprep_colab_run.ipynb")
WEB_APP = Path("web/app.js")


def notebook(path=NOTEBOOK) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    return data


def code_cells() -> list[str]:
    return ["".join(cell["source"]) for cell in notebook()["cells"] if cell["cell_type"] == "code"]


def all_source() -> str:
    return "\n".join("".join(cell["source"]) for cell in notebook()["cells"])


def test_the_notebook_is_committed_without_outputs():
    """Stored outputs make diffs unreadable and leak local paths."""

    assert all(not cell.get("outputs") for cell in notebook()["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook()["cells"] if cell["cell_type"] == "code")


@pytest.mark.parametrize("index", range(len(code_cells())))
def test_every_code_cell_is_valid_python(index):
    ast.parse(code_cells()[index])


def test_it_installs_mdprep_from_the_same_place_as_the_web_builder():
    app = WEB_APP.read_text(encoding="utf-8")
    spec = re.search(r"const MDPREP_INSTALL = '([^']+)'", app)
    assert spec, "web/app.js no longer declares MDPREP_INSTALL"
    url = re.search(r"git\+https://[^\"' ]+", spec.group(1)).group(0)

    assert url in all_source()


def test_the_programs_are_installed_before_there_is_a_bundle():
    """Section A runs while the student is still answering the builder, so it
    installs every program mdprep runs rather than reading a manifest; B then
    checks the bundle's manifest against what A installed."""

    a = _cell("#@title A.")

    assert "full_toolchain_manifest()" in a
    assert "micromamba_setup_commands(toolchain" in a
    assert "MANIFEST" not in a
    assert "condacolab" not in "\n".join(code_cells())
    assert 'mdprep_in_runtime("doctor", MANIFEST.name)' in _cell("#@title B.")


def test_the_structure_comes_only_from_the_builders_bundle():
    """The input builder decides the structure. The notebook neither builds
    one -- no predicted model, no crystal waters -- nor offers a ready-made
    example or an earlier run to load in its place."""

    executed = "\n".join(code_cells())
    b = _cell("#@title B.")

    assert 'upload("bundle", (".zip",))' in b
    assert "#@param" not in b
    for gone in ("FINISHED_RUN", "search_closest_xray", "transplant_waters", "urlretrieve"):
        assert gone not in executed, gone


def test_the_cells_run_in_the_order_the_class_does():
    """Install while the student is in the builder; prepare and the MD back to
    back, so a run from top to bottom never waits on a student reading a
    report; then everything that reads what they wrote."""

    order = [source.splitlines()[0].removeprefix("#@title ").split(".")[0] for source in code_cells()]

    assert order == [
        "1", "Helper functions (run once; expand to read them)", "A", "B", "C", "D",
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "F", "F2", "G", "H", "I",
    ]


def test_md_runs_once_in_its_own_cell():
    """prepare runs the MD itself when the manifest enables it; the notebook
    stops it before the MD and runs it in section F, or it ran twice."""

    executed = "\n".join(code_cells())

    assert '"--stop-after", "tleap"' in executed
    assert executed.count('"run-md"') == 1


def test_a_failed_command_stops_the_notebook():
    """Colab carries on past a failed subprocess unless the cell raises."""

    helpers = next(cell for cell in code_cells() if "def run(" in cell)

    assert "raise RuntimeError" in helpers


def test_every_step_the_course_needs_is_looked_at_through_mdprep():
    """The analysis is mdprep.analysis, tested in test_analysis.py, not
    notebook code nobody runs before class."""

    executed = "\n".join(code_cells())
    for call in (
        "intermediate",  # E: the structure at each stage
        "protonation_report.csv",  # E: protonation states
        "charge_tables.summary",  # E: charges
        "charge_tables.atom_charges",
        "site_bonds",  # E: the metal site's force constants
        "eq.equilibration_table",  # F: how it equilibrated
        "eq.stage_spans",
        "eq.production_table",
        "md_dir() / f\"{STEP}.pdb\"",  # G: each step's structure
        "tr.load_trajectory",  # H: the MD
        "tr.rmsd",
        "tr.rmsf",
        "tr.pdb_models",
    ):
        assert call in executed, call
    assert "%FLAG" not in executed, "the notebook must not parse prmtops itself"


def test_uploads_and_downloads_work_outside_colab_for_testing():
    helpers = next(cell for cell in code_cells() if "def upload(" in cell)

    assert "MDPREP_TEST_UPLOAD_" in helpers
    assert "IN_COLAB" in helpers


def test_the_old_run_notebook_points_here():
    """Links to it exist in the wild; it must not 404 or run the MD twice."""

    cells = notebook(POINTER)["cells"]
    assert all(cell["cell_type"] == "markdown" for cell in cells)
    assert "mdprep_colab.ipynb" in "".join("".join(cell["source"]) for cell in cells)


def test_the_web_builder_links_to_this_notebook():
    app = WEB_APP.read_text(encoding="utf-8")
    url = re.search(r'const COLAB_NOTEBOOK_URL =\s*"([^"]+)"', app).group(1)

    assert url.endswith("/examples/mdprep_colab.ipynb")
    assert app.count("COLAB_NOTEBOOK_URL") >= 2, "declared but never used"


def _cell(title: str) -> str:
    return next(source for source in code_cells() if source.startswith(title))


def test_the_final_structure_is_drawn_from_the_topology():
    """tleap's PDB has no element column, and a viewer then guesses elements
    from atom names -- HG21, a hydrogen, as mercury -- and draws bonds from
    them to everything near. The topology has every element exactly."""

    e3 = _cell("#@title E3.")

    final = e3[e3.index('if STAGE == "final":'):e3.index("else:")]
    assert "tr.load_structure(" in final and "tr.pdb_models(" in final
    assert 'system.pdb' not in final


def test_a_lone_ion_is_drawn_and_the_added_salt_is_not():
    """A chloride on a halogenase's iron is one atom with no bond to draw as a
    stick, so it needs a sphere; the hundreds of counterions tleap adds do not."""

    helpers = _cell("#@title Helper functions")

    assert 'view.addStyle({"elem": IONS, "hetflag": True, "not": {"resn": ADDED_IONS}}' in helpers
    assert '"Cl-"' in helpers and '"Na+"' in helpers


def test_reinstalling_caps_the_cuda_runtime_at_the_driver():
    """Rerunning section A is what repairs a session whose CUDA platform would
    not start, so the existing-environment path has to apply the cap too."""

    a = _cell("#@title A.")

    assert "micromamba_install_commands(toolchain" in a
    assert "micromamba_setup_commands(toolchain" in a
