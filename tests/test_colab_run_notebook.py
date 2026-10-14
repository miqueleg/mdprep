"""Tests for the Colab notebook that runs a prepared manifest.

The notebook deliberately does not build manifests: the web input builder and
`mdprep init --interactive` do that, and all three share one decision engine.
What this notebook must get right is the handoff -- taking a bundle, installing
exactly the external programs the manifest needs, and running them.
"""

import ast
import json
import re
from pathlib import Path

import pytest


NOTEBOOK = Path("examples/mdprep_colab_run.ipynb")
WEB_APP = Path("web/app.js")


def notebook() -> dict:
    data = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert data["nbformat"] == 4
    return data


def code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def all_source() -> str:
    return "\n".join("".join(c["source"]) for c in notebook()["cells"])


def strip_ipython(source: str) -> str:
    """Drop the %magic and !shell lines, which are not Python."""
    kept = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("%", "!")):
            kept.append("pass")
            continue
        kept.append(line)
    return "\n".join(kept)


def test_the_notebook_is_committed_without_outputs():
    """Stored outputs make diffs unreadable and leak local paths."""
    assert all(not cell.get("outputs") for cell in notebook()["cells"])


@pytest.mark.parametrize("index", range(len(code_cells())))
def test_every_code_cell_is_valid_python(index):
    ast.parse(strip_ipython(code_cells()[index]))


def test_the_toolchain_is_derived_from_the_manifest():
    """Installing everything unconditionally would waste a long conda solve."""
    source = all_source()

    assert "conda_packages_for" in source
    assert "micromamba_setup_commands" in source


def test_the_notebook_does_not_reimplement_the_decision_engine():
    """Its job is to run a manifest, not to build one."""
    source = all_source()

    for duplicated in ("def collect_component_chemistry", "def group_components"):
        assert duplicated not in source
    # It drives mdprep's own entry points instead.
    assert "mdprep plan" in source
    assert "mdprep init" in source
    assert "mdprep prepare" in source


def test_it_installs_mdprep_from_the_same_place_as_the_web_builder():
    """A student must not get one version from the page and another here."""
    app = WEB_APP.read_text(encoding="utf-8")
    spec = re.search(r"const MDPREP_INSTALL = '([^']+)'", app)
    assert spec, "web/app.js no longer declares MDPREP_INSTALL"

    source_url = re.search(r"git\+https://[^\"' ]+", spec.group(1))
    assert source_url
    assert source_url.group(0) in all_source()


def test_micromamba_is_used_rather_than_condacolab():
    """condacolab replaces the kernel's interpreter and forces a restart.

    Checked against the code only; the prose explains the choice.
    """
    executed = "\n".join(code_cells())

    assert "micromamba" in executed
    assert "condacolab" not in executed


def test_mdprep_runs_inside_the_environment_it_was_installed_in():
    source = all_source()

    assert "micromamba" in source and "mdprep-runtime" in source


def test_a_failed_preparation_stops_the_notebook():
    """Colab happily continues past a failed cell unless it raises."""
    prepare_cells = [c for c in code_cells() if "mdprep\", \"prepare" in c or "'prepare'" in c]
    assert prepare_cells

    assert any("raise RuntimeError" in cell for cell in prepare_cells)


def test_the_results_are_validated_rather_than_assumed():
    assert "mdprep\", \"validate" in all_source() or "mdprep validate" in all_source()


def test_md_is_only_attempted_when_the_manifest_enables_it():
    source = all_source()

    assert "molecular_dynamics" in source
    assert "enabled" in source
