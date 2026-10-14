"""The decision engine must stay importable inside a browser.

The web manifest builder runs the real mdprep package under Pyodide rather than
reimplementing the planning logic in JavaScript. That only works while the
planning path imports nothing heavier than pydantic and PyYAML: a stray
``import numpy`` at module scope in, say, ``mdprep.structure.inspect`` would
break the site at load time and nowhere else, so it is pinned here instead.

Everything the preparation workflows need is unaffected; this is specifically
about what `plan_decisions` and `build_manifest` drag in.
"""

import subprocess
import sys

import pytest


# Available as Pyodide wheels, and small enough to load in a browser.
BROWSER_SAFE_PACKAGES = {
    "annotated_types",
    "pydantic",
    "pydantic_core",
    "typing_extensions",
    "typing_inspection",
    "yaml",
}

# Python's site machinery imports these at interpreter startup if the
# environment provides them, which several do: a venv built by pip, a coverage
# hook, anything shipping a .pth file. They are not imported by the planning
# path and the browser never downloads them, so counting them would make this
# fail for reasons that have nothing to do with what it is checking.
INTERPRETER_HOOKS = {"sitecustomize", "usercustomize"}

PLANNING_IMPORTS = (
    "import mdprep.config.decisions\n"
    "import mdprep.workflows.wizard\n"
)


def third_party_imports(script: str) -> set[str]:
    """Top-level non-stdlib packages a script pulls in, in a clean interpreter."""

    probe = (
        "import sys\n"
        f"{script}"
        "stdlib = set(sys.stdlib_module_names)\n"
        "tops = {m.split('.')[0] for m in sys.modules if not m.startswith('_')}\n"
        # Compiled extensions inject pseudo-modules such as `cython_runtime`
        # that have no file and are not separately installable, so they say
        # nothing about what the browser has to download.
        "real = {t for t in tops "
        "if getattr(sys.modules.get(t), '__file__', None) is not None}\n"
        f"ignored = {sorted(INTERPRETER_HOOKS)!r}\n"
        "print(' '.join(sorted(t for t in real "
        "if t not in stdlib and t != 'mdprep' and t not in ignored)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(result.stdout.split())


def test_planning_path_imports_only_browser_safe_packages():
    imported = third_party_imports(PLANNING_IMPORTS)

    unexpected = imported - BROWSER_SAFE_PACKAGES
    assert not unexpected, (
        "The web manifest builder loads mdprep under Pyodide with only "
        f"{sorted(BROWSER_SAFE_PACKAGES)} installed. These new imports would "
        f"break it: {sorted(unexpected)}. Either keep them out of the planning "
        "path, import them lazily inside the function that needs them, or add "
        "them here after confirming a Pyodide wheel exists."
    )


def test_planning_path_does_not_import_numpy():
    """Called out separately: numpy is the likeliest accidental addition."""
    assert "numpy" not in third_party_imports(PLANNING_IMPORTS)


def test_planning_path_does_not_import_the_cli_stack():
    """typer and rich are terminal concerns and have no place in the browser."""
    imported = third_party_imports(PLANNING_IMPORTS)

    assert "typer" not in imported
    assert "rich" not in imported


@pytest.mark.parametrize("package", sorted(BROWSER_SAFE_PACKAGES))
def test_browser_safe_packages_are_actually_installed(package):
    """Guards the allowlist itself against naming something that is not used."""
    __import__(package)
