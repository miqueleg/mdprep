import json
from pathlib import Path


def test_colab_smoke_notebook_is_valid_and_targets_pypi() -> None:
    path = Path("examples/colab_smoke_test.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "mdprep[md,qm]==0.2.0" in source
    assert "https://test.pypi.org/simple/" not in source
    assert "selftest\", \"--quick" in source
