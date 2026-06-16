from pathlib import Path
import tomllib

import mdprep


def test_pyproject_release_metadata_is_consistent():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert project["name"] == "mdprep"
    assert project["version"] == mdprep.__version__ == "0.1.0"
    assert project["requires-python"].startswith(">=3.11")
    assert project["scripts"]["mdprep"] == "mdprep.cli:app"
    assert "README.md" == project["readme"]
