from pathlib import Path

from mdprep.config.loader import load_manifest


def test_all_examples_validate_with_loader():
    examples = sorted(Path("examples").glob("*.yaml"))

    assert len(examples) >= 7
    for example in examples:
        manifest = load_manifest(example)
        assert manifest.project.name
