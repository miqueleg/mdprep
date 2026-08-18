from pathlib import Path
import re

import yaml

from mdprep.config.models import ManifestConfig


def test_readme_basic_manifest_is_valid_and_matches_example():
    readme = Path("README.md").read_text(encoding="utf-8")
    match = re.search(r"```yaml\n(.*?)\n```", readme, flags=re.DOTALL)
    assert match is not None
    readme_data = yaml.safe_load(match.group(1))
    example_data = yaml.safe_load(
        Path("examples/00_basic_protein.yaml").read_text(encoding="utf-8")
    )

    ManifestConfig.model_validate(readme_data)
    assert readme_data == example_data
