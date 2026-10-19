# Contributing

## Setup

```bash
mamba env create -f environment.yml
conda activate mdprep
pip install -e .
mdprep selftest --quick
```

For packaging and lint tooling, use `environment-dev.yml` or install the
`dev` extra in a suitable environment.

## Tests

Default tests must not require AmberTools, xTB, PropKa, PySCF, ParmEd, or
OpenMM:

```bash
pytest -q
python -m mdprep.cli config-check examples/*.yaml
python -m mdprep.cli selftest --quick
```

External tests must skip cleanly when optional tools are unavailable:

```bash
pytest -q -m "external"
pytest -q -m "external or ambertools or tleap or propka or xtb or pyscf or qmmesp"
```

## Chemistry Rules

- The legacy molecule-conversion toolkit identified by the dedicated ban test
  is intentionally prohibited.
- Do not add molecule-conversion fallbacks.
- User overrides always win.
- Do not silently guess catalytic chemistry, ligand charge, protonation, or
  metal-center behavior.
- Every chemical decision and every residue rename must be reported.
- Unsupported chemistry must fail with a clear error.
- External commands must go through `mdprep.external.runner`.

## QMMESP Rules

- MM point charges polarize only the selected target ligand QM density.
- The RESP/ESP fit is performed only on target ligand atoms.
- Environment point charges must not be fitted.
- Environment point charges must not be written to ligand mol2 files.
- The final Amber system must use final PySCF-fitted mol2 charges, not
  provisional charges.

## Adding Features

New user-facing features need docs, tests, examples when practical, and clear
failure behavior for unsupported chemistry. Keep changes local to the relevant
workflow stage and avoid broad rewrites unless there is a tested reason.

## Releasing

Publishing is automatic and tokenless. `.github/workflows/release.yml` fires
on a `v*` tag and uploads to PyPI through Trusted Publishing, so no API token
exists to leak or rotate. PyPI is configured to trust this repository, this
workflow file by name, and the `pypi` environment; renaming either breaks
publishing until the publisher at
<https://pypi.org/manage/project/mdprep/settings/publishing/> is updated.

To cut a release:

1. Bump the version in `pyproject.toml` and `src/mdprep/__init__.py`, and in
   the README's first line. `tests/test_packaging_metadata.py` fails if the
   first two disagree.
2. Retitle the changelog's `## Unreleased` section to `## <version> - <date>`.
   The workflow refuses a tag with no such heading.
3. Update the `mdprep[md,qm]==` pin in `examples/*.ipynb` and the assertion in
   `tests/test_colab_smoke_notebook.py`. The web builder installs from git
   master and needs no bump.
4. Execute the official notebook end to end (below) and fix whatever it
   finds: it is what students run, and nothing else exercises it whole.
5. Merge, then `git tag -a vX.Y.Z` and push the tag.

The workflow then refuses a tag that does not match the version in the tree,
builds, checks that the sdist carries the project and nothing that was lying
in the working tree, installs the wheel and runs `mdprep selftest --quick`
from it, publishes to PyPI, and attaches the same two files to the GitHub
release -- creating one from the changelog section if you have not written
one by hand.

Use semantic versioning against the *manifest*, not just the API: a change
that makes an existing manifest stop validating is breaking, whatever it did
to the Python surface.

### Executing the official notebook outside Colab

`examples/mdprep_colab.ipynb` runs anywhere with a Linux GPU. Its uploads and
installs are redirected by environment variables, so a Python with mdprep,
`pymol-open-source-whl`, `py3Dmol`, `pandas`, `matplotlib` and `nbclient`
can execute it against a micromamba runtime built once:

```bash
MDPREP_COLAB_PREFIX=/scratch/colab \
MDPREP_SKIP_KERNEL_INSTALL=1 \
MDPREP_SPEC=$PWD \
MDPREP_TEST_UPLOAD_MODEL=/path/to/predicted_model.cif \
MDPREP_TEST_UPLOAD_BUNDLE=/path/to/mdprep-bundle.zip \
python -c "import nbformat; from nbclient import NotebookClient; \
nb = nbformat.read('examples/mdprep_colab.ipynb', 4); \
NotebookClient(nb, timeout=5400).execute(); nbformat.write(nb, 'executed.ipynb')"
```

`MDPREP_COLAB_PREFIX` stands in for `/content` (micromamba goes in its `bin/`,
the environment in `mamba-root/`), and `MDPREP_TEST_EXAMPLE_DIR=examples`
makes the SyrB2 example read the manifest from the checkout instead of GitHub.

