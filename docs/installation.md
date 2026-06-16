# Installation

Recommended user install:

```bash
mamba env create -f environment.yml
conda activate mdprep
pip install -e .
mdprep selftest --quick
```

The environment includes the optional external tooling used by supported
workflows: AmberTools, PropKa, xTB, OpenMM, ParmEd, and PySCF.

For development and release checks:

```bash
mamba env create -f environment-dev.yml
conda activate mdprep-dev
```

Pip-only installs can run the pure Python tests, but real preparation workflows
that call external chemistry tools require those executables or libraries.

## Quick Verification

```bash
mdprep --version
mdprep config-check examples/*.yaml
mdprep selftest --quick
pytest -q
```
