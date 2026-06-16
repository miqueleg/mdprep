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
