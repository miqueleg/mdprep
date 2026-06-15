# AGENTS.md

You are developing `mdprep`, a Python package for reproducible Amber MD system
preparation.

## Development Rules

- Use Python 3.11+ and type hints.
- Use pydantic models for YAML manifest validation.
- The legacy molecule-conversion toolkit identified by the dedicated ban test
  is intentionally prohibited.
- User overrides always win over automated predictions.
- Do not silently guess catalytic chemistry, protonation, ligand charge, or
  metal-center behavior.
- Structure normalization must never delete unknown heterogens silently; require
  explicit manifest configuration.
- Generated manifests must validate with `mdprep config-check`.
- Every residue rename must be recorded in reports.
- Automated protonation must not be silently skipped; fail clearly when a
  requested backend is unavailable or cannot produce an unambiguous assignment.
- Manual overrides always override PropKa and xTB decisions.
- PropKa/xTB tests must skip cleanly or use fakes if executables are
  unavailable.
- Temporary xTB tautomer hydrogens must never be written to the final prepared
  PDB.
- Unsupported chemistry must fail with a clear error.
- External commands must go through `mdprep.external.runner`.
- External command records must include command, working directory, return code,
  stdout, stderr, and runtime.
- Unit tests must not require AmberTools, xTB, PySCF, or PropKa.
- External tests must skip cleanly if required executables are unavailable.
- All user-facing features need tests and docs.
- All example YAML files must validate in tests.

## Current v0.1 Limits

Do not implement ligand parameterization, QM charge derivation, or final Amber
topology generation until the corresponding task explicitly requests it.

Explicitly unsupported for v0.1:

- noncanonical amino acids inside peptide chains
- covalent ligands
- bonded metal centers
- MCPB.py-like metal models
- ORCA backend
- Multiwfn dependency

## Required Checks

Run these before handing off:

```bash
pytest -q
python -m mdprep.cli --help
python -m mdprep.cli config-check examples/*.yaml
python -m mdprep.cli selftest --quick
```
