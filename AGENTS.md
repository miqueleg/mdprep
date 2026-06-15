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
- Ligand atom names, atom order, coordinates, residue identity, and total
  charge must be preserved or failures must be explicit.
- Do not add molecule-conversion fallback paths.
- Final topology generation must go through `tleap`.
- All `tleap` scripts and logs must be preserved.
- Do not silently ignore `tleap` warnings or errors; report them and fail when
  configured to do so.
- All final `.prmtop`/`.inpcrd` outputs must be validated.
- Unsupported chemistry must fail with a clear error.
- External commands must go through `mdprep.external.runner`.
- External command records must include command, working directory, return code,
  stdout, stderr, and runtime.
- Unit tests must not require AmberTools, xTB, PySCF, or PropKa.
- External tests must skip cleanly if required executables are unavailable.
- External AmberTools tests must skip cleanly if `antechamber` or `parmchk2`
  is unavailable.
- External `tleap`, ParmEd, and OpenMM tests must skip cleanly when optional
  executables or libraries are unavailable.
- All user-facing features need tests and docs.
- All example YAML files must validate in tests.

## Current v0.1 Limits

Do not implement QM charge derivation beyond the currently requested task.

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
