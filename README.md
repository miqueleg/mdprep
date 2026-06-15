# mdprep

`mdprep` is a Python package for reproducible Amber molecular-dynamics
preparation of standard proteins plus independent ligands and cofactors.
The YAML manifest is the source of truth: every preparation decision should be
recorded, validated, and reproduced.

## v0.1 Scope

This initial version provides the package scaffold, manifest validation, CLI
entry points, examples, self-test checks, PDB inspection, safe structure
normalization, manual protonation handling, PropKa-based automatic residue
state assignment, xTB-based neutral histidine tautomer ranking, and ligand
extraction plus starter AmberTools ligand parameter generation.

Planned supported chemistry includes standard amino-acid proteins,
crystallographic waters, manual and detected disulfides, ASP/GLU/LYS/ARG/HIS
protonation assignment, manual protonation overrides, HID/HIE histidine
selection through xTB/GFN2, independent ligands, GAFF/GAFF2 ligand setup,
AM1-BCC charges, PySCF-based embedded ligand charge derivation, and final
`tleap` generation of `prmtop` and `inpcrd` files.

## Unsupported Chemistry

The v0.1 design intentionally rejects noncanonical amino acids inside peptide
chains, covalent ligands, bonded metal centers, MCPB.py-style metal models,
ORCA backends, and required Multiwfn workflows. Chemically sensitive states
must not be guessed silently.

The legacy molecule-conversion toolkit identified by the dedicated ban test is
intentionally prohibited. Use RDKit, AmberTools, ParmEd, MDTraj, BioPython, and
direct file parsers instead.

## Installation

Create the development environment with conda or mamba:

```bash
mamba env create -f environment.yml
conda activate mdprep
```

The conda environment includes the optional PropKa and xTB executables used by
`protonation.method: propka` and `protonation.method: propka_xtb_his`, plus
AmberTools for `antechamber`/`parmchk2` ligand parameter generation.

For a lighter local install:

```bash
python -m pip install -e ".[test]"
```

## Quick Start

Validate the bundled examples:

```bash
mdprep config-check examples/*.yaml
```

Run package-level checks:

```bash
mdprep selftest --quick
```

Inspect a PDB structure:

```bash
mdprep inspect input.pdb
```

`mdprep inspect` is functional for PDB input. It reports atom and residue
counts, chains, waters, likely ligands or cofactors, histidines, titratable
residues, possible disulfides, alternate-location handling, and multi-model
warnings. Machine-readable output is available with:

```bash
mdprep inspect input.pdb --json
```

Create a starter manifest:

```bash
mdprep init input.pdb -o system.yaml
mdprep config-check system.yaml
```

Run the currently supported safe preparation stage:

```bash
mdprep prepare system.yaml --stop-after structure
```

This stage resolves alternate locations, keeps or removes crystal waters
according to the manifest, validates configured ligand selectors, refuses
unknown heterogens unless the manifest explicitly allows removing them, writes
a normalized PDB, and produces structure reports.

This stage does not assign protonation states, parameterize ligands, derive
QM-based charges, run AmberTools, or build Amber files.

Run the protonation stage:

```bash
mdprep prepare system.yaml --stop-after protonation
```

Supported protonation modes are:

- `manual_only`: applies only user overrides and disulfide `CYX` assignment.
- `propka`: runs PropKa and assigns pH-dependent ASP/GLU/CYS/LYS/HIS states,
  but requires every neutral HIS to already be HID/HIE by input state or manual
  override.
- `propka_xtb_his`: runs PropKa, assigns HIP for protonated HIS, and ranks
  neutral HID/HIE tautomers with xTB/GFN2 by default.

Manual overrides always win over PropKa and xTB. Input Amber-specific residue
states such as `ASH`, `GLH`, `HID`, `HIE`, `HIP`, `LYN`, `CYM`, and `CYX` are
preserved unless explicitly overridden. The protonation stage assigns
disulfide-linked cysteines to `CYX` from configured or detected pairs,
optionally removes input hydrogens according to
`structure.remove_input_hydrogens`, writes
`intermediate/01_protonation_assigned.pdb`, and produces JSON, CSV, and
Markdown protonation reports.

No hydrogens are added to the final prepared PDB. Temporary xTB tautomer
hydrogens are written only to local HID/HIE comparison XYZ files and are never
propagated to `01_protonation_assigned.pdb`. g-xTB can be used in single-point
or optimization mode through the `histidine.xtb` manifest block.

PropKa and xTB remain optional external tools. Unit tests do not require them;
requested automated workflows fail clearly if an executable is unavailable.
The external integration test is marked `external` and skips cleanly unless
`propka3` and `xtb` are installed.

Run the ligand stage:

```bash
mdprep prepare system.yaml --stop-after ligands
```

Task 6 supports two ligand charge/parameter input modes:

- `am1bcc`: extracts each configured ligand residue, runs AmberTools
  `antechamber`, validates the generated mol2, then runs `parmchk2`.
- `user_mol2`: validates a user-provided mol2 against the extracted ligand.
  If `user_frcmod` is provided it is copied; otherwise `parmchk2` is required.

`gas_resp_pyscf` and `qmmesp_pyscf` remain planned and fail clearly at the
ligand stage. Multiple configured ligands are processed independently; mdprep
does not reuse charges between ligand instances in this stage. The user must
provide the correct `net_charge` for every ligand. Atom count, atom names,
element order, coordinates, residue identity, and charge sum are validated, and
small mol2 charge residuals are corrected only within a strict tolerance.

The ligand stage writes extracted ligand PDBs, identity files, final mol2 files,
frcmod files, charge CSVs, validation JSON files, and ligand reports. It does
not build the final system topology yet.

Ligand troubleshooting:

- If AmberTools is unavailable, install the conda environment from
  `environment.yml` or add `antechamber` and `parmchk2` to `PATH`.
- If `antechamber` fails, inspect the ligand-specific stdout/stderr files in
  `prepared/ligands/<ligand_id>/parameters/`.
- If charge validation fails, check the manifest `net_charge` and mol2 charges.
- If atom-name validation fails, verify that the mol2 atom order matches the
  extracted ligand PDB.
- If coordinate validation fails, provide a mol2 generated from the same
  coordinates or explicitly allow coordinate changes in the manifest.
- Missing ligand hydrogens may cause parameterization failure; provide a
  chemically complete ligand input when needed.

The downstream full-system build commands remain placeholders:

```bash
mdprep prepare system.yaml
mdprep validate prepared/final/system.prmtop prepared/final/system.inpcrd
```

## Planned Workflow

The intended preparation flow is:

1. Normalize and inspect the input structure.
2. Apply manual overrides before automated decisions.
3. Assign protein protonation states with PropKa-like logic and xTB/GFN2
   neutral histidine tautomer selection where requested.
4. Parameterize independent ligands while preserving atom names, atom order,
   coordinates, residue identity, and total charge.
5. For embedded ligand charges, run provisional Amber setup, PySCF QM
   evaluation, RESP/ESP-like fitting, and final Amber rebuild.
6. Generate reproducible manifests, versions, intermediate structures, reports,
   and final Amber topology/coordinate files.
