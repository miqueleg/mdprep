# mdprep

`mdprep` is a Python package for reproducible Amber molecular-dynamics
preparation of standard proteins plus independent ligands and cofactors.
The YAML manifest is the source of truth: every preparation decision should be
recorded, validated, and reproduced.

## v0.1 Scope

This initial version provides the package scaffold, manifest validation, CLI
entry points, examples, and self-test checks. Real chemistry workflows will be
implemented in later tasks.

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

Run the manual protonation stage:

```bash
mdprep prepare system.yaml --stop-after protonation
```

For now this stage supports only `protonation.method: manual_only`. It applies
manual residue-state overrides, assigns disulfide-linked cysteines to `CYX`
from configured or detected pairs, optionally removes input hydrogens according
to `structure.remove_input_hydrogens`, writes
`intermediate/01_protonation_assigned.pdb`, and produces JSON, CSV, and
Markdown protonation reports. It does not add hydrogens.

PropKa and xTB/GFN2 execution are planned for the next task. Automated
protonation methods are schema-valid but intentionally fail at the protonation
stage until that implementation exists.

The downstream chemistry-producing commands remain placeholders:

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
