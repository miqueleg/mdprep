# Developer Notes

## Architecture

The workflow is stage-based:

1. config loading and validation
2. structure inspection/normalization
3. protonation assignment
4. ligand extraction and parameterization
5. final `tleap` build
6. validation and reports

Each stage writes machine-readable reports and should fail before silently
changing unsupported chemistry.

## Future Noncanonical Residues

Add noncanonical residue support at the structure classification and `tleap`
template-loading boundary. Do not treat noncanonical polymer residues as simple
independent ligands.

## Future Metal Centers

Bonded metal-center workflows should live in a dedicated module with explicit
manifest configuration, tests, and reports. Do not infer bonded metal models
from distances alone.

## Adding A Ligand Charge Method

Add a manifest enum value, extraction/parameter workflow branch, mol2
validation, reports, examples, and tests. Preserve atom names, atom order,
coordinates, residue identity, and total charge unless the manifest explicitly
allows a change.

## External Tools

All external commands must use `mdprep.external.runner` and preserve command,
working directory, return code, stdout, stderr, and runtime.

## QMMESP Correctness

MM point charges polarize only the target ligand density. Fit only target
ligand atom charges. Never include environment point charges as fitted centers
or write them to ligand mol2 files.
