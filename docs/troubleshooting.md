# Troubleshooting

## Unknown Heterogens

Add the residue to `ligands:` or set `structure.remove_unknown_heterogens:
true` if removal is intended.

## Missing External Tool

Install from `environment.yml`, activate the environment, and confirm the
executable is on `PATH`.

## PropKa Or xTB Failure

Inspect the protonation-stage outputs under `protonation/` and the
protonation report.

## antechamber Failure

Inspect ligand-specific stdout/stderr files under
`ligands/<ligand_id>/parameters/`. Confirm ligand charge, multiplicity, atom
names, and input geometry.

## tleap Unknown Residue

Check that configured ligands have final mol2/frcmod files and that PDB residue
names match mol2 substructure names.

## CYX Without Disulfide Pair

Add a `disulfides.force` entry or change the residue state to `CYS`/`CYM`.

## PySCF Did Not Converge

Check ligand charge and multiplicity, use a chemically valid geometry, and
inspect `ligands/<ligand_id>/qm/`.

## Poor RESP Fit

Inspect `fit_report.json`. Equivalent-atom constraints are not implemented in
0.1.0, so fitted charges are atom-specific.

## Ambiguous QMMESP Mapping

Use unique ligand residue names/selectors. mdprep must map the target ligand
unambiguously in the provisional Amber system.
