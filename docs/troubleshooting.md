# Troubleshooting

## Unknown Heterogens

Add the residue to `ligands:`, configure a one-atom ion under `metals:`, or set
`structure.remove_unknown_heterogens: true` if removal is intended.

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

Inspect `fit_report.json` and the preserved `respgen`/`resp` inputs and
outputs, including generated atom-equivalence constraints.

## Ambiguous QMMESP Mapping

Use unique ligand residue names/selectors. mdprep must map the target ligand
unambiguously in the provisional Amber system.

## Amber Metal Parameter Is Missing

The requested element, formal charge, parameter family, and water model must
exist together in the active AmberTools Li/Merz frcmod. mdprep does not replace
an unavailable combination with a different ion model.

## Undeclared MCPB Bond Candidate

MCPB.py would treat an N/O/S/halogen atom inside `mcpb.cutoff_angstrom` as a
coordinator. Add every intended pair to `mcpb.bonds`, or reduce the cutoff
after reviewing the site. Do not use the cutoff to hide an intended bond.

## MCPB Workflow Is Incomplete

Run with `workflow: prepare_inputs` and `--stop-after metals`, complete the
generated Gaussian/GAMESS jobs, then set `workflow: complete` and supply the
reported artifact paths. See [Metal centers](metals.md).

For `workflow: pyscf` with `geometry_source: external_pdb`, a missing
`optimized_small_model_pdb` is an intentional stop before QM execution.
Optimize the reported MCPB small-model PDB outside mdprep while preserving
exact atom order, names, elements, and residue identity; save it at the
configured path and rerun with `--overwrite`. Do not point the setting back to
MCPB.py's unoptimized step-1 file.

With `geometry_source: qmmm_refinement`, ensure `refinement.enabled: true` and
list that exact bonded site in `refinement.qm_components.metal_sites`. mdprep
rejects an absent or incomplete refinement rather than silently using the raw
MCPB step-1 coordinates.

## MCPB parameter comparison fails

Inspect `mcpb_parameter_comparison.json` before changing a threshold. A term-set
mismatch usually means the candidate and reference did not use the same MCPB
model, atom typing, coordination bonds, or geometry. Force-constant differences
with identical terms reflect QM-method sensitivity, but agreement alone does
not validate the MD model. A non-positive Seminario force constant is an invalid
bonded parameter; do not suppress it by loosening a statistical threshold.

For xTB Hessians, mdprep records a Seminario compatibility-adapter count in
`mcpb.step_2s.stdout.txt`. This is expected when complex-conjugate 3x3-block
eigenpairs sum to a real projected force constant. The adapter accepts only a
negligible imaginary residual; a materially complex value remains a hard
failure and should be investigated as a Hessian/model problem.

For MACE-POLAR-1, inspect `mace_polar1.probe.stderr.txt`,
`mace_polar1.hessian.stderr.txt`, and `mace_polar1_hessian.json`. Confirm that
the configured Python contains MACE 0.3.16 or newer and `graph-longrange`, that
the selected checkpoint supports every element, and that its recorded digest
matches `expected_model_sha256`. A failed force finite-difference check is a
hard error; do not loosen the tolerance until atom order, float64 inference,
checkpoint identity, and geometry have been reviewed. License acceptance must
remain explicit in the manifest.

## 12-6-4 In OpenMM

Load the final Amber `.prmtop` with `AmberPrmtopFile`. OpenMM represents the
12-6 part in `NonbondedForce` and `-C4/r^4` in a separate
`CustomNonbondedForce`; inspecting only the first force is incomplete. The
validation report records detection of both the Amber C4 flag and the OpenMM
r^-4 force.
