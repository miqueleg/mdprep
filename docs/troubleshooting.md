# Troubleshooting

## Unknown Heterogens

Add the residue to `ligands:`, configure a one-atom ion under `metals:`, list
it under `structure.remove_residues` with the reason it goes, or set
`structure.remove_unknown_heterogens: true` if removal is intended.

## Glycans, Bonded Heterogens Or Modified Residues Stop The Build

mdprep builds neither sugars nor other heterogens bonded to the protein, nor a
modified amino acid inside a chain. Answering no to removing the first two, or
to replacing the third by its standard parent, stops the build at that
question. See [heterogens.md](heterogens.md).

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

## xTB Hessian Failed: SCF Not Converged

An open-shell metal site with near-degenerate d orbitals -- a low-spin Fe(III)
heme is the common one -- can make xTB's SCF oscillate without converging, and
more cycles do not help. Smear the occupations instead:

```yaml
      pyscf:
        xtb:
          electronic_temperature_kelvin: 1000.0
```

The error names this setting when it applies. The builder writes it for a site
whose donors include a heme. See
[metals.md](metals.md#smearing-the-xtb-occupations).

## A Metal Cluster, A Bonded Heme Or A Vanadate Is Refused

mdprep models a metal as one atom in a site of its own. A residue still holding
several metal atoms once single metals have been read out of their cofactors --
an iron-sulfur cluster -- cannot be described that way, and heme c is bonded to
two cysteines, which makes it a covalent ligand. A myeloperoxidase's heme is
bonded to the protein too, through the bonds its file declares. A vanadate is
not split: what would remain is four oxygens bonded to nothing. All are refused
before any question is asked rather than failing in antechamber or tleap later.

## Two Bonded Sites Would Each Need A Fit

mdprep fits one bonded MCPB site per preparation. In an oligomer, give every
copy the same parameters: the other copies' sites then reuse the one fit
(`parameters_from`). A copy whose site is not a copy of the first -- one donor
more or less, or a carboxylate bound through the oxygen named the other way
round -- has to be modelled nonbonded. See [oligomers.md](oligomers.md).

## Two Residues With One Name Differ

tleap builds every residue of one name from one template, so two ligands
called AKG, or two metals called FE2, cannot carry different charges or
parameters. Give them the same answers -- mdprep then parameterises the name
once -- or rename one of them in the input.

## A Copy Cannot Take The Template

A copy of a bonded site takes the residue templates fitted for the first copy,
atom for atom. A copy residue protonated differently, with a different
residue beside a coordinating side chain, or with other atom names, is
refused; make the copies alike, or answer no to sharing parameters and model
that copy's metal nonbonded.

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

## CUDA Fails But OpenCL Works

`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` means the CUDA runtime installed with
OpenMM is newer than the GPU's driver. Install one no newer than the version
`nvidia-smi` prints, `micromamba install -n mdprep-runtime -c conda-forge
"cuda-version<=12.4"`, or rerun section C of the official notebook, which does
it. See [Roe--Brooks OpenMM MD](molecular_dynamics.md#cuda-and-the-gpu-driver).

## 12-6-4 In OpenMM

Load the final Amber `.prmtop` with `AmberPrmtopFile`. OpenMM represents the
12-6 part in `NonbondedForce` and `-C4/r^4` in a separate
`CustomNonbondedForce`; inspecting only the first force is incomplete. The
validation report records detection of both the Amber C4 flag and the OpenMM
r^-4 force.

## Particle Coordinate Is NaN During MD

Historically caused by two defects in the Roe--Brooks stage, both fixed:
positional restraints were applied to free `Na+`/`Cl-` ions because the ion
residue names Amber writes do not match a bare element symbol, and coordinates
were wrapped into the periodic box between stages while the restraint
references stayed unwrapped. Either one puts a restrained particle a full box
length from its reference, which at 5 kcal/mol/A^2 is a restraint force of
order 1e5 kJ/mol/nm.

Two causes found on a bonded iron site (SyrB2, PDB 2FCT) are now handled at
build time:

- **A peptide bond across a missing loop.** tleap bonds consecutive residues of
  a chain unless a TER separates them. mdprep now writes a TER wherever a
  residue's C is more than 2 A from the next residue's N, measured rather than
  read from the numbering, so the two ends become termini.
- **A 1-4 pair through the metal with no repulsion.** See
  [1-4 pairs through the metal](metals.md#1-4-pairs-through-the-metal).

If a NaN still appears, inspect `md_report.json`: a stage whose
`potential_energy_kj_mol` jumps by a large positive amount relative to the
previous stage indicates the input is strained rather than the protocol
diverging. Check the input with a single-point energy before heating.

## Final PDB Atom Serials

The PDB atom-serial field is five columns wide and cannot hold a sixth digit,
so systems above 99,999 atoms are written in one of three incompatible ways.
mdprep reads all of them: `tleap`/`ambpdb` widen the field into column 12,
cctbx-derived tools switch to hybrid-36 (`A0000` follows `99999`), and writers
that wrap around leave duplicates, which mdprep renumbers in file order and
reports as a warning. Duplicated serials are only fatal when the file also
contains `CONECT` records, because the connectivity they encode cannot then be
resolved; supply a file with unique or hybrid-36 serials in that case.

mdprep writes the same convention it reads, so a structure above 99,999 atoms
round-trips through `read_pdb`/`write_pdb` unchanged.
