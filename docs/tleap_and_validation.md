# tleap And Validation

The final stage writes a leap-ready PDB, loads protein/water force fields,
loads ligand mol2/frcmod files, emits disulfide bond commands, and runs
`tleap`. Configured nonbonded ion templates or completed MCPB.py templates,
parameters, residue renames, and explicit bonds are composed into the same
mdprep-owned scripts.

## Ligands Named After A Force-Field Residue

`tleap` resolves every residue in the input PDB by name against the templates
the sourced force fields define, and for an amino-acid name it maps the name to
its terminal form first -- a lysine that opens a chain becomes `NLYS`. Nothing
a manifest loads can intercept that, so a free amino acid declared under
`ligands:` -- BesD's lysine substrate, a free proline, a deposited dipeptide --
would be built from the protein template and `tleap` would stop on the first
ligand-named hydrogen it cannot type.

mdprep refuses such a ligand and names it, saying which force field template
claims the name and what to do:

```
Residue name the Amber force fields already define: LYS (an amino acid;
a_lys_301). tleap resolves every residue by name against the force field
templates, so it would build this residue from that template instead of from
parameters of its own. An amino-acid name would also redirect every matching
residue in the protein to it. Rename it in the input structure to a code no
force field uses (LIG, or any unused three-character name), then run mdprep
again.
```

The refusal is raised twice over: by `plan_decisions`, so the GUI rejects a
structure before asking a single question, and in `prepare_system`, against
the normalized structure, before protonation or any quantum chemistry. Both
use one wording, from `describe_reserved_residue_names`.

mdprep used to rename the ligand for the build instead -- `LYS` became `LY1`
in the leap input, a copy of the fitted mol2 and the `tleap` script. That kept
`tleap` happy and moved the problem: the prepared system then described a
residue under a name the input never used, so the MCPB site builder, the
QM/MM point-charge reader and the refinement selection each had to know about
the substitution, and some of them did not. Nobody can tell what a residue
called `LYS` is meant to be, so mdprep says so and stops, and the fix happens
once, in the input.

Binding the ligand unit to `LYS` instead would point every real lysine in the
protein at the ligand's parameters, which is worse again.

### Renaming it

The input is the user's to change. The web builder offers the change rather
than leaving it to a text editor, because the structure this matters most for
is 6NIE -- fetched by PDB id, substrate deposited as `LYS A 301` -- and shows
the refusal with a button that applies it. The rewrite goes through mdprep's
own reader and writer (`mdprep.structure.rename.rename_residues`), so the
structure the page plans, draws and ships to Colab is one file rather than
three descriptions of one. Renaming is by residue identity and never by name:
6NIE's four chains hold 32 lysines and only 4 of them are the substrate.

Accepting the rename also drops the page's record of the PDB id, because the
Colab cell re-downloads by id when there is one -- which would fetch the
deposition that was just refused. Without an id the page hands over the
bundle, which carries the renamed structure.

Supported choices:

- protein: `ff14SB`, `ff19SB`
- water: `TIP3P`, `OPC`
- ligand atom types: `gaff`, `gaff2`
- solvation: truncated octahedron or rectangular box

Final outputs:

```text
final/system.prmtop
final/system.inpcrd
final/system.pdb
```

Validation checks:

- output files exist and are non-empty
- final PDB is parseable
- configured ligands are present with expected atom names
- water presence matches solvation settings
- disulfide consistency
- ParmEd topology/coordinate load when available
- OpenMM finite-energy sanity check when requested and available
- for 12-6-4 topologies, Amber C4 flag creation and OpenMM r^-4 custom-force
  construction when OpenMM validation is available

For 12-6-4 models, ParmEd post-processes each published dry/solvated topology.
The command, input backup, output, and logs are preserved. OpenMM exposes the
total potential as ordinary electrostatics/12-6 plus a separate r^-4
`CustomNonbondedForce`; both are required.

Reports are written under `reports/`.
