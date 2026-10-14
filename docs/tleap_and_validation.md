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

mdprep gives such a ligand a residue name the force fields have not claimed,
derived from its own (`LYS` becomes `LY1`), and uses that name in the leap
input PDB, in a copy of the fitted mol2 written beside the original, and in the
`tleap` script. Atom names, atom order, coordinates and charges are untouched;
only the residue name moves, and the final PDB carries the new name. Every
rename is listed in the tleap report under "Ligand Residue Renames". A ligand
whose name is not already taken keeps it.

The alternative -- binding the ligand unit to `LYS` -- would point every real
lysine in the protein at the ligand's parameters.

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
