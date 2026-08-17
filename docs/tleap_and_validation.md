# tleap And Validation

The final stage writes a leap-ready PDB, loads protein/water force fields,
loads ligand mol2/frcmod files, emits disulfide bond commands, and runs
`tleap`. Configured nonbonded ion templates or completed MCPB.py templates,
parameters, residue renames, and explicit bonds are composed into the same
mdprep-owned scripts.

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
