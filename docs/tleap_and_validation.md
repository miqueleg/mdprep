# tleap And Validation

The final stage writes a leap-ready PDB, loads protein/water force fields,
loads ligand mol2/frcmod files, emits disulfide bond commands, and runs
`tleap`.

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

Reports are written under `reports/`.
