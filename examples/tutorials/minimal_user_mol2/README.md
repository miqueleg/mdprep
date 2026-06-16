# Minimal user_mol2 Tutorial

This is a tiny toy system for checking mdprep mechanics. It contains only one
two-atom ligand residue and a user-provided mol2/frcmod pair. It is not a
scientifically meaningful MD system.

From this directory:

```bash
mdprep config-check system.yaml
mdprep prepare system.yaml --overwrite
mdprep validate prepared/final/system.prmtop prepared/final/system.inpcrd
```

The final build requires `tleap` from AmberTools.
