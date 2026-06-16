# Quickstart

Inspect a PDB:

```bash
mdprep inspect input.pdb
mdprep inspect input.pdb --json
```

Create and validate a starter manifest:

```bash
mdprep init input.pdb -o system.yaml
mdprep config-check system.yaml
```

Run the full supported workflow:

```bash
mdprep prepare system.yaml
mdprep validate prepared/final/system.prmtop prepared/final/system.inpcrd
```

Debug by stopping after individual stages:

```bash
mdprep prepare system.yaml --stop-after structure
mdprep prepare system.yaml --stop-after protonation --overwrite
mdprep prepare system.yaml --stop-after ligands --overwrite
mdprep prepare system.yaml --stop-after tleap --overwrite
```

Use `--overwrite` only for mdprep output directories you intend to replace.
