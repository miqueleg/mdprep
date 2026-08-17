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

Starter manifests default to the Amber ff14SB protein force field and TIP3P
water. Use `mdprep init --forcefield ... --water-model ...` to select another
supported combination explicitly.

Run the full supported workflow:

```bash
mdprep prepare system.yaml
mdprep validate prepared/final/system.prmtop prepared/final/system.inpcrd
```

Debug by stopping after individual stages:

```bash
mdprep prepare system.yaml --stop-after structure
mdprep prepare system.yaml --stop-after protonation --overwrite
mdprep prepare system.yaml --stop-after refinement --overwrite
mdprep prepare system.yaml --stop-after ligands --overwrite
mdprep prepare system.yaml --stop-after metals --overwrite
mdprep prepare system.yaml --stop-after tleap --overwrite
mdprep prepare system.yaml --stop-after md --overwrite
```

Use `--overwrite` only for mdprep output directories you intend to replace.
The `refinement` stop is available only when `refinement.enabled: true`.
