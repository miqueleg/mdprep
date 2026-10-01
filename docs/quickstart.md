# Quickstart

Inspect a PDB:

```bash
mdprep inspect input.pdb
mdprep inspect input.pdb --json
```

See which manifest decisions the structure actually requires:

```bash
mdprep plan input.pdb
```

Answer them and write a validated manifest:

```bash
mdprep init input.pdb --interactive -o system.yaml
```

The wizard asks only the questions this structure needs and refuses to default
anything chemistry-sensitive. See [guided_manifests.md](guided_manifests.md).

Or create and validate a starter manifest non-interactively:

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
