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

## Input formats

mdprep reads PDB (`.pdb`, `.ent`) and mmCIF (`.cif`, `.mmcif`), chosen by
extension. Structure prediction writes mmCIF -- Boltz emits ModelCIF, and the
RCSB serves mmCIF for everything -- so a predicted model goes in without a
conversion step, in the command line tools and in the web builder alike.

Only the `atom_site` table is read. The chemical component dictionary a file
carries is not consulted, so nothing mdprep decides can be misled by a wrong
one, and a predicted structure is treated exactly like a deposited one.

Two things are worth knowing about a predicted model specifically. It
usually has no hydrogens at all, including on its ligands, so each ligand
needs a `smiles:` stating its protonation. And its B-factor column holds a
confidence score rather than a temperature factor, which mdprep neither reads
nor acts on -- judging whether a low-confidence region belongs in a
simulation is yours.

