# CYP119, a cytochrome P450 with its heme (PDB 1IO7)

The reference case for a **metal inside a cofactor**: a heme's iron, bonded to
the porphyrin's four nitrogens, the proximal cysteine and a water.

`1IO7.pdb` is RCSB entry 1IO7, *Thermophilic cytochrome P450 (CYP119) from
Sulfolobus solfataricus* (Park et al., 2002), verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/1IO7.pdb
```

## What makes it the reference case

- **The iron is deposited inside the heme**: HEM is one residue of 43 atoms,
  the iron among them. mdprep reads the iron out into a residue of its own,
  A:FE1401, and the porphyrin left behind, A:HEM1401, is a ligand that binds
  it through four nitrogens (see
  [docs/metals.md](../../../docs/metals.md#a-metal-inside-a-cofactor-hemes)).
- **The resting state**: Fe(III), six-coordinate and low-spin, with a water
  across the heme plane from the cysteine. Measured in each chain:

  | Chain A | | Chain B | |
  | --- | --- | --- | --- |
  | 1.98 A | HEM1401 NA | 1.87 A | HOH902 O |
  | 1.98 A | HEM1401 NC | 1.97 A | HEM901 NB |
  | 1.99 A | HEM1401 ND | 1.99 A | HEM901 NC |
  | 2.01 A | HEM1401 NB | 2.00 A | HEM901 NA |
  | 2.27 A | CYS317 SG | 2.01 A | HEM901 ND |
  | 2.27 A | HOH1402 O | 2.37 A | CYS817 SG |

- **A thiolate donor**: Cys317 binds the iron deprotonated, as CYM.
- **Two copies**: chains A and B are one protein; the example keeps chain A.

The PDB code HEM does not say whether the iron is Fe(II) or Fe(III), and the
spin follows from the ligands across the heme from each other, so mdprep asks
both. The porphyrin's SMILES and charge, -4 with both propionates ionised, are
proposed, because heme b's chemistry is fixed by its name.

[`examples/19_cyp119_heme_mcpb.yaml`](../../19_cyp119_heme_mcpb.yaml) is the
manifest the decision engine writes for chain A.
