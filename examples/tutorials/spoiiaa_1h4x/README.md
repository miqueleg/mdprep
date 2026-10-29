# SpoIIAA, phosphorylated on Ser57 (PDB 1H4X)

The reference case for a **noncanonical residue kept in its chain**: the
Bacillus cell-fate determinant SpoIIAA, phosphorylated on Ser57.

`1H4X.pdb` is RCSB entry 1H4X (Seavers et al., *Structure* 2001), at
1.16 A, verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/1H4X.pdb
```

## What makes it the reference case

- **A phosphoserine in each chain**, A:SEP57 and B:SEP57, declared by
  `MODRES` records and bonded to their neighbours by `LINK`s. Replacing one by
  serine would take away the phosphate the structure was solved to show, so
  nothing is proposed: the student chooses between replacing it and keeping
  it.
- **Kept, it can be built two ways.** AmberTools ships a phosphoserine
  template fitted to ff14SB and ff19SB (phosaa14SB, phosaa19SB): SEP is the
  dianion, S1P the monoanion, and the dianion is proposed, the phosphate's
  second pKa being about 6. Or mdprep parameterises it from its SMILES, as it
  would a ligand: AM1-BCC charges on the residue between an acetyl and a
  methylamide, the backbone's fixed to the force field's, GAFF2 on the side
  chain. Its SMILES and charge are proposed too.
- **Two copies.** Chain A's phosphate has two alternate conformations at half
  occupancy each; chain B's has one, and chain B is prepared. With both
  chains kept, both phosphoserines are one residue name, built from one
  template.
- **A Tris** beside each chain, from the buffer, removed as a
  crystallisation additive.

[`examples/22_spoiiaa_phosphoserine.yaml`](../../22_spoiiaa_phosphoserine.yaml)
is the manifest the decision engine writes for it. See
[docs/noncanonical_residues.md](../../../docs/noncanonical_residues.md).
