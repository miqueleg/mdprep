# PrnA, a flavin-dependent halogenase (PDB 2AR8)

The reference case for an **organic cofactor whose chemistry its name fixes**:
the tryptophan 7-halogenase PrnA with its FAD and its product,
7-chloro-L-tryptophan.

`2AR8.pdb` is RCSB entry 2AR8 (Dong et al., *Science* 2005), at 2.2 A,
verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/2AR8.pdb
```

## What makes it the reference case

- **FAD**, A:FAD600, with all 53 heavy atoms and no hydrogens. Its SMILES and
  net charge, -2, are proposed: oxidised FAD, the diphosphate fully ionised,
  the flavin's N3 protonated. The reduced FADH- that reacts with O2 in the
  catalytic cycle is a different molecule, and is typed in if that is the
  state you simulate.
- **The product**, A:CTE650, 7-chloro-L-tryptophan. Nothing proposes a
  substrate's or a product's chemistry: its SMILES and charge are typed in,
  here as the zwitterion, `[NH3+]C(Cc1c[nH]c2c(Cl)cccc12)C(=O)[O-]`, charge 0.
- **A chloride**, kept as a crystal ion and built from Amber's ion library.
- **No metal**, so there is no QM/MM refinement and no MCPB step: AM1-BCC
  charges for both ligands, then tleap.

[`examples/21_prna_fad_halogenase.yaml`](../../21_prna_fad_halogenase.yaml)
is the manifest the decision engine writes for it. See
[docs/ligands.md](../../../docs/ligands.md#cofactors-whose-smiles-is-proposed)
and [docs/enzyme_families.md](../../../docs/enzyme_families.md).
