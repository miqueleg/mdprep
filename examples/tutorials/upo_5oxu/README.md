# An unspecific peroxygenase, with its glycans (PDB 5OXU)

The reference case for **heterogens that are not ligands**: an unspecific
peroxygenase (UPO), a fungal heme-thiolate enzyme that hydroxylates,
epoxidises and halogenates, deposited as the glycoprotein it is and with what
it was crystallised from.

`5OXU.pdb` is RCSB entry 5OXU, the laboratory-evolved *Agrocybe aegerita*
peroxygenase (Ramirez-Escudero et al., *ACS Chem. Biol.* 2018), at 1.47 A,
verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/5OXU.pdb
```

## What makes it the reference case

- **Five N-glycans.** A:NAG404-408, each bonded to an asparagine, as the
  file's LINK records declare. mdprep does not build glycans, and asks once
  whether to remove all five (proposed yes).
- **Fourteen crystallisation additives**: a phosphate, three glycerols and ten
  methanols. Asked about once, proposed for removal, every one listed; none
  is asked about as a ligand unless they are kept.
- **A heme-thiolate iron**, as in a P450. The iron is read out of the heme
  into A:FE401 and bonded to the porphyrin's four nitrogens (1.92-2.09 A),
  Cys36 SG (2.43 A) and a water, HOH570 (2.04 A). The heme's SMILES and
  charge, the cysteine as a thiolate and the site's charge, -2, are proposed
  as for CYP119.
- **A magnesium on the heme.** A:MG402 is held by a heme propionate oxygen
  (2.18 A), Glu122, Gly123's carbonyl, Ser126 and two waters. Magnesium is
  proposed nonbonded. The heme coordinates both metals and takes its charges
  from the iron's MCPB fit; the QM/MM refinement of the iron's site treats the
  magnesium as part of the MM environment.

[`examples/20_upo_5oxu_heme_glycans.yaml`](../../20_upo_5oxu_heme_glycans.yaml)
is the manifest the decision engine writes for it. See
[docs/heterogens.md](../../../docs/heterogens.md) and
[docs/enzyme_families.md](../../../docs/enzyme_families.md).
