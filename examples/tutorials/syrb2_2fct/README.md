# SyrB2, a halogenase with a water on its iron (PDB 2FCT)

The reference case for a **water bound to a bonded metal**, and the example
the [official Colab notebook](../../mdprep_colab.ipynb) runs without any input.

`2FCT.pdb` is RCSB entry 2FCT, *SyrB2 with Fe(II), chloride, and
alpha-ketoglutarate* (the halogenase of syringomycin biosynthesis), verbatim.
Re-download it with:

```bash
curl -O https://files.rcsb.org/download/2FCT.pdb
```

## What makes it the reference case

- **A six-coordinate Fe(II)**: two histidines, alpha-ketoglutarate bidentate,
  a chloride, and a water where O2 later binds. Measured in chain A:

  | Distance | Donor |
  | --- | --- |
  | 2.03 A | AKG897 O2 |
  | 2.18 A | HIS116 NE2 |
  | 2.19 A | AKG897 O5 |
  | 2.22 A | HIS235 NE2 |
  | 2.24 A | HOH1072 O |
  | 2.47 A | CL901 CL |

  The water is an MCPB coordinator like the others (see
  [docs/metals.md](../../../docs/metals.md#waters-bound-in-a-metal-site)).
- **A free crystal chloride**, CL903, which is a crystal ion, not a ligand.
- **A detergent**, DSU896, which is removed.
- **Unresolved loops**: residues 57-60 of chain A are missing, and a TER is
  written there so tleap does not bond 56 to 61.
- **Two copies**, so the build keeps chain A.

[`examples/17_syrb2_bound_water_mcpb.yaml`](../../17_syrb2_bound_water_mcpb.yaml)
is the manifest the decision engine writes for it, and
[docs/predicted_structures.md](../../../docs/predicted_structures.md#worked-example-syrb2)
rebuilds the same system from a Boltz prediction.
