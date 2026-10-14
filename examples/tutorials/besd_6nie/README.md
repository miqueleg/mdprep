# BesD, an alpha-ketoglutarate-dependent halogenase (PDB 6NIE)

The reference case for the student-facing workflow, and the structure the
tests in `tests/test_besd_halogenase_case.py` measure against.

`6NIE.pdb` is RCSB entry 6NIE, *BesD with Fe(II), chloride, and
alpha-ketoglutarate*, with **only** the `ANISOU` records removed. Nothing in
mdprep reads anisotropic B-factors and they are half the file; every
coordinate, header and `REMARK` is the deposition verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/6NIE.pdb
```

## What makes it the reference case

- **Four copies** of the complex in the asymmetric unit, so a build has to
  select one chain: MCPB inputs are single-chain.
- **Fe(II) in a 2-His facial triad** with chloride occupying the position a
  carboxylate takes in a hydroxylase. Measured in chain A:

  | Distance | Donor |
  | --- | --- |
  | 2.16 A | AKG304 O1 |
  | 2.18 A | AKG304 O5 |
  | 2.22 A | HIS204 NE2 |
  | 2.25 A | HIS137 NE2 |
  | 2.39 A | CL303 CL |

  The sixth site is open; that is where O2 binds.
- **A coordinating cofactor**, alpha-ketoglutarate, which therefore takes its
  charges from the metal site's MCPB fit rather than an independent AM1-BCC
  calculation.
- **A lysine substrate** with no hydrogens, like every deposited ligand.
- **33 residues with heavy atoms missing** (`REMARK 470`) and an unresolved
  N-terminal expression tag (`REMARK 465`), which is what the completeness
  detection and the `unresolved_residues` policy exist for.

The manifest is
[`examples/16_besd_halogenase_gfn2_mcpb.yaml`](../../16_besd_halogenase_gfn2_mcpb.yaml).
