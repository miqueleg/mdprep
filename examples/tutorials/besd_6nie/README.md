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
- **A lysine substrate** with no hydrogens, like every deposited ligand --
  and recorded as `LYS A 301`, which is where this entry stops a run that
  has not been prepared for it. See below.
- **33 residues with heavy atoms missing** (`REMARK 470`) and an unresolved
  N-terminal expression tag (`REMARK 465`), which is what the completeness
  detection and the `unresolved_residues` policy exist for.

The manifest is
[`examples/16_besd_halogenase_gfn2_mcpb.yaml`](../../16_besd_halogenase_gfn2_mcpb.yaml).

## The substrate has to be renamed first

The deposition records the substrate as `LYS A 301`. Every Amber protein force
field defines `LYS`, and `tleap` resolves residues by name against those
templates, so it would build the substrate from the protein template -- as
`NLYS`, since it opens a chain -- and bind every real lysine in the chain to
the ligand's parameters. mdprep refuses to build it and says so, in the GUI
before it asks anything and on the command line before any quantum chemistry.

The fix is in the input, once. The web builder offers it: load 6NIE, and the
page shows the refusal with a button that renames the four substrate copies to
`LIG` and nothing else -- the 28 lysines in the protein keep their name. From
there the flow is unchanged.

By hand, or for the command-line path:

```python
from mdprep.structure.pdb import read_pdb
from mdprep.structure.rename import rename_residues
from mdprep.structure.writer import write_pdb

structure = read_pdb("6NIE.pdb")
write_pdb(
    rename_residues(
        structure,
        [
            {"chain": chain, "resname": "LYS", "resid": 301, "new_resname": "LIG"}
            for chain in "ABCD"
        ],
    ),
    "6NIE_prepared.pdb",
)
```

`examples/16_besd_halogenase_gfn2_mcpb.yaml` declares the substrate as `LIG`
and expects `6NIE_prepared.pdb`. The test suite builds the same file from the
vendored deposition in `tests/besd_structure.py`, so `6NIE.pdb` here stays the
deposition, verbatim.
