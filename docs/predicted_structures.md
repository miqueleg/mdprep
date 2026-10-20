# Predicted Structures

There is often no crystal structure of the exact enzyme you want to simulate:
a point mutant, a homologue from a sequence database, a variant designed last
week. You then build the starting model yourself, in three steps:

1. **Predict the complex** -- protein, metal, cofactors and substrate -- with a
   structure predictor.
2. **Find the closest X-ray structure** of a related protein.
3. **Bring its crystal waters across**, above all any water bound to the
   metal, which a prediction never contains.

The result is an ordinary PDB file for the
[web input builder](https://miqueleg.github.io/mdprep/) or `mdprep init
--interactive`, and from there the preparation is the same as for a crystal
structure. The [official Colab notebook](../examples/mdprep_colab.ipynb) runs
steps 2 and 3 for you.

## 1. Predict the complex

The [ColabFold](https://github.com/sokrypton/ColabFold) notebooks run in
Google Colab and need nothing installed. Two of them place ligands:

| Notebook | Protein | Metal and cofactors | Substrate | Output |
|---|---|---|---|---|
| `Boltz1.ipynb` | `query_sequence` | `ligand_input_ccd`, colon-separated: `FE2:AKG:CL` | `ligand_input` as SMILES, or a CCD code | mmCIF |
| `AlphaFold3_of3.ipynb` (`model`: `openfold3`) | `protein` | `ligand_ccd`, colon-separated: `FE2:AKG:CL` | `ligand_smiles` | mmCIF |

The AlphaFold2 notebooks do not place ligands, so they are not suitable here.
Boltz can also be run locally (`boltz predict input.yaml --use_msa_server`)
with the same inputs; mdprep was tested on its output.

What to give the predictor:

- **The sequence of your construct**, without purification tags. A His-tag
  the crystal never resolved is a disordered tail in the model and a floppy
  appendix in the simulation.
- **Metals and cofactors by CCD code.** The code carries the chemistry:
  `FE2` is Fe(II) and `FE` is Fe(III), so mdprep proposes the oxidation state
  from it. For an Fe(II)/alpha-ketoglutarate enzyme: `FE2`, `AKG`, and `CL` or
  `BR` for a halogenase.
- **The substrate as SMILES in the protonation state it binds in** -- at pH 7
  an amino acid is a zwitterion: L-proline is `C1C[C@H]([NH2+]C1)C(=O)[O-]`.
  A CCD code works too when the substrate has one.

What to check before going on:

- **Confidence.** pLDDT is in the B-factor column (Boltz writes it 0-100).
  ipTM and ligand ipTM above about 0.8 mean the complex is predicted
  confidently; the PAE between the protein and the ligands says how sure the
  model is of where they sit.
- **The metal site.** The 2-His-1-carboxylate facial triad, alpha-ketoglutarate
  bidentate (keto and C1 carboxylate oxygens), the halide where your enzyme
  binds one. Distances to the metal of 1.9-2.4 A.
- **The substrate**, near the metal and in a pose that makes chemical sense.

A prediction has no waters and no hydrogens. mdprep adds the hydrogens; the
waters are the next two steps.

## 2. Find the closest X-ray structure

```bash
mdprep closest-xray model.cif
```

This sends the model's sequence to the RCSB sequence search and lists the
X-ray structures with a similar chain: identity, coverage, resolution, how many
waters were modelled, and which ligands are bound. Among equally close hits,
those holding your model's metal come first, because a crystal without the
metal cannot have the water bound to it. The same search is on
[rcsb.org](https://www.rcsb.org) under *Advanced Search > Sequence*.

Choose a structure **in the same state as your model**. Fe(II)/alpha-KG
enzymes release the water on the iron when the substrate binds -- the
"substrate trigger" that opens the site for O2. Trans-3/4-proline hydroxylase
H11 shows it: 7E06 (Fe and alpha-KG) has a water on the iron in chain A, and
7E07 (Fe, alpha-KG and proline) has none in either chain. A model with the
substrate bound should not inherit a water from a substrate-free crystal
without thought.

## 3. Transplant the waters

```bash
mdprep transplant-waters model.cif --reference 2FCT
```

`--reference` takes a PDB ID (downloaded from the RCSB) or a file. The output
is `model_waters.pdb` -- the model, unmoved, plus the waters that fit -- and
`model_waters.json`, a report of everything decided.

How it works:

1. **PyMOL pairs the residues** of each protein chain of the model with each
   chain of the reference, by sequence (`align`) and by structure alone
   (`super`, for distant homologues). The fit is a Kabsch superposition with
   PyMOL's outlier rejection, scored by TM-score; each model chain takes the
   best reference chain, and only that chain's waters.
2. **A water bound to a metal moves with it.** The model's iron is rarely
   exactly where the crystal's is, and a water 2.2 A from the crystal's iron
   can be 2.9 A from the model's. It is shifted by however far the metal is
   displaced, so it stays bound.
3. **Waters that do not fit are dropped**: closer than 2.5 A to a nitrogen,
   oxygen or halide of the model, or 2.9 A to a carbon or sulfur, or to another
   transplanted water; or farther than 6 A from the model (bulk solvent, which
   the solvation box provides). Metals are not part of the clash test.
4. **The waters join the protein chain nearest them**, numbered after its last
   residue, so a chain selection later keeps them.

It refuses a reference that does not share the model's fold (TM-score below
0.5), and warns when the reference is distant (TM-score below 0.7 or identity
below 30%), when another copy in the crystal has a water on its metal that the
copy used does not, and when a water bound to the reference's metal no longer
fits in the model -- often because the model put the substrate there.

Useful options:

| Option | Use |
|---|---|
| `--reference-chain A` | Take the waters from this chain of the reference, e.g. the copy with a water on its metal. |
| `--around-metal 8` | Keep only waters within 8 A of a metal: the active site, nothing else. |
| `--clash-polar`, `--clash-apolar` | Change the clash distances. |

PyMOL itself is a dependency of this command only: `conda install -c
conda-forge pymol-open-source`. The Colab notebook installs it.

### The same thing by hand in PyMOL

For a quick look, or to understand what the command does. (`model` is a
reserved word in PyMOL selections, so the prediction is called `pred`.)

```
load model.cif, pred
fetch 2FCT, xray
remove xray and not alt ''+A
align xray and chain A and polymer.protein, pred and polymer.protein
select keep, (xray and chain A and resn HOH) within 6 of pred
select clash, keep within 2.5 of (pred and not elem Fe)
create waters, keep and not clash
save model_waters.pdb, pred or waters
```

On the SyrB2 model this keeps 251 waters, the one on the iron (HOH1072)
among them. It does not move a bound water with its metal, and it uses one clash
distance for every element; `mdprep transplant-waters` does both properly,
chooses the reference chain for you, and reports what it did.

## 4. Prepare it

Load `model_waters.pdb` into the web input builder. A transplanted water on
the metal is proposed as a donor with the rest of the shell; confirm it and it
becomes part of the MCPB site (see
[Waters bound in a metal site](metals.md#waters-bound-in-a-metal-site)).

Residue names that did not fit a PDB column -- a predictor's `LIG1` -- were
renamed by mdprep, and the file records that they were assigned, so nothing
is inferred from them: an iron that came in as `LIG1` and is now `FE` gets no
oxidation state proposed from its name, and you are asked for it.

## Worked example: SyrB2

SyrB2 (the halogenase of syringomycin biosynthesis, [PDB
2FCT](https://www.rcsb.org/structure/2FCT)) predicted with Boltz-2 from its
sequence plus `FE2`, `AKG` and `CL`: ipTM 0.99, pLDDT 94.
`mdprep closest-xray` returns 2FCT first (100% identity, Fe(II), Cl-,
alpha-KG). `mdprep transplant-waters` superposes it with a TM-score of 0.96
and an RMSD of 0.26 A, transplants 241 waters, and moves HOH1072 0.41 A with
the iron, onto the model's at 2.24 A. The decision engine then proposes
the six-coordinate shell -- two histidines, bidentate alpha-KG, the chloride
and the water -- and a model charge of -1 (Fe +2, alpha-KG -2, Cl -1). The
system prepares and equilibrates with the site intact.
