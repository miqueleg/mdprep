# Heterogens That Are Not Ligands

A deposition's heterogens are not all molecules to parameterise. Some are
pieces of the protein: a selenomethionine is a methionine labelled to solve
the phases, written as a heterogen because its name is not one of the twenty.
Some are bonded to it: the sugars of a glycoprotein, a covalent intermediate,
the heme of a myeloperoxidase. Many are what the crystal was grown in:
glycerol, sulfate, PEG, methanol.

Offering each of these as a ligand asks for the SMILES and charge of fifteen
glycerols, and builds a sugar bonded to an asparagine as a free molecule at
bond length from it. So mdprep sorts them first, and asks about each group as
a group. Nothing is removed without the manifest saying so: every removal is
listed under `structure.remove_residues` with its reason, and reported.

## How mdprep knows

A bond cannot be told from a clash by distance: a predicted complex has ligand
atoms 1.8 A from the protein that are not bonded to it. A deposition says
which atoms are bonded across residues, and mdprep reads that:

| | PDB file | mmCIF |
| --- | --- | --- |
| Bonds between residues | `LINK` | `_struct_conn` |
| Modified polymer residues and their parents | `MODRES` | `_pdbx_struct_mod_residue` |

A bond to a symmetry mate is not a bond in the molecule being built and is
ignored. Two things are also found without them, for a file another program
rewrote or a prediction:

- a heterogen with a backbone (`N`, `CA`, `C`) peptide-bonded to the protein
  is a residue of the chain; a free amino acid, which has the same atoms and no
  peptide bond, stays a ligand;
- a carbohydrate (NAG, MAN, BMA, FUC, ...) with an atom within 1.75 A of the
  protein's is bonded to it. Nothing else sits that close to a protein in a
  crystal; a sugar is the one heterogen for which distance is enough.

`mdprep plan` lists what it found under `modified_residues`, `glycans`,
`covalent_heterogens` and `additives`.

## Modified residues

A modified amino acid inside a chain is built either as its standard parent
or as it is. Each residue name is one question, offering what applies to it:

- **replace it** by its parent (`MSE` with `MET`, `CSO` with `CYS`, `SEP` with
  `SER`, `PCA` with `GLN`), which keeps the backbone and the atoms the two
  share and builds the rest through PDBFixer:

  ```yaml
  structure:
    repair:
      backend: pdbfixer
      replace_modified_residues:
        - selector: {chain: A, resname: MSE, resid: 14}
          parent: MET
  ```

- **keep it with AmberTools' template**, where AmberTools ships one fitted to
  the protein force field (phosphoserine, phosphothreonine, phosphotyrosine,
  acetyl-lysine, selenomethionine, hydroxyproline...);
- **keep it, parameterised from its SMILES**, as a ligand would be, and bonded
  into the chain.

Keeping one is [noncanonical_residues.md](noncanonical_residues.md). For
selenomethionine replacing it is proposed: the selenium is how the methionine
was labelled, not chemistry of the protein. For anything else nothing is
proposed, because the modification may be what the structure is about: a
phosphoserine replaced by serine has lost its phosphate. The parent comes
from the file's `MODRES` records when it has them, otherwise from a short
table in `mdprep.structure.heterogens`. A residue with no known parent can only
be kept; one that can be neither -- no known parent, and at the end of its
chain, where a kept residue cannot be -- is refused when the structure is
loaded.

A modified residue the manifest neither keeps, replaces nor removes (with
`structure.remove_residues`) stops the build as the structure is read,
whatever `remove_unknown_heterogens` says. It used to be removed with the
unknown heterogens, which cut its chain in two.

A replacement needs nothing else repaired: what the parent has and the
modified residue lacked is built whether or not `add_missing_heavy_atoms` is
set, and a chain's terminal `OXT` is left to tleap, as it is when nothing is
repaired.

A modified residue deposited with a metal inside it -- a cysteine and the zinc
it binds written as one residue -- is read as a heme is
([metals.md](metals.md#a-metal-inside-a-cofactor-hemes)): the metal is
moved into a residue of its own, written after the chain, and the residue is
what remains. One whose metal is bonded to more than the amino acid, such as
S-(methylmercury)cysteine, stays whole and can only be replaced by its parent;
see [noncanonical_residues.md](noncanonical_residues.md#metals-and-kept-residues).

A pyroglutamate (`PCA`), the N-terminus of many secreted proteins, is a
glutamine whose side chain has closed onto its own nitrogen as a lactam. The
two share atom names but not their places -- kept where they are, the
glutamine's CD would sit 1.5 A from its backbone N -- so a side chain bonded
to its own backbone (only proline's may be) is deleted beyond CB and built
again from the parent's template. The report lists the rebuilt atoms under
`side_chain_rebuilt`, and a warning says how close the new side chain comes
to its neighbours: PDBFixer places it in its template's rotamer and only
pushes apart atoms closer than 1.3 A, so on 1CPO the glutamine's NE2 starts
1.5 A from Ser4's OG, and tleap puts Ser4's hydroxyl hydrogen almost on top of
it. Minimise before anything else: a plain minimisation of the prepared 1CPO
separates them, leaving Ser4's hydroxyl hydrogen-bonded to Glu1's carbonyl and
every bond of both residues at its length. The replacement also changes the
chemistry: a pyroglutamate's N-terminus is a neutral lactam, a glutamine's a
charged amine.

## Glycans

Sugars bonded to the protein, and every sugar bonded to those. mdprep does not
build glycans -- their force field, GLYCAM, is not part of it -- so the
question is whether to remove them, and the proposal is yes: the protein is
prepared as if deglycosylated, which is the usual choice when they are far
from the site being studied. Check that none is near yours. Answering no stops
the build.

## Heterogens bonded to the protein

Anything else the file declares bonded to the protein: a covalent
intermediate, an inhibitor, a cofactor attached to a side chain. mdprep does
not build covalent ligands, so the question is whether to remove them, and
nothing is proposed, because the intermediate may be what the structure was
solved to show. Answering no stops the build.

Refused when the structure is loaded, with the reason:

- a bonded cofactor that holds a metal -- the heme of a myeloperoxidase or
  lactoperoxidase, held by ester and sulfonium bonds; removing it would take
  its metal site with it;
- a bonded heterogen that coordinates a metal;
- a declared bond between two protein residues that is neither a peptide bond
  nor a disulfide (a Tyr-Cys or His-Tyr crosslink), which tleap would build
  unbonded.

## Crystallisation additives

Components of buffers, cryoprotectants and precipitants: GOL, EDO, PEG, PG4,
PGE, 1PE, P6G, 2PE, 12P, 15P, PE4, PE5, P33, MPD, MRD, MOH, EOH, IPA, DMS, ACT,
ACY, FMT, SO4, PO4, NO3, NH4, TRS, EPE, MES, BTB, B3P, CIT, FLC, MLI, IMD, BME,
DTT and CAC. Removing them is proposed, every one listed with what it is. Some
are not artefacts -- a sulfate can sit where a substrate's phosphate binds, an
alcohol in a substrate tunnel -- so check the list against your active site.
Answer no and each is asked about as a ligand instead, where it can be
included or declined one by one.

A heterogen that coordinates a metal is part of that site, and is never on
this list whatever its name.

## Metals that are not part of the system

A metal that is not inside a cofactor can also be removed: a nickel from a
His-tag purification, a cadmium or zinc from the crystallisation buffer. Its
model question offers **Remove it** besides bonded and nonbonded, and nothing
more is asked about it. A cofactor's metal -- a heme's iron -- cannot be
removed without its cofactor.

Magnesium and calcium are proposed nonbonded even with a coordination shell:
they are closed-shell ions held by electrostatics, which an ion model
describes. Every other metal with a shell is proposed bonded.

## What is refused outright

- **Metal clusters**: a residue holding several metal atoms, such as an
  iron-sulfur cluster (SF4, FES, F3S) -- the [4Fe-4S] clusters of a reductive
  dehalogenase, the [2Fe-2S] of a Rieske oxygenase.
- **Oxoanions**: a metal whose other atoms are bonded to nothing but it,
  such as the vanadate of a vanadium haloperoxidase. It is neither a metal
  site mdprep can build around a one-atom metal nor a ligand antechamber can
  parameterise.

See [enzyme_families.md](enzyme_families.md) for how each family of
halogenating and dehalogenating enzymes is handled.
