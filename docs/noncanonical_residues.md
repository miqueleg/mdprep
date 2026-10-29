# Noncanonical Residues

A residue of a protein chain that is not one of the twenty -- a
phosphoserine, an acetyl-lysine, a p-azido-phenylalanine -- is found as the
structure is read (see [heterogens.md](heterogens.md#how-mdprep-knows)) and
can be built in one of two ways:

- **replaced** by its standard parent, which keeps the backbone and the atoms
  the two share and builds the rest (`structure.repair.replace_modified_residues`,
  see [heterogens.md](heterogens.md#modified-residues)); the modification is
  gone from the simulation;
- **kept** as it is, under `noncanonical_residues`, and bonded into its chain
  by tleap like any amino acid.

Nothing is decided for you. A modified residue that the manifest neither
keeps, replaces nor removes stops the build as the structure is read,
whatever `remove_unknown_heterogens` says: removing it as an unknown
heterogen would cut its chain in two.

The reference case is SpoIIAA phosphorylated on Ser57, PDB 1H4X:
[examples/22_spoiiaa_phosphoserine.yaml](../examples/22_spoiiaa_phosphoserine.yaml)
and [its tutorial](../examples/tutorials/spoiiaa_1h4x/README.md).

## Keeping a residue

tleap builds every residue of a name from one template, so a block is one
residue name, listing every residue of it the build keeps:

```yaml
noncanonical_residues:
- resname: SEP
  residues:
  - {chain: B, resname: SEP, resid: 57}
  parameters: amber_library
  template: SEP
```

`parameters` says where the template comes from: `amber_library`, a template
AmberTools ships for the protein force field, or `am1bcc`, one mdprep
parameterises from the SMILES you give. Where AmberTools has one, use it: its
charges were fitted with the protein force field's own protocol.

A kept residue must sit inside its chain, bonded by a peptide bond to a
residue on each side, and have every heavy atom; mdprep makes no terminal
templates and builds no missing atoms of a residue it has no template for.
It is written with ATOM records, as a residue of its chain -- as a HETATM
after an ATOM residue the PDB writer would have started a new chain, and
tleap would never have bonded it to the residue before it.

## From AmberTools' templates

| Template | Deposited as | Net charge | Library (ff14SB / ff19SB) |
| --- | --- | ---: | --- |
| `SEP` | SEP | -2 | phosaa14SB / phosaa19SB |
| `S1P` | SEP | -1 | phosaa14SB / phosaa19SB |
| `TPO` | TPO | -2 | phosaa14SB / phosaa19SB |
| `T1P` | TPO | -1 | phosaa14SB / phosaa19SB |
| `PTR` | PTR | -2 | phosaa14SB / phosaa19SB |
| `ALY` | ALY | 0 | ff14SB_modAA / ff19SB_modAA |
| `MSE` | MSE | 0 | ff14SB_modAA / ff19SB_modAA |
| `AZF` | 4II (p-azido-L-phenylalanine) | 0 | ff14SB_modAA / ff19SB_modAA |
| `CYF` | 4CF (p-cyano-L-phenylalanine) | 0 | ff14SB_modAA / ff19SB_modAA |
| `HYP` | HYP | 0 | the protein force field itself |

These are AmberTools' own sets, each fitted to the protein force field it
is named after; the AmberTools manual gives their references. The library is
sourced right after the protein force field, `leaprc.phosaa14SB` after
`leaprc.protein.ff14SB`, and the ff19SB versions carry the backbone CMAP each
residue needs.

A template's name is its protonation: SEP is a phosphoserine with its
phosphate dianionic, S1P the monoanion. A residue deposited as SEP and wanted
monoanionic is renamed S1P as the structure is read, and the rename is
reported. tleap builds the hydrogens, as it does a standard residue's. A
template whose heavy atoms are not the residue's, by name, is refused: tleap
matches atoms by name, and would drop the residue's extra atoms and build the
template's missing ones from nothing. That is why phosaa's monoanionic
phosphotyrosine, Y1P, is not offered: it names the phenolic oxygen OG where
a deposited PTR names it OH. The SMILES route builds that state instead.

## From its SMILES

```yaml
noncanonical_residues:
- resname: SEP
  residues:
  - {chain: B, resname: SEP, resid: 57}
  parameters: am1bcc
  smiles: CC(=O)NC(COP(=O)([O-])[O-])C(=O)NC
  net_charge: -2
  parent: SER
```

**The SMILES** is the residue between an acetyl and a methylamide,
`CC(=O)N...C(=O)NC` -- the capped residue amino acids have always been
parameterised as -- or the free amino acid, `N[C@@H](COP(=O)([O-])[O-])C(=O)O`,
whose termini mdprep replaces by the chain's. It is the residue's protonation:
mdprep places the hydrogens it gives, and its formal charge must be
`net_charge`. Bond orders and protonation come from it, stereochemistry from
the structure, as for a ligand. It must have every heavy atom the residue has.

**The caps** are built from the structure itself: the previous residue's CA,
C and O are the acetyl's methyl, carbonyl carbon and oxygen, the next
residue's N and CA the methylamide's. Every bond to the residue keeps the
length and angle the structure gives it.

**Hydrogens** are placed as the structure is normalised, between those caps,
and named as the PDB names an amino acid's (H, HA, HB2, HB3, HG21...); the
structure report counts them. Protonation keeps them, as it keeps a ligand's.

**Charges** are AM1-BCC for the capped molecule (antechamber). The capped
residue's backbone N, H, C and O then take the charges ff14SB and ff19SB give
the backbone of a residue of the same sign -- ff94 fitted one set for neutral
residues, one for ASP and GLU and one for LYS, ARG and HIP, and AmberTools'
phosphoserine uses the anionic one -- the caps are dropped, and the charge
then missing from the residue's integer is spread evenly over its other
atoms:

| Backbone set | N | H | C | O |
| --- | ---: | ---: | ---: | ---: |
| neutral | -0.4157 | 0.2719 | 0.5973 | -0.5679 |
| anionic | -0.5163 | 0.2936 | 0.5366 | -0.5819 |
| cationic | -0.3479 | 0.2747 | 0.7341 | -0.5894 |

The residue report lists each atom's AM1-BCC and final charge, and warns if
the shift per atom is over 0.05 e.

**Atom types** are the protein force field's on the backbone -- N, H, CA (CX
in ff14SB, XC in ff19SB), HA, C, O, and CB with its hydrogens when it is an
sp3 carbon -- and GAFF2's on the rest of the side chain. Every term about the
backbone, its phi and psi torsions and the CA-CB bond included, is then the
protein force field's own, and the peptide bonds to the neighbours are those
of any other residue.

**The parameters joining the two** are parmchk2's. Asked about the capped
residue against GAFF2, parmchk2 also estimates every term of the backbone --
the protein's types are unknown to it there -- and writes GAFF2 values for
them, C-N-CX-C among them; tleap applies an frcmod to every atom of those
types, so loading it would rewrite the backbone of the whole protein. Only the
terms that involve at least one GAFF2 type are kept; the rest are the protein
force field's, already loaded. A term parmchk2 could not estimate ("ATTN")
fails the build unless `validation.fail_on_missing_parameters` is false.

**ff19SB** corrects each residue's backbone with a CMAP chosen by residue
name, so a new name would have none; the residue is given its `parent`'s, as
AmberTools gives SEP Ser's. `parent` is required with ff19SB; the planner
fills it in from the file's MODRES records or mdprep's table of parents.

**The library** is made by tleap from the residue's mol2, with its head on N
and its tail on C, so tleap bonds it into the chain when it reads the
structure; every atom's element is stated, and checked in the library tleap
writes. Its files are under `residues/<NAME>/` in the output directory --
the capped molecule as given to antechamber and as typed, the residue's mol2,
parmchk2's raw frcmod and the one kept, the library, and every command's
output -- and every tleap build of the system loads them.

When several residues share the name, the library is made from the first one
listed and all of them must have the same atoms.

## Protonation

PropKa does not titrate a modified residue, and mdprep does not guess one's
state: it is the template's, or the SMILES's. `protonation.overrides` may not
name a kept residue. The xTB comparison of a histidine's tautomers embeds the
side chain in the surroundings' MM charges; a kept residue contributes its
own -- the template's, each hydrogen's charge given to its heavy atom until
tleap builds them, or those mdprep fitted for it, which is why kept residues
are parameterised right after the structure is normalised.

## What is checked

- As the structure is read: each residue a block names exists, in a kept
  chain, bonded on both sides; a template has the residue's heavy atoms; the
  SMILES matches the residue and its charge.
- In the dry topology and the final one, read back without ParmEd: every kept
  residue is there, under its name and as many times as the manifest lists
  it, carries the charge it was built with, and is bonded by peptide bonds to
  the residues on either side. `reports/validation_report.md` lists them.
- The residue report, `reports/residue_report.md`, records how each was
  built.

## In the planner and the web builder

Each modified residue name is one question -- replace it by its parent, keep
it with AmberTools' template, or keep it parameterised from its SMILES --
offering what applies to it, and nothing is proposed, except replacing a
selenomethionine, which is a label, not chemistry. Kept with a template that
has protonation states, which one is the next question; parameterised, its
net charge and SMILES are. For the common modifications -- phosphorylated
serine, threonine and tyrosine, methylated and acetylated lysines, oxidised
cysteines, carboxylysine -- the template, SMILES and charge are proposed,
with the reason for the protonation chosen, and can be changed.

## Not supported

- A residue at a chain terminus or next to a gap: it has no neighbour to be
  bonded to, and mdprep makes no terminal templates.
- From its SMILES, a residue whose backbone nitrogen carries no hydrogen --
  a proline analogue, an N-methyl amino acid: the backbone charges fixed are
  those of an amide N-H. Hydroxyproline is built from the protein force
  field's own template.
- Residues of one name with different chemistry: tleap holds one template
  per name. Rename one of them in the input.
- A kept residue in the same build as a bonded MCPB metal site (MCPB.py
  builds its own tleap system, which does not load the residue's template), or
  as a QM/MM refinement.
- RESP charges for a parameterised residue; AM1-BCC is the method.
