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
AmberTools ships for the protein force field, or one mdprep parameterises
from the SMILES you give -- `am1bcc`, with AM1-BCC charges, or
`qmmesp_pyscf`, with charges fitted to the residue's quantum-mechanical
electrostatic potential where it sits in the protein. Where AmberTools has
one, use it: its charges were fitted with the protein force field's own
protocol.

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

**Its bonds** are read from its coordinates, and the SMILES is matched onto
them, which a strained conformation can defeat. Two atoms bonded to a third
that sit too far apart for a bond -- more than 0.15 A past their covalent
radii -- yet close enough to be read as one are an angle squeezed shut, read
as a three-membered ring the residue does not have. 1H4X's chain A
phosphoserine, in the alternate location the file lists first, has P and CB
both on OG and 2.04 A apart, at 97 degrees. Such a residue is refused from
its SMILES, with the atoms, their distance and their alternate location
named; AmberTools' template, which bonds a residue's atoms by name, builds
it, and so does keeping only the copies that are sound -- or, in the input,
the alternate location that is. Every copy is checked, not only the one the
template is made from: each is given its hydrogens from the SMILES.

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

A residue parameterised from its SMILES may hold only what GAFF2 types and
AM1-BCC charges: H, C, N, O, F, P, S, Cl, Br and I. Anything else is refused
with the atom named. antechamber would not refuse a selenium: it types it as
GAFF2's thioether sulfur, and the residue would be built with sulfur's mass
and van der Waals radius under a selenium's name. A selenomethionine is built
from AmberTools' MSE template, or replaced by methionine. A metal is
[a site of its own](#metals-and-kept-residues).

## QMMESP charges

```yaml
noncanonical_residues:
- resname: SEP
  residues:
  - {chain: B, resname: SEP, resid: 57}
  parameters: qmmesp_pyscf
  smiles: CC(=O)NC(COP(=O)([O-])[O-])C(=O)NC
  net_charge: -2
  parent: SER
  qmmesp:
    method: HF
    basis: 6-31G*
```

The parameterisation of the previous section, with its charges fitted instead
to the electrostatic potential of the residue polarised by the protein around
it: the QMMESP method mdprep uses for ligands
([qmmesp_pyscf.md](qmmesp_pyscf.md)), applied to a residue of the chain.

1. **Provisional parameters.** The residue is parameterised with AM1-BCC as
   the structure is read, as above, into `residues/<NAME>/provisional_am1bcc/`.
   That is the residue the histidine embedding and the provisional system see.
2. **The provisional system.** After protonation, the ligand stage builds the
   dry provisional Amber system QMMESP ligands are fitted in -- protein,
   crystal waters and ions, every ligand with AM1-BCC charges, every metal with
   its provisional nonbonded model, this residue with its provisional library
   -- whether or not any ligand is fitted with QMMESP. One system serves every
   QMMESP fit of the preparation.
3. **The QM region** is the capped residue, the caps built from the
   neighbours' atoms as in the AM1-BCC route. PySCF runs a single point on it
   at `qmmesp.method`/`qmmesp.basis` (HF/6-31G*, the level Amber's protein
   charges were fitted at, is what the planner writes) with the closed-shell
   charge `net_charge`.
4. **The embedding** is every atom of the provisional system as a fixed MM
   point charge in the SCF Hamiltonian (`pyscf.qmmm.mm_charge`), but for
   - the residue itself;
   - the two residues the caps stand in for, whole: a cap's methyl carbon is
     where a neighbour's CA is, and its hydrogens almost where that
     neighbour's N, HA and CB are;
   - the peptide bonds that join those two to the chain beyond -- the C and O
     of the residue before the one before, the N and H of the residue after
     the one after -- a bond's length from the caps' hydrogens, and so as good
     as bonded to the QM region.

   `qmmesp.environment` and `qmmesp.embedding_cutoff_angstrom` select the
   rest as they do for a ligand; a kept residue elsewhere counts as protein.
   An MM atom closer than 1.0 A to a QM atom stops the fit: that is a clash,
   not an environment.
5. **The fit** is Amber's two-stage RESP (`respgen` and `resp`) on the
   polarised ESP, under the constraints of Amber's protocol for a new residue,
   given to respgen as predefined charges and charge groups: the residue's
   backbone N, H, C and O held at the backbone set of its sign (the table
   above) in both stages, and each cap summing to zero. The residue's own
   atoms then carry its integer charge; only theirs are kept. The MM charges
   are not fitted and reach no file but the fit's record of them.
6. **The final library** is built from the provisional one's atom types and
   frcmod, with the fitted charges: charges are all the fit changes. It is
   written to `residues/<NAME>/`, where the final build looks; the provisional
   library is never loaded by the final build, and a build without the fitted
   one stops rather than fall back on it.

Everything is kept under `residues/<NAME>/qm/qmmesp_pyscf/`: the SCF input and
result, the checkpoint, the MM charges and their summary, the ESP grid and
values, and the RESP inputs, outputs and constraints. The residue report
lists both sets of charges side by side, the embedding's size and charge, and
the fit's RMS.

On the stretch of 1H4X's chain B the tests use (residues 52-62), the whole
preparation with SEP57 fitted at HF/6-31G* in 115 MM charges takes half a
minute on four threads, and no charge of the residue moves more than 0.11 e
from the AM1-BCC set. Two things to know:

- **One environment.** tleap has one template per residue name, so the
  charges are fitted in the environment of the first residue a block lists
  and shared by all of them. List first the one whose environment matters
  most.
- **Buried atoms.** RESP fits charges to the potential outside the molecule,
  which an atom buried among its neighbours barely shapes, and a fit to one
  conformation can let such an atom trade charge with the atoms around it.
  On osteocalcin's Gla (PDB 1Q8H) the methine carbon between the two
  carboxylates comes out at +1.2 e and its hydrogen at -0.5 -- in the gas
  phase as well, if less so. Any atom more than 0.5 e from its AM1-BCC charge
  is reported as a warning; compare the two sets before simulating, and keep
  the residue with `am1bcc` if the fitted set does not look like chemistry.

## Protonation

PropKa does not titrate a modified residue, and mdprep does not guess one's
state: it is the template's, or the SMILES's. `protonation.overrides` may not
name a kept residue. The xTB comparison of a histidine's tautomers embeds the
side chain in the surroundings' MM charges; a kept residue contributes its
own -- the template's, each hydrogen's charge given to its heavy atom until
tleap builds them, or those mdprep fitted for it, which is why kept residues
are parameterised right after the structure is normalised. A residue whose
charges QMMESP fits contributes its provisional AM1-BCC ones: its fit needs
the protonated system around it, which comes later.

## Metals and kept residues

**A metal on a kept residue.** A phosphoserine holding a magnesium, a Gla
residue a calcium, a carbamylated lysine the nickel ions of a urease: the
metal is a metal site like any other ([metals.md](metals.md)), and the planner
lists the kept residue among its donors. Modelled nonbonded, it builds: the
residue's charges are its template's or its fitted ones, and the ion's are
its Li/Merz model's. QMMESP is where a metal on the residue matters most --
the ion's formal charge is among the MM charges that polarise the fit -- and
osteocalcin, three Gla residues on three calcium ions, is the reference case:
[examples/23_osteocalcin_gla_calcium.yaml](../examples/23_osteocalcin_gla_calcium.yaml)
and [its tutorial](../examples/tutorials/osteocalcin_1q8h/README.md).

**Bonded, with MCPB.py.** A kept residue on a bonded metal is modelled the
way a histidine on it is. MCPB.py tells an amino acid by its atoms, not its
name, so the residue's side chain joins the small model, and in the large
model its backbone is capped from its neighbours; its atom types and charges
MCPB.py reads from a mol2, as it reads a ligand's. mdprep hands it one for
every kept residue -- the large model reaches the residues beside those on
the metal -- with its frcmod: a residue parameterised from its SMILES has
both under `residues/<NAME>/`, and one built from an AmberTools template has
its unit written out by tleap (`savemol2`) with its family's frcmod
(`frcmod.phosaa14SB`, `frcmod.ff14SBmodAA`...). Hydroxyproline is part of the
protein force field, which MCPB.py reads itself. A residue parameterised from
its SMILES brings GAFF2, and MCPB.py takes one GAFF family per run, so the
site's ligands must use GAFF2 too.

What comes back is the residue on the metal under a name of MCPB.py's -- the
first and third letters of its own and a number, HSK becoming HK1 -- with a
template carrying the metal-bonded atom types and the charges MCPB.py fitted
to the large model, and explicit peptide bonds into the chain, since a
template loaded from a mol2 knows nothing of it. Every other residue of the
name keeps its library: mdprep takes only the renamed templates from
MCPB.py's tleap commands, never a `loadmol2` of the residue's own name. The
topology checks count the renamed residue under its new name and
require its peptide bonds, and report its charge rather than hold it to an
integer: MCPB.py shares charge across the site, as it does for the histidines
on it. The QM/MM refinement a bonded site takes its geometry from treats a
kept residue as protein, on the metal at the net charge its block states.

QMMESP charges for a residue on a bonded metal are refused, with the
residue's question to go back to: MCPB.py refits them, so they would never
reach the build. Kept with AM1-BCC charges, they only seed that fit; a kept
residue beside the site, not on its metal, may still be fitted with QMMESP.

Two things to know. MCPB.py fits the large model's charges with RESP, and a
residue buried in an ionic site can come out of that fit as badly as from
any: on osteocalcin's Gla24 bonded to Ca71, the GFN2-ESP fit gave CB +3.8 e
and CG -3.1. And the Seminario force constants of a calcium or a magnesium
at a crystal geometry are where negative ones are most likely, which mdprep
refuses ([metals.md](metals.md)); on that site the O-Ca-O angles came out
negative. Both are further reasons to model those two ions nonbonded, as the
planner proposes.

**A metal deposited inside a residue.** Some depositions write a metal and the
residue that binds it as one residue: CSB, a cysteine bound to a lead ion,
or CMH, S-(methylmercury)cysteine. The PDB's chemical component dictionary
now marks both obsolete, but older files and models built by hand still
hold them. mdprep reads one as it reads a heme
([metals.md](metals.md#a-metal-inside-a-cofactor-hemes)): a single metal of
the kinds it builds sites for is moved into a residue of its own, named after
its element with the residue's number, and what is left is the modified
residue, to be replaced or kept like any other. Replaced, it is its parent on a metal: with the site
bonded, the parent's protonation is asked, as a cysteine's on a zinc is. The
metal is written after its chain, not after the residue: a one-atom residue
between two residues of a chain is where tleap would end the chain.

A metal bonded to atoms beyond the amino acid -- CMH's mercury holds a methyl
as well as the sulfur -- is an organometallic group: taking the metal out
would leave the residue in pieces, so the residue stays whole, and holding a
metal it cannot be parameterised from a SMILES. Replacing it by its parent,
which drops the metal and what it carried, is all that is offered; that is
also what a heavy-atom derivative made to solve the phases usually calls for.
A residue holding a metal mdprep builds no sites for, such as CSB's lead,
stays whole for the same outcome.

## What is checked

- As the structure is read: each residue a block names exists, in a kept
  chain, bonded on both sides; a template has the residue's heavy atoms; a
  residue kept from its SMILES has no conformation too strained to read its
  bonds from; the SMILES matches the residue and its charge.
- In the dry topology and the final one, read back without ParmEd: every kept
  residue is there, under its name and as many times as the manifest lists
  it, carries the charge it was built with, and is bonded by peptide bonds to
  the residues on either side. One MCPB.py renamed is counted under its new
  name, and its refitted charge reported. `reports/validation_report.md`
  lists them.
- The residue report, `reports/residue_report.md`, records how each was
  built.

## In the planner and the web builder

Each modified residue name is one question -- replace it by its parent, keep
it with AmberTools' template, or keep it parameterised from its SMILES with
AM1-BCC or with QMMESP charges -- offering what applies to it, and nothing is
proposed, except replacing a selenomethionine, which is a label, not
chemistry. Kept with a template that has protonation states, which one is the
next question; parameterised, its net charge and SMILES are, and QMMESP is
written with HF/6-31G*. For the common modifications -- phosphorylated
serine, threonine and tyrosine, methylated and acetylated lysines, oxidised
cysteines, carboxylysine, gamma-carboxyglutamate -- the template, SMILES and
charge are proposed, with the reason for the protonation chosen, and can be
changed. Choosing QMMESP adds PySCF to what the Colab setup installs.

What one copy rules out -- keeping it, when it ends its chain; its SMILES,
when its conformation is strained -- is offered while another choice of
chains allows it, the question saying which, and refused at the question as
soon as the chains kept include that copy. 1H4X, both chains kept as the web
builder keeps them unless told otherwise, is refused its SMILES there, with
chain B on its own and the template as the ways out; nothing is left to fail
in Colab.

A kept residue on a metal is one of the metal's donors like any other; on a
bonded site, the model charge the planner proposes counts it at its net
charge or its template's, and replaced, at its parent's.

## Not supported

- A residue at a chain terminus or next to a gap: it has no neighbour to be
  bonded to, and mdprep makes no terminal templates.
- From its SMILES, a residue whose conformation is too strained to read its
  bonds from (see [From its SMILES](#from-its-smiles)): its template, or a
  sound alternate location, builds it.
- From its SMILES, a residue whose backbone nitrogen carries no hydrogen --
  a proline analogue, an N-methyl amino acid: the backbone charges fixed are
  those of an amide N-H. Hydroxyproline is built from the protein force
  field's own template.
- Residues of one name with different chemistry: tleap holds one template
  per name. Rename one of them in the input.
- QMMESP charges for a residue on a bonded metal, which MCPB.py refits; and
  a kept residue with GAFF2 in a bonded site whose ligands use GAFF.
- A residue holding a metal, a selenium or any element GAFF2 does not type,
  from its SMILES; an organometallic residue at all.
- Gas-phase RESP for a residue, and a QMMESP fit averaged over several
  residues or conformations: one residue's environment is fitted.
