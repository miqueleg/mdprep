# Ligands

Ligands and cofactors are independent HETATM residues configured under
`ligands:`. mdprep does not infer ligand net charge; the manifest must provide
it.

Supported charge methods:

- `am1bcc`
- `user_mol2`
- `gas_resp_pyscf`
- `qmmesp_pyscf`

For all methods, mdprep validates atom count, atom names, element order,
coordinates, residue identity, and total charge. One molecule configured more
than once, such as the copies of a cofactor in a dimer, is parameterised once
and built under one name; two different molecules are built under two. See
[Names](#names).

## Names

tleap matches a residue to the template of its name, and each atom to the
template atom of its name. So a structure whose names do not say which atom
is which, or which molecule is which, cannot be built as it is. mdprep
renames rather than stopping, and says where each rename happens:

- **Atoms sharing a name.** A molecule editor names atoms by element: Avogadro
  writes every carbon as `C` and every hydrogen as `H`. As the structure is
  read, every atom of a heterogen whose name another atom of it also has gets
  its element and a count (`C1`, `C2`, ... `C9`, `CA`, `CB`), skipping names
  the residue already uses. Atoms whose names were their own keep them.
  Residues with a force-field name (amino acids, nucleotides, water, ions)
  are matched to templates by name, so theirs are not changed. Neither is a
  residue whose like-named atoms sit on one spot: that is one atom written
  twice, and naming the copies apart would build both. The structure
  report's warnings list every rename, and the questions and the manifest
  use the new names.
- **Several molecules under one residue number**, as an editor writes
  everything it holds: a ligand and the water drawn beside it arrive as one
  residue with two of everything. A residue whose atoms fall into separate
  molecules is split, each molecule taking the next free number in its chain,
  before its atoms are named apart.
- **One molecule under two names**, such as a hand-assembled dimer's THC and
  THC2. It is built under one name and parameterised once (`parameters_from`)
  when its answers are the first one's (SMILES, charge, multiplicity, atom
  types and charge method) and its atoms are named and bonded alike. A copy
  of the protein shares its reference's answers when you say the copies are
  alike; see [oligomers.md](oligomers.md#copies-named-apart). Atoms named
  alike but bonded to other neighbours are not one template, whatever the
  SMILES: built from one, the second molecule's atoms would be bonded as the
  first's. Those are parameterised separately.
- **Two different molecules under one name**, or one name answered two ways.
  The second is built under a free name (`LIG`, then `LG1`, `LG2`, ...), or a
  metal under its element and a number.

Residue renames are made by the builders, `mdprep init --interactive` and the
web builder, and recorded in the manifest under `structure.rename_residues`,
each with its reason ([manifest_reference.md](manifest_reference.md#structure)).
Preparation applies them first, as it reads the input, so a residue has one
name from there to the prepared system. The structure report lists them
under "Residues built under another name".

A ligand named like a force-field residue, such as 6NIE's free lysine `LYS`,
is the one rename the web builder still asks about, with a button; see
[tleap_and_validation.md](tleap_and_validation.md).

## Why a SMILES, and what it can and cannot be checked against

A crystal structure usually resolves no ligand hydrogens, and a formal charge
does not supply them. The keto and enol forms of alpha-ketoglutarate are both
C5H4O5(2-) over the same heavy atoms and the same connectivity; they differ in
where two hydrogens sit and in two bond orders. Its monoanion is C5H5O5(-)
whichever carboxylate keeps the proton. Nothing in a PDB file distinguishes
these, so mdprep does not try: `ligands[].smiles` states the protonation and
the bond orders, and the hydrogens are placed to match it.

Three things are checked, and one is not:

- The SMILES must describe the same number of heavy atoms as the residue.
- It must match the residue's heavy-atom connectivity, which is what lets the
  bond orders be transferred onto the coordinates. A connectivity without
  bond orders can match more than one way. Tetrahydrocarbazole's saturated
  ring and its benzene ring are interchangeable until the bond orders are
  known. So every distinct way of laying the SMILES on the residue is tried,
  and the first one that the bond lengths below agree with is kept. A SMILES
  is refused for its bond orders only when no way of laying it on fits.
- Its formal charge must equal `net_charge`. Without this a SMILES one proton
  out is accepted, because it still matches the same heavy atoms -- and the
  hydrogens then describe a different molecule from the charge the rest of
  the run uses.
- The bond orders it implies must be consistent with the distances in the
  structure. This is what catches a wrong tautomer. Formula, charge and
  connectivity cannot tell the keto and enol forms apart, but the coordinates
  can: in 6NIE the C2-O5 of alpha-ketoglutarate is 1.207 A, a ketone, where an
  enol C-OH would refine near 1.36. Delocalised groups -- carboxylate, nitro,
  guanidinium, phosphate, sulfonyl, the core of a porphyrin -- and aromatic
  rings are skipped, because a SMILES has to write their symmetric geometry as
  alternating orders and the formal numbers there say nothing about length.
  A heme's SMILES draws two pyrroles with a single beta-beta bond and two with
  a double one, which two being the writer's choice; 2QFK's heme was refined
  with one of them at 1.52 A, and was refused as a wrong tautomer of heme b,
  which has none. A chlorin's reduced pyrrole is still checked. A single bond
  a pi system reaches across is measured against a conjugated length rather
  than a saturated one: C-C between two sp2 carbons against 1.46 A, and C-N
  from an sp2 carbon to a neutral three-coordinate nitrogen -- an amide, an
  amino group on a ring, the ring nitrogen of NADH -- against 1.36 A. Measured
  against 1.52 and 1.47, every amide, adenine and flavin looked like a wrong
  tautomer. C-O is not relaxed: an enol's C-OH against a ketone's C=O is what
  the check is for.

Note what the geometry check does and does not establish. At typical protein
resolution the ligand's bond lengths come from the refinement restraints
rather than from the density, so they record the tautomer whoever built the
monomer dictionary chose. That is usually right, and it is certainly the
model you are starting from, but it is not independent evidence.

Deriving the bond orders from the coordinates alone is not a substitute for
the SMILES. Heavy atoms carry no record of how many hydrogens were attached,
so a perception algorithm fills the spare valence with bond order instead: on
this same alpha-ketoglutarate, RDKit returns a C-C triple bond where the
molecule has a CH2-CH2.

## A ligand that arrives with hydrogens

A ligand's hydrogens in the input were put there by something: PyMOL's
`h_add`, reduce, PDBFixer, a format converter, a docking program, a riding-model
refinement. Each follows its own rules, and none of them is necessarily the
protonation you meant. PDBFixer takes alpha-ketoglutarate's hydrogens from its
chemical component definition, which is the neutral acid, where at pH 7 it is
a dianion; a docking program usually writes only the polar hydrogens, so a
docked substrate can arrive with two of its twenty-five.

So the SMILES is used whether or not the ligand has hydrogens, and the ones it
has are compared with it, heavy atom by heavy atom -- a formula is not enough,
since the keto and enol forms share one:

- if every heavy atom has the hydrogens the SMILES gives it, they are kept as
  they are, names and positions included;
- if not, they are replaced by the SMILES's, and the structure report lists the
  atoms that disagreed (`LIGAND_HYDROGENS_REPLACED`).

A ligand that arrives with hydrogens therefore needs a SMILES; without one it
is refused as the structure is read, rather than handed to antechamber with
whatever another program left on it. The builder and `mdprep init
--interactive` ask for every ligand's SMILES. Exceptions: a ligand with a
`user_mol2`, which states the molecule itself (its hydrogens are compared with
a SMILES if one is given, but never replaced, since the mol2 lists them), and
a hydrogen-free molecule declared with `expected_formula`.

`keep_input_hydrogens: true` makes a disagreement an error instead -- for
hydrogens placed on purpose. `keep_input_hydrogens: false` replaces them
without comparing.

Once a ligand has its hydrogens, its electrons are counted against its net
charge and multiplicity: an odd count for a closed shell is a molecule that
cannot exist, and stops the run with the counts, before antechamber reports it
as an aside to a failed semi-empirical step.

## Cofactors whose SMILES is proposed

Some cofactors' chemistry is fixed by their name up to a protonation state,
and typing their SMILES in the middle of a form is not something to ask of
anyone. For these the builders propose the SMILES and the net charge at pH 7,
with the protonation it assumes, filled in and changeable:

| Residue | Cofactor | Net charge | Assumes |
| --- | --- | --- | --- |
| HEM | heme b, without its iron | -4 | both propionates ionised |
| FAD | flavin adenine dinucleotide, oxidised | -2 | diphosphate ionised, flavin N3 protonated |
| FMN | flavin mononucleotide, oxidised | -2 | phosphate dianion |
| SAM | S-adenosylmethionine | +1 | sulfonium, zwitterionic amino acid |
| SAH | S-adenosylhomocysteine | 0 | zwitterionic amino acid |
| NAD | NAD+ | -1 | diphosphate ionised |
| NAI | NADH | -2 | diphosphate ionised |
| NAP | NADP+ | -3 | every phosphate ionised |
| NDP | NADPH | -4 | every phosphate ionised |

A residue gets a proposal only when it has exactly the heavy atoms of its
cofactor: one that lost atoms to disorder gets nothing. Each SMILES is tested
against the cofactor as a deposition holds it (`tests/data/cofactors`). Some
depositions of NADH and NADPH refine the nicotinamide ring flat, as if it were
oxidised; the bond-length check then reports it, and which state you simulate
is yours to decide. A reduced flavin -- the FADH- that reacts with O2 in a
flavin-dependent halogenase -- is a different molecule: change the SMILES and
the charge.

## AM1-BCC

Runs AmberTools `antechamber` and `parmchk2`.

## user_mol2

Validates a user mol2 against the extracted ligand. If `user_frcmod` is not
provided, `parmchk2` is required.

## PySCF Charge Methods

`gas_resp_pyscf` performs a gas-phase PySCF single point, evaluates a layered
Merz-Kollman ESP grid, and runs canonical two-stage Amber `respgen`/`resp`.

When any ligand requests `qmmesp_pyscf`, mdprep first generates AM1-BCC
charges for every configured ligand/cofactor and builds one provisional dry
Amber topology. For each QMMESP target, the target ligand is the QM region and
the selected non-target topology atoms are fixed MM point charges. PySCF adds
those charges to the SCF Hamiltonian with `pyscf.qmmm.mm_charge`, then the
polarized target-ligand ESP is fitted with two-stage Amber RESP.

Only target-ligand atoms are RESP centers. MM environment charges are never
fitted or written to a ligand mol2. Final `tleap` builds use final fitted mol2
files. User mol2/frcmod choices remain authoritative for final non-target
ligands even though their provisional embedding charges are AM1-BCC.

Configured substrates/cofactors may be explicit MCPB.py coordinators or
additional model residues. Their final ligand mol2/frcmod files are staged for
MCPB. MCPB RESP step 3 replaces charges for residues in the bonded site, so a
ligand cannot be both a QMMESP target and an MCPB-refitted coordinator or
additional residue.
