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
coordinates, residue identity, and total charge. Multiple ligands are processed
independently, even if they share a residue name.

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
  bond orders be transferred onto the coordinates.
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
