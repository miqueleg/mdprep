# Dimers, Trimers and Tetramers

Many enzymes work as oligomers, and many structures hold several copies of
one: a predicted homodimer, or a crystal asymmetric unit with four chains.
mdprep prepares any number of protein chains. When the chains are copies of
one protein, it asks whether every copy should get the same parameters. If
so, it asks about one copy only and applies the answers to the others.

## Which chains

With more than one protein chain, the first question is which chains to
prepare: one chain on its own, or all of them. A predicted oligomer is usually
the assembly you mean to simulate. The chains of a crystal structure are often
copies of a complex that is a monomer in solution, packed together by the
crystal. For any other selection, list the chains under
`structure.keep_chains` by hand.

Ligands, metals and ions go with the chain they belong to. For a crystal
structure that is the chain letter they were deposited under. A prediction
writes every ligand as a chain of its own, so mdprep assigns each such chain
to the protein chain it touches most, counted in atom contacts within 4.5 A.
Keeping chain A of a predicted dimer keeps chain A's iron and cofactor and
leaves out chain B's. Chains that contain only waters are kept either way.

## Copies of one protein

Two protein chains are copies when their sequences align residue for residue
over at least 90% of the shorter chain. Residues the crystal did not resolve
are not in either sequence, so a copy missing a loop or a terminus still
counts. A point mutant also counts. Two different subunits of a complex do
not.

Residue numbers are not used to match a residue in one copy with its
counterpart in another, because copies are often numbered differently. In
SyrB2 (PDB 2FCT), chain A's alpha-ketoglutarate is AKG897 and chain B's is
AKG898. Chain A's iron is FE2900 and chain B's is FE2899. mdprep matches
residues this way instead:

- **Protein residues** are paired by aligning the two sequences, so an offset,
  a gap or a missing terminus does not shift the match.
- **A ligand, ion or bound water** is paired with the residue of the same
  name, with the same atom names, that sits in the same place. The protein
  around it is superposed onto the reference copy first, so a loop that moved
  between copies does not break the match.
- **A metal** is paired only when its whole first shell is paired too, donor
  for donor and atom for atom. **A ligand bound to a metal** is paired only
  when it binds that metal's counterpart the same way.

A residue that cannot be paired is reported, never guessed at. Its questions
are then asked on their own.

## The question

When all chains are kept and some of them are copies, mdprep asks:

> Give every copy of the protein the same parameters?

**Yes** (the default): the questions about ligands, metals and coordinating
side chains are asked for the first copy only. Each answer is then given to
the matching residue of every other copy:

- **A ligand** is parameterised once, in the first copy. The other copies use
  the same mol2 and frcmod. This is also how tleap works: it builds every
  residue of one name from one template.
- **A nonbonded metal** gets the same charge and ion parameter set in every
  copy.
- **A bonded metal site** is fitted once, with MCPB.py, in the first copy.
  Each copy is then built from that fit:
  - every residue of the copy's site takes the template the fit made for its
    counterpart (HD1, HD2, AG1, CL1, WT1, FE1, ...), with that template's
    atom types and charges;
  - the copy's metal is bonded to its own donors;
  - the copy's histidines are linked into its chain the way the fit linked
    chain A's.

  The QM/MM refinement and the quantum calculations cover the fitted copy
  only, so one fit costs the same for a tetramer as for a monomer.
- **The protonation of a coordinating side chain** is copied with its site.
  Every other residue is still protonated on its own by PropKa and xTB, in
  its own environment: an interface makes copies differ there, and those
  differences are real.

**No**: every copy's questions are asked separately. Use this for copies in
different states, for example a substrate bound in one active site and not in
the other.

The [web input builder](https://miqueleg.github.io/mdprep/) and `mdprep init
--interactive` both ask the question.

## When copies differ

The question lists every residue it could not pair, with the reason. Those
residues are asked about on their own whichever way you answer. Two cases from
real structures:

- **BesD (PDB 6NIE), four chains.** In chains B and C, alpha-ketoglutarate
  binds the iron through the carboxylate oxygen named O2; in chains A and D it
  binds through O1. These are the two oxygens of one carboxylate, named the
  other way round. A template matches atoms by name, so B's and C's sites are
  not copies of A's, and are asked about on their own. Swapping the two names
  in the input makes them copies. Chain B also has a second iron on its
  surface, with no counterpart in A.
- **A copy whose metal has lost a water**, or gained one. Its first shell is
  not the reference's, so it is a different site.

## Limits

These are refused with a message that says what to change, before anything
runs:

- **One MCPB fit per preparation.** MCPB.py gives every fit the same template
  and atom-type names, so two separate fits cannot be loaded together. Bonded
  sites in several copies therefore have to share parameters. A copy that
  differs from the reference, or a bonded site in a different subunit of a
  heteromer, has to be modelled nonbonded. Fitting each site separately is
  not supported yet.
- **A bonded site within one chain.** MCPB.py caps each coordinating residue
  with its neighbours in one numbering, which is wrong across a chain
  boundary. A metal bound between two chains has to be modelled nonbonded.
- **One template per residue name.** Two ligands named alike, or two metals,
  can only differ if one of them is renamed in the input. If two of them get
  the same answers, mdprep parameterises them once and says so in the
  manifest. If they get different answers, it refuses.

## In the manifest

A copy is an ordinary ligand or metal-site block, with its own residues. It
says where its parameters come from with `parameters_from`:

```yaml
ligands:
  - id: a_akg_897
    selector: {chain: A, resname: AKG, resid: 897}
    net_charge: -2
    smiles: "[O-]C(=O)CCC(=O)C(=O)[O-]"
    atom_types: gaff2
    charge_method: mcpb_resp_pyscf
  - id: b_akg_898
    selector: {chain: B, resname: AKG, resid: 898}
    net_charge: -2
    smiles: "[O-]C(=O)CCC(=O)C(=O)[O-]"
    atom_types: gaff2
    charge_method: mcpb_resp_pyscf
    parameters_from: a_akg_897
```

Every setting of a copy has to repeat its reference's. The file then reads
correctly, and `mdprep config-check` refuses a copy that has drifted from its
reference. A ligand copy must have the same residue name. A metal-site copy
lists its ions, coordinating ions, bonds and additional residues **in the
reference's order**: its n-th bond is the counterpart of the reference's n-th
bond. That order is how the build knows which residue takes which template,
since residue numbers may differ. The builder writes copies this way, and
names them at the top of the manifest. A ligand whose charges come from a
bonded fit (`mcpb_resp_pyscf`) can only be a copy if its site is a copy of the
reference ligand's site.

See [examples/18_syrb2_dimer_shared_mcpb.yaml](../examples/18_syrb2_dimer_shared_mcpb.yaml)
for both copies of SyrB2, with one bonded site fitted and copied.

## What the reports say

- `metal_report.md` lists each copy of the bonded site under *Copies built
  from this fit*: every residue renamed to a template, from what to what, and
  the number of metal bonds. `metal_report.json` has the bond commands too.
- `ligand_report.json` marks each ligand copy as built with its reference's
  parameters.
- The final topology is validated as usual. Every residue renamed for a copy
  is in the reports.
