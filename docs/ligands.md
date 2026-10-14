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
- **The tautomer is not checked and cannot be.** Two tautomers share a
  formula, a charge and a connectivity, so the SMILES is the only thing that
  says which one you meant. That is the judgement the question is asking you
  to make.

A ligand that already carries hydrogens is left alone, and a SMILES is neither
asked for nor applied to it.

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
