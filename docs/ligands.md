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

## AM1-BCC

Runs AmberTools `antechamber` and `parmchk2`.

## user_mol2

Validates a user mol2 against the extracted ligand. If `user_frcmod` is not
provided, `parmchk2` is required.

## PySCF Charge Methods

AmberTools still provides GAFF/GAFF2 atom types and bonded terms. Provisional
AM1-BCC charges are replaced by PySCF-fitted charges in the final mol2.
