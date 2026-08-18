# QMMESP Charges with PySCF

`qmmesp_pyscf` implements the QMMESP workflow with PySCF and AmberTools:

1. Run AM1-BCC for every configured ligand/cofactor, including non-target
   ligands, and preserve every generated command and log.
2. If a user mol2 is configured, retain its atom types and bonded topology but
   place the generated AM1-BCC charges on that scaffold for the provisional
   system. User files remain authoritative for the final non-target model.
3. Build one provisional Amber dry system with `tleap` and preserve its input,
   topology, coordinates, PDB, and log.
4. Select one ligand as the QM target by residue identity, atom names/order,
   and coordinates.
5. Extract non-target atoms as MM point charges with ParmEd and exclude the
   target ligand completely from its own MM embedding.
6. Run an electrostatically embedded QM/MM single point on the target ligand.
   `pyscf.qmmm.mm_charge` places the fixed MM charges from the provisional
   Amber system directly in the SCF Hamiltonian, so they polarize the QM
   density at every SCF iteration.
7. Preserve `pyscf.chk`, SCF input metadata, output, convergence data, the MM
   charge set, and the ESP grid.
8. Evaluate the polarized target-ligand QM ESP with PySCF.
9. Convert that ESP to Amber's exact fixed-width ESP format.
10. Use `respgen` to create standard stage-one/stage-two constraints, including
    atom equivalences, then run Amber `resp` for both stages.
11. Fit and write charges for target-ligand atoms only.
12. Rebuild final Amber files with the final fitted mol2, never the provisional
    mol2.

This is electrostatic QM/MM embedding, not a gas-phase SCF. The QM region is
the target ligand and the MM region is the selected non-target part of the
provisional Amber topology. The MM charges polarize the target-ligand density;
they are not fitted, they are not written into the ligand mol2, and their
direct potential is not added to the RESP/ESP fitting target.

The PySCF energy is the embedded QM single-point energy. mdprep does not
combine it with an MM bonded or Lennard-Jones energy and does not perform a
QM/MM geometry optimization. At this fixed geometry those MM terms do not
polarize the SCF density, but the distinction is recorded in
`pyscf_input.json`, `pyscf_result.json`, and `fit_report.json`.

The reference program constructs this embedding through ASH
`QMMMTheory(..., embedding="elstat")` and an OpenMM MM calculator. mdprep
applies the charge-relevant electrostatic coupling directly in PySCF instead:
the same selected MM charges modify the one-electron Hamiltonian. This avoids
an ASH/OpenMM runtime dependency without turning the calculation into a
gas-phase approximation; the omitted quantity is the combined MM energy, not
the MM polarization of the QM wavefunction used for ESP fitting.

Multiple QMMESP ligands are handled one target at a time. Other ligands can be
included as MM point charges according to the manifest environment settings.

`embedding_cutoff_angstrom: null` includes the complete selected non-target
environment and matches the reference workflow. A finite cutoff is supported
only as an explicit user-selected approximation and is recorded in the
embedding report.

## RESP implementation

The PySCF wavefunction is the source of the ESP. The fitting backend is Amber's
reference two-stage RESP implementation (`respgen` plus `resp`); there is no
in-process RESP-like approximation. Stage 1 uses Amber's standard 0.0005
restraint weight. Stage 2 reads the stage-1 charges, uses the standard 0.001
weight, and applies the constraints generated from molecular connectivity by
`respgen`.

Multiwfn is not required or invoked. There is no Mulliken-charge or other
population-analysis fallback: missing PySCF/AmberTools, SCF failure, malformed
ESP data, RESP non-convergence, atom-order changes, or charge-count mismatches
all fail explicitly.

## Scope of the provisional environment

The provisional QM/MM environment is the dry, normalized structure available
at the ligand stage: protein, retained crystallographic waters/ions, and all
configured independent ligands/cofactors. Solvent and ions requested later by
the manifest's `solvation` block do not yet exist and therefore are not part of
the QMMESP polarization environment. mdprep does not invent an unrelaxed
solvent shell for this single point.

## Current limitations

- The calculation is a fixed-geometry electrostatic-embedding single point;
  it is not a QM/MM optimization, molecular dynamics calculation, or combined
  QM-plus-MM total-energy evaluation.
- The fitted charges describe the one bound conformation supplied in the
  structure. Multi-conformer RESP averaging is not performed.
- Full-environment embedding can be expensive for large systems because every
  selected MM charge contributes to the PySCF one-electron potential. A finite
  cutoff reduces that cost but is an explicit scientific approximation.
- The provisional environment contains the dry normalized structure and
  retained crystallographic solvent/ions only. Manifest-requested bulk
  solvent and salt are added later and cannot polarize this fit.
- MM atoms are fixed point charges. Polarizable MM, Lennard-Jones coupling in
  the SCF Hamiltonian, covalent QM/MM boundaries, and covalent ligands are
  outside v0.2 scope. Configured metals enter through their explicit
  provisional nonbonded charge model. A QMMESP target cannot also be a
  residue whose charges MCPB.py step 3 will refit.
- Every independent ligand/cofactor must be representable by
  AmberTools/GAFF or by an explicit user mol2/frcmod scaffold. Unsupported
  chemistry fails instead of being guessed.
- mdprep does not infer ligand protonation or add missing ligand hydrogens.
  The configured ligand structure must already contain the intended explicit
  atoms and be consistent with its user-supplied net charge and multiplicity.
- PySCF, ParmEd, `antechamber`, `parmchk2`, `tleap`, `respgen`, and `resp` must
  be available for a complete QMMESP preparation. Multiwfn is deliberately
  not used as a fallback.
