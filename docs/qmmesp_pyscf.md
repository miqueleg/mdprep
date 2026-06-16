# PySCF QMMESP-Style Charges

`qmmesp_pyscf` mimics the QMMESP chemical idea with PySCF and AmberTools:

1. Build a provisional Amber dry system with the intended force fields and
   temporary ligand charges.
2. Select one ligand as the QM target.
3. Extract non-target atoms as MM point charges with ParmEd.
4. Exclude the target ligand completely from its own MM embedding.
5. Run PySCF on the target ligand only with those MM point charges in the
   Hamiltonian.
6. Generate ESP grid points around the target ligand only.
7. Evaluate the polarized target-ligand QM ESP only.
8. Fit charges only on target ligand atoms.
9. Replace only the target ligand mol2 charges.
10. Rebuild final Amber files with the fitted charges.

The MM environment point charges polarize the target ligand density. They are
not fitted, they are not written into the ligand mol2, and their direct
potential is not added to the RESP/ESP fitting target.

Multiple QMMESP ligands are handled one target at a time. Other ligands can be
included as MM point charges according to the manifest environment settings.
