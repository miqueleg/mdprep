# Structures that arrive with hydrogens

The same iron site, as each program people use to "clean" a structure writes
it, and a deposition refined with riding hydrogens. Each was produced from
the whole structure and cut down to the residues listed below, so the
hydrogens are exactly what the program wrote.

`syrb2_site.pdb` is the deposited SyrB2 iron site (PDB 2FCT, chain A):
His116, Ala118, Thr143, His235, Arg254, alpha-ketoglutarate (AKG 897), the
iron (FE2 900), the bound chloride (CL 901), a second chloride (CL 903) and
four waters, with the deposition's metal CONECT records. The others are that
file after:

| File | Program | Notable |
| --- | --- | --- |
| `syrb2_site.pymol.pdb` | PyMOL `h_add` | ligand hydrogens named H01..., a chloride turned into HCl |
| `syrb2_site.pymol_polar.pdb` | PyMOL `h_add elem N+O` | polar hydrogens only |
| `syrb2_site.converter.pdb` | a format converter, adding hydrogens | hydrogens appended at the end of the file, two named H in each water, HCl again |
| `syrb2_site.reduce.pdb` | reduce `-BUILD` | every added hydrogen has serial 0, CONECT records kept |
| `syrb2_site.pdbfixer.pdb` | PDBFixer `addMissingHydrogens(7.0)` | AKG protonated from its CCD definition, as the neutral acid |

`riding_peptide.pdb` is Pro80-Ala88 of chain A of PDB 4YJ6, an X-ray
structure deposited with its riding hydrogens (Lys84 included, with HZ1-HZ3),
and the waters within 3.2 A of it.

`docked_polar_only.pdb` is lidocaine as a docking program writes a pose: the
heavy atoms and the polar hydrogen, nothing else. The pose is RDKit's (ETKDG
with seed 7, MMFF-optimised), not a docking result.
