# Protonation

Manual researcher overrides always win.

Supported methods:

- `manual_only`
- `propka`
- `propka_xtb_his`

Manual overrides are explicit residue renames within compatible chemical
families, for example ASP/ASH, GLU/GLH, HIS/HID/HIE/HIP, LYS/LYN, and
CYS/CYM/CYX.

PropKa assigns pH-dependent states. If neutral histidine remains unresolved
under `propka`, mdprep fails and asks for either manual HID/HIE assignment or
`propka_xtb_his`.

`propka_xtb_his` ranks neutral HID and HIE with xTB/GFN2 by default. g-xTB is
also supported in single-point or optimization mode. Temporary tautomer
hydrogens used for ranking are written only to local XYZ files and never to the
final prepared PDB.

If a retained crystallographic water enters the histidine xTB cluster without
hydrogens, mdprep adds deterministic temporary water hydrogens only to the HID
and HIE candidate XYZ files. These temporary atoms are reported and are not
written to `01_protonation_assigned.pdb` or final Amber files. Set
`protonation.histidine.xtb.add_missing_water_hydrogens: false` to preserve the
strict requirement for pre-hydrogenated cluster waters.

Disulfide-linked cysteines are assigned `CYX` from forced or detected SG-SG
pairs unless forbidden by the manifest.
