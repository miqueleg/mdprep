# Minimal Amber system

`ace_ala_gly_nme` is ACE-ALA-GLY-NME built in vacuum with ff14SB:

```
source leaprc.protein.ff14SB
pep = sequence { ACE ALA GLY NME }
saveamberparm pep ace_ala_gly_nme.prmtop ace_ala_gly_nme.inpcrd
```

29 atoms, 15 of them hydrogens. Small enough to commit and to minimise in
milliseconds, which is what the force-field hydrogen refinement tests need: a
real Amber topology, without requiring AmberTools to produce one.
