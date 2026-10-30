# Porcine osteocalcin, three Gla residues and their calcium (PDB 1Q8H)

The reference case for a **metal on a noncanonical residue**: osteocalcin,
the bone Gla protein, whose gamma-carboxyglutamates hold the calcium ions
it binds hydroxyapatite with.

`1Q8H.pdb` is RCSB entry 1Q8H (Hoang et al., *Nature* 2003), at 2.0 A,
verbatim. Re-download it with:

```bash
curl -O https://files.rcsb.org/download/1Q8H.pdb
```

## What makes it the reference case

- **Three Gla residues in one chain**, A:CGU17, A:CGU21 and A:CGU24,
  declared by `MODRES` records with glutamate as their parent. Each is a
  glutamate the vitamin K-dependent carboxylase has given a second
  carboxylate on CG. Replaced by glutamate it would lose the malonate that
  holds the calcium, so nothing is proposed for the treatment. AmberTools has
  no template for Gla, so keeping one means parameterising it from its
  SMILES. The SMILES and charge are proposed, both carboxylates ionised, net
  -2.
- **Three Ca2+**, each on Gla24. Asp30 holds Ca72 as well, and waters fill
  the rest of each shell. Gla21 is 2.8 A from Ca73, beyond the 2.6 A the
  planner counts as a bond, and Gla17's contacts with calcium are all to the
  copies the crystal's symmetry makes, which are not part of the model. Ca73
  is at half occupancy. Each calcium is proposed nonbonded, as calcium always
  is: a closed-shell ion held by electrostatics, which an ion model
  describes.
- **QMMESP sees the metal.** Fitted with `parameters: qmmesp_pyscf`, a Gla's
  charges are fitted with the calcium ions' +2 among the MM charges around it.
  It is also where that method shows its weak point: the methine carbon
  between the two carboxylates comes out at +1.2 e and its hydrogen at -0.5,
  which mdprep reports as a warning. The example keeps AM1-BCC charges, and
  shows the QMMESP block in its comments.
- **One template for three residues.** tleap builds every residue of a name
  from one template, so the three are parameterised once, from Gla17, the
  first listed.

[`examples/23_osteocalcin_gla_calcium.yaml`](../../23_osteocalcin_gla_calcium.yaml)
is the manifest the decision engine writes for it. See
[docs/noncanonical_residues.md](../../../docs/noncanonical_residues.md#metals-and-kept-residues).
