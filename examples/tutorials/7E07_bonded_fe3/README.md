# 7E07 Fe(III), aKG, and proline tutorial

This example prepares the curated `input.pdb` system with:

- protein residues parameterized with ff14SB;
- retained crystallographic waters and TIP3P water;
- free proline substrate `LPR` with AM1-BCC/GAFF2 parameters;
- alpha-ketoglutarate `AKG`, formal charge -2, provisionally parameterized with
  AM1-BCC and finally refitted by electrostatically embedded MCPB RESP;
- high-spin Fe(III), charge +3 and multiplicity 6;
- explicit Fe bonds to AKG O1/O5, HIS 114 NE2, ASP 116 OD1, and HIS 214 NE2;
- a 10 A rectangular solvent box and neutralizing counterions, with no added
  bulk salt.

The original `7E07_prepared.pdb` remains untouched. The curated copy adds the
declared ligand hydrogens and renames the free proline residue from `PRO` to
`LPR`, preventing it from being mistaken for a peptide residue. mdprep repairs
the explicitly declared missing GLN 54 and missing standard-residue heavy
atoms, records every repair/rename, runs PropKa plus GFN2 HID/HIE ranking for
neutral nonmetal histidines, applies the mandatory metal-aware HID states to
HIS 114/HIS 214, and preserves the X-ray waters. The normalized heterogen
inventory is exactly Fe, αKG, proline, and 199 waters; there are no other
heterogens to remove in this input.

Because the X-ray protein lacks hydrogens, PDBFixer/OpenMM first creates a
reported temporary hydrogenated environment for the GFN2 HID/HIE comparisons.
Original atoms must remain identity- and coordinate-identical. These temporary
hydrogens never enter the prepared structure; the provisional and final tLEAP
builds add the authoritative ff14SB/TIP3P hydrogens, including water protons.

## QM charge, spin, and proton audit

The generated 55-atom small model has the following explicit formal-charge
bookkeeping:

| Fragment | Protonation used by MCPB.py | Charge |
| --- | --- | ---: |
| HIS 114 side chain | HID; HD1 present and coordinating NE2 unprotonated | 0 |
| ASP 116 side chain | carboxylate; neither OD oxygen protonated | -1 |
| HIS 214 side chain | HID; HD1 present and coordinating NE2 unprotonated | 0 |
| aKG | `C5H4O5`, both carboxylates deprotonated | -2 |
| Fe(III) | high-spin ferric ion | +3 |
| **Total** |  | **0** |

The resulting small-model formula is `C18H25N4O7Fe`: 243 electrons at charge
0. Multiplicity 6 is passed to PySCF as spin 5, giving 124 alpha and 119 beta
electrons; g-xTB receives the equivalent `--uhf 5`. The capped 92-atom large
model has the same formal-charge sum and multiplicity. ACE/NME caps and the
non-ionized peptide groups are neutral.

There is no crystallographic water or other N/O/S donor within 3.0 Å of Fe in
the curated input. The five declared donors are therefore the complete
first-shell set represented by this test model. These checks establish
internally consistent input bookkeeping; they do not establish that a
particular SCF solution is the chemically correct electronic state.

## Integrated preparation

The manifest now exercises the complete requested path:

1. The protein, both non-protein molecules, retained waters, and provisional
   nonbonded Fe model are prepared with ff14SB/TIP3P and AM1-BCC after PropKa
   plus GFN2 histidine-tautomer selection.
2. ASH relaxes only active-region hydrogens with Fe(III), the full coordinating
   residues, α-ketoglutarate, and proline in the GFN2-xTB QM region. Complete
   non-water contact residues use the 4 Å active cutoff and waters use 8 Å;
   all heavy atoms and non-active hydrogens remain fixed.
   The initial PropKa/xTB states are then reused, and the relaxed proton
   coordinates are validated and preserved through MCPB/QMMESP rather than
   being deleted and rebuilt.
3. MCPB.py builds its capped model from the relaxed geometry.
4. g-xTB evaluates the fixed-geometry small-model Hessian for Seminario.
5. PySCF evaluates the embedded large-model ESP; MCPB.py performs RESP and
   generates the bonded model. mdprep validates and promotes MCPB's fitted AKG
   mol2; the provisional AM1-BCC AKG mol2 is not loaded as a final ligand
   template.
6. mdprep builds, solvates, and validates the final Amber topology with its own
   preserved `tleap` workflow.

Validate the manifest and inspect the relaxed structure first:

```bash
mdprep config-check examples/tutorials/7E07_bonded_fe3/system.yaml
mdprep prepare examples/tutorials/7E07_bonded_fe3/system.yaml \
  --stop-after refinement --overwrite
```

Then run the complete preparation:

```bash
mdprep prepare examples/tutorials/7E07_bonded_fe3/system.yaml --overwrite
```

The large-model QMMMESP-like calculation is a fixed-geometry single point,
electrostatically embedded in point charges from the provisional Amber system.
Environment charges polarize the density but are never RESP fit centers.
Inspect the preserved ASH/xTB/PySCF/MCPB inputs, logs, charge reports, `tleap`
scripts/logs, comparison reports, and final topology validation before using
the system for MD.

An earlier preserved benchmark documents an unstabilized open-shell embedded
B3LYP large-model SCF that oscillated and was stopped. The production manifest
uses 15 coarse-grid ADIIS preconditioning cycles followed by final-grid CDIIS
with a static 0.5 Eh level shift. RI-JK density fitting is explicitly enabled
and pins `def2-universal-jkfit`; the B3LYP/6-31G* orbital model, QM region, MM
embedding charges, and RESP centers are unchanged. Both phases are reported.
A nonconverged final SCF remains a hard failure and never yields RESP charges
or a production topology.

The committed `optimized_small_model.pdb` is retained as a legacy atom-identity
fixture. The integrated manifest does not use it as a production geometry.

## Integrated relaxation and xTB comparison

This site is the reference benchmark for the integrated ASH refinement → xTB
Hessian → MCPB → MD workflow. The controlled candidate/reference protocol is
documented in [the MCPB benchmark guide](../../../docs/mcpb_xtb_benchmark.md).
The GFN2-xTB and g-xTB parameter sets must be generated from the identical
preserved 55-atom small-model PDB; comparing separately optimized small models
would mix geometry and Hessian-method effects. A DFT Hessian reference is
deferred.
