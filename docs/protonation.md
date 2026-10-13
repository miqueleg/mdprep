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

## How the two tautomers are compared

`protonation.histidine.xtb.embedding` chooses what xTB is actually given.

### `qmmm` (the default)

The QM region is the histidine side chain alone, capped at CA. That is
4-methylimidazole: neutral, closed shell, and the same C5H8N2 for both
tautomers with one proton moved between the ring nitrogens. The charge and
spin handed to xTB are therefore constants, not quantities derived from the
surroundings, and the two energies being compared differ only by the thing
being decided.

Everything within `embedding_cutoff_angstrom` (default 12 A) enters as fixed MM
point charges, read from the ff14SB and TIP3P residue templates that `tleap`
will later use for the same atoms. Residues enter whole: a cutoff that cuts
through one would leave a fragment carrying a charge the residue does not have.
mdprep checks that the embedded charges sum to an integer and says which
residue is responsible when they do not.

A cofactor has no force-field residue template, so a declared ligand gets GFN2
atomic charges from a cached single point, written to
`protonation/histidine_xtb/embedding_cofactor_charges/`. Leaving it out would
put a hole in the field where a charge should be; on BesD, including
alpha-ketoglutarate at 2- changes the answer for two of the six histidines. A
heterogen that is neither declared nor templated is reported by name rather
than silently treated as neutral.

MM charges within `link_exclusion_angstrom` (default 1.8 A) of a capping
hydrogen are redistributed over the rest of their own residue, so the link atom
is not polarised by the atoms it replaced while that residue keeps its charge.
Because both tautomers use the same caps in the same field, whatever the caps
get wrong they get wrong equally, and the difference that decides the tautomer
is unaffected.

### `cluster`

The older treatment: a few hundred atoms are carved out around the histidine
and the whole fragment is given to xTB. That requires stating the fragment's
total charge, which mdprep computes by summing formal per-residue charges -- a
quantity a CA-truncated, hydrogen-capped fragment does not actually carry. On
BesD three of six clusters came out with a charge contradicting their own atom
composition and xTB refused them outright. Use it when the neighbours of a
histidine have no force-field charges to embed, and read the warnings.

The sections below on temporary hydrogens, metal sites and thread counts apply
to the `cluster` backend. Under `qmmm` the metal and its ligands are point
charges rather than part of the QM region, so none of the cluster's charge and
spin bookkeeping arises.

If the input protein is not fully hydrogenated,
`histidine.xtb.add_missing_protein_hydrogens: true` (the default) uses
PDBFixer/OpenMM to create a complete temporary protein environment before the
HID/HIE comparisons. Every original atom identity and coordinate is validated,
the Reference platform and package version are reported, and the temporary
hydrogens are never copied into the protonation-stage or final prepared PDB.
OpenMM initially places these hydrogens stochastically before its short
minimization. mdprep seeds that placement with
`histidine.xtb.temporary_hydrogen_random_seed` (default `20260722`) and records
the seed, so repeated HID/HIE comparisons start from the same environment.
The later `tleap` pass remains the authoritative system hydrogenation. Set the
option to `false` to require a fully hydrogenated input and retain strict
failure on an incomplete xTB environment. If temporary hydrogenation is needed
but PDBFixer/OpenMM is unavailable, protonation fails clearly.

HID and HIE trials run in separate, freshly created work directories. This
prevents xTB restart, optimized-coordinate, and charge files from one tautomer
or an earlier preparation attempt from contaminating another trial. The common
histidine directory retains the candidate XYZ/input files and energy summary;
each tautomer subdirectory retains its complete external-program record.
Optimization failure is fatal and does not trigger an implicit single-point
fallback. Select `mode: sp` explicitly when fixed-geometry scoring is the
reviewed protocol.

If a retained crystallographic water enters the histidine xTB cluster without
hydrogens, mdprep adds deterministic temporary water hydrogens only to the HID
and HIE candidate XYZ files. These temporary atoms are reported and are not
written to `01_protonation_assigned.pdb` or final Amber files. Set
`protonation.histidine.xtb.add_missing_water_hydrogens: false` to preserve the
strict requirement for pre-hydrogenated cluster waters.

## Metal sites in a histidine cluster

The xTB cluster is cut at a fixed distance from the histidine, which does not
respect a metal's coordination sphere. On BesD the cluster around His134 picked
up the iron but not the chloride bonded to it, leaving a fragment whose charge
and electron count were artefacts of where the cut happened to fall; xTB
rejected it as an inconsistent number of unpaired electrons.

A metal is only meaningful together with what it is bonded to, so whenever one
is drawn into a cluster, the residues declared as its coordinators in
`metals[].mcpb.bonds` and `additional_residues` come with it.

The cluster is also not closed shell when it contains a paramagnetic metal.
The unpaired-electron count passed to xTB comes from the multiplicities the
manifest declares, so high-spin Fe(II) contributes four. A metal that appears
in a cluster without a declared multiplicity fails clearly: a spin state
cannot be inferred from an element.

Histidines that coordinate the metal are assigned from their coordination
bonds and never reach xTB at all. It is the ones *near* a metal that this
affects.

## xTB thread usage

Left to its own defaults, xTB opens one OpenMP thread per hardware thread for
every tautomer cluster it is given. A histidine cluster is a few hundred atoms
at most, far below the size at which that parallelism pays for itself, and
mdprep evaluates several clusters concurrently, so an unbounded default
oversubscribes the machine badly: on a 63-thread host a single 257-atom cluster
consumed 7,879 CPU seconds to reach geometry step 13, with no wall-clock
benefit over a serial run.

mdprep therefore pins `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to `protonation.histidine.xtb.num_threads` (default `1`)
around every xTB call, and sets `OMP_STACKSIZE` from
`protonation.histidine.xtb.omp_stacksize` (default `1G`) because xTB segfaults
on small per-thread stacks:

```yaml
protonation:
  histidine:
    xtb:
      num_threads: 1
      omp_stacksize: 1G
```

Raise `num_threads` only for unusually large clusters, and only when clusters
are not already running concurrently. Setting `OMP_NUM_THREADS` in the calling
shell has no effect: mdprep sets it explicitly so that runs are reproducible
and the recorded external-command environment matches what was executed.

Disulfide-linked cysteines are assigned `CYX` from forced or detected SG-SG
pairs unless forbidden by the manifest.

## Metal-bound residues

Explicit bonded-metal coordinators are processed before PropKa/xTB. ND1-bound
histidine is assigned HIE and NE2-bound histidine is assigned HID so the donor
nitrogen is unprotonated. A matching manual override remains authoritative; a
conflicting override and bond fail together rather than silently changing
either input.

For metal-bound ASP/GLU/CYS/LYS/ARG, donor identity alone does not uniquely
determine protonation, so a manual override is mandatory. Metal-bound TYR
ionization is not represented in the current state model and fails clearly.
The pre-MCPB `tleap` hydrogenation pass verifies the final HID/HIE donor
hydrogen pattern. See [Metal centers](metals.md).
