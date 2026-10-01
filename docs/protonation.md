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
