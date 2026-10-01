# Changelog

## 0.2.1 - 2026-10-01

### Fixed

- Roe--Brooks stage 2 no longer fails with "Particle coordinate is NaN" on
  solvated systems. Free ions were positionally restrained because Amber names
  them `Na+`/`Cl-`, which match no bare element symbol, and coordinates handed
  between stages were wrapped into the periodic box while the restraint
  references stayed unwrapped, placing every wrapped restrained particle a full
  box length from its reference. Stage hand-off and XML restarts now stay in
  the unwrapped frame; only the per-step PDBs and the trajectory are wrapped.
- Final-output validation no longer rejects its own PDB with "PDB atom serials
  must be unique when present" above 99,999 atoms. Serials are decoded for the
  column-12 overflow `tleap`/`ambpdb` write, for hybrid-36, and for `*****`; a
  genuine wrap-around is renumbered in file order with a warning, and stays
  fatal only when `CONECT` records make the connectivity ambiguous. The PDB
  writer uses the same convention it reads, so structures of that size
  round-trip unchanged.
- `mdprep inspect --json` no longer emits unparseable JSON on a narrow
  terminal, where rich reflowed long strings to the console width.

### Changed

- xTB histidine calculations pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` to the new `protonation.histidine.xtb.num_threads`
  (default `1`), with `OMP_STACKSIZE` from `omp_stacksize` (default `1G`).
  xTB otherwise opened one thread per hardware thread for every tautomer
  cluster, spending large amounts of CPU time without reducing wall time.
- `mdprep inspect` also reports free monatomic metal ions.

## 0.2.0 - 2026-08-17

### Added

- Explicit nonbonded and bonded MCPB.py metal-center workflows, including
  PySCF QMMESP/RESP charges and PySCF, GFN2-xTB, g-xTB, or MACE-POLAR-1
  Hessian sources for Seminario parameters.
- Optional ASH active-site QM/MM refinement with explicit QM components,
  coordinating residues, 4 A residue and 8 A water active regions, and
  hydrogen-only relaxation.
- Manifest-driven Roe--Brooks OpenMM minimization, heating, density
  equilibration, and production MD with accelerator-first/CPU-fallback
  platform selection.
- User-controlled production step and reporting/checkpoint intervals.
- Manifest-directory-relative input paths and portable executable discovery.
- GAFF/GAFF2, ff14SB/ff19SB, TIP3P/OPC, AM1-BCC, gas RESP, and embedded
  QMMESP examples covering the supported preparation combinations.

### Fixed

- Roe--Brooks positional restraints are installed before Context creation and
  their reference coordinates are converted explicitly from input units to nm.
- Constraint-compatible heating handoff, evolving NPT box propagation, and
  production progress totals.
- MCPB/tLEaP residue-index and renamed-residue coordinate validation.

## 0.1.0 - 2026-06-15

Initial usable release.

### Added

- YAML manifest validation with pydantic.
- PDB inspection, residue/atom selector parsing, altloc handling, water and heterogen classification, histidine/titratable residue detection, and possible disulfide detection.
- `mdprep init` starter-manifest generation and safe structure normalization.
- Manual protonation overrides, input-state preservation, disulfide `CYX` assignment, and input-hydrogen removal.
- PropKa-based residue-state assignment and xTB/GFN2 or g-xTB HID/HIE histidine tautomer selection.
- Ligand extraction and AmberTools parameter generation for `am1bcc` and `user_mol2`.
- PySCF gas-phase RESP/ESP-like ligand charges.
- PySCF QMMESP-like environment-polarized ligand charges with explicit target-ligand-only fitting.
- Final `tleap` build, optional solvation, neutralization/salt handling, final Amber files, and validation reports.
- External tests that skip cleanly when optional executables or libraries are unavailable.

### Unsupported In 0.1.0

- Noncanonical amino acids inside polymer chains.
- Covalent ligands.
- Bonded metal-center parameterization and MCPB-like workflows.
- ORCA and Multiwfn backends.
- mmCIF input.
- Automatic loop modeling.
- Production minimization or MD.
