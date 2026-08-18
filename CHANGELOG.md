# Changelog

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
