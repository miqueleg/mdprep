# Changelog

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
