# mdprep

`mdprep` 0.2.0 prepares reproducible Amber molecular-dynamics systems from a
PDB structure and a validated YAML manifest. It supports standard proteins,
independent ligands and cofactors, explicit metal models, optional active-site
QM/MM refinement, final `tleap` assembly, validation, and an optional OpenMM
equilibration/production stage.

The manifest is the source of truth. Chemistry-sensitive decisions—ligand
charge and multiplicity, catalytic protonation, metal oxidation/spin state,
coordinating bonds, and QM components—must be supplied explicitly and are
recorded in the output reports.

## Highlights

- ff14SB or ff19SB proteins with TIP3P or OPC water.
- PropKa protonation, manual catalytic overrides, disulfides, and optional
  GFN2-xTB/g-xTB HID–HIE selection.
- Multiple GAFF or GAFF2 ligands/cofactors with AM1-BCC, user mol2/frcmod,
  PySCF gas-phase RESP, or electrostatically embedded PySCF QMMESP charges.
- Amber nonbonded metal models or bonded MCPB.py models with explicit
  coordinating atoms, charges, and multiplicities.
- MCPB Seminario Hessians from PySCF, GFN2-xTB, g-xTB, or MACE-POLAR-1;
  PySCF supplies the embedded large-model ESP/RESP calculation.
- Optional ASH/OpenMM active-site QM/MM refinement before final
  parameterization.
- Final `prmtop`, `inpcrd`, and PDB generation through `tleap`, followed by
  topology, coordinate, charge, and optional OpenMM energy validation.
- Optional Roe–Brooks OpenMM minimization, heating, density equilibration, and
  manifest-sized production MD with GPU-first/CPU-fallback execution.

## Installation

Conda or Mamba is recommended because the real workflows use AmberTools,
PropKa, xTB, OpenMM, ParmEd, and PySCF:

```bash
git clone https://github.com/miqueleg/mdprep.git
cd mdprep
mamba env create -f environment.yml
conda activate mdprep
mdprep selftest --quick
```

For a lighter Python-only installation, which does not install external
chemistry executables:

```bash
python -m pip install .
```

Optional Python extras are available as `.[md]` for OpenMM and `.[qm]` for
PySCF/ParmEd. External commands are discovered through `PATH`; manifests may
also name a reviewed relative executable. No developer-machine path is embedded
in the distributed examples.

## Basic example

Start from a standard protein PDB:

```bash
mdprep inspect input.pdb
mdprep init input.pdb -o system.yaml
```

A minimal protein-only `system.yaml` is:

```yaml
project:
  name: my_system
  input_structure: input.pdb
  output_dir: prepared/my_system

structure:
  keep_crystal_waters: true
  altloc_policy: highest_occupancy
  remove_unknown_heterogens: false
  preserve_chain_ids: true
  remove_input_hydrogens: true

protein:
  forcefield: ff14SB
  water_model: TIP3P

protonation:
  ph: 7.0
  method: manual_only
  overrides: []

disulfides:
  auto_detect: true
  detection_cutoff_angstrom: 2.2
  force: []
  forbid: []

ligands: []
metals: []

solvation:
  enabled: true
  box: truncated_octahedron
  buffer_angstrom: 10.0
  neutralize: true
  salt_concentration_molar: 0.15
  positive_ion: Na+
  negative_ion: Cl-

validation:
  run_openmm_energy_check: true
  fail_on_warnings: false
  fail_on_missing_parameters: true
  fail_on_noninteger_ligand_charge: true

molecular_dynamics:
  enabled: false
```

The same validated manifest is available as
[`examples/00_basic_protein.yaml`](examples/00_basic_protein.yaml).

Validate and prepare it:

```bash
mdprep config-check system.yaml
mdprep prepare system.yaml
mdprep validate \
  prepared/my_system/final/system.prmtop \
  prepared/my_system/final/system.inpcrd
```

Relative paths are resolved from the manifest directory. Unknown heterogens
are not deleted or guessed: configure them under `ligands` or `metals`, or
explicitly request their removal.

## Common workflows

Generate a starter manifest using PropKa and ff19SB/OPC:

```bash
mdprep init input.pdb -o system.yaml \
  --forcefield ff19SB \
  --water-model OPC \
  --protonation-method propka
```

Stop after a stage while reviewing a complex system:

```bash
mdprep prepare system.yaml --stop-after structure
mdprep prepare system.yaml --stop-after protonation --overwrite
mdprep prepare system.yaml --stop-after refinement --overwrite
mdprep prepare system.yaml --stop-after ligands --overwrite
mdprep prepare system.yaml --stop-after metals --overwrite
mdprep prepare system.yaml --stop-after tleap --overwrite
```

Run the manifest-configured Roe–Brooks protocol from an existing Amber build:

```bash
mdprep run-md system.yaml \
  --prmtop prepared/my_system/final/system.prmtop \
  --inpcrd prepared/my_system/final/system.inpcrd
```

When `molecular_dynamics.enabled: true`, an ordinary `mdprep prepare` continues
through MD. Production length and trajectory/state/checkpoint intervals are
specified in steps in the manifest.

## Where to go next

| Task | Documentation | Example |
| --- | --- | --- |
| Manifest fields | [Manifest reference](docs/manifest_reference.md) | [All examples](examples/README.md) |
| Protonation and histidines | [Protonation](docs/protonation.md) | `examples/02_manual_catalytic_protonation.yaml` |
| Ligands, AM1-BCC, RESP, QMMESP | [Ligands](docs/ligands.md) and [QMMESP](docs/qmmesp_pyscf.md) | `examples/03_multi_ligand_am1bcc.yaml`, `04_qmmesp_pyscf_ligand.yaml` |
| Metal centers and MCPB.py | [Metal centers](docs/metals.md) | `examples/08_nonbonded_zinc_1264.yaml`, `09_bonded_zinc_mcpb_prepare.yaml` |
| Active-site QM/MM refinement | [QM/MM refinement](docs/qmmm_refinement.md) | `examples/10_qmmm_refinement_ash.yaml` |
| Roe–Brooks OpenMM MD | [Molecular dynamics](docs/molecular_dynamics.md) | `examples/15_roe_brooks_openmm.yaml` |
| Complete Fe(III) workflow | [7E07 tutorial](examples/tutorials/7E07_bonded_fe3/README.md) | `examples/tutorials/7E07_bonded_fe3/system.yaml` |

All public example manifests are schema-validated in the test suite.

## Outputs and reproducibility

A complete run preserves:

- the input and locked manifests;
- normalized/protonated/refined intermediate PDBs;
- ligand, QM, MCPB.py, and `tleap` inputs and logs;
- final Amber topology, coordinates, and PDB;
- structure, protonation, ligand, metal, `tleap`, and validation reports;
- executable and Python-package version information;
- optional MD PDBs, density history, trajectory, state log, and restarts.

External failures are reported with command, working directory, return code,
stdout, stderr, and runtime. mdprep does not silently fall back to another
charge model, protonation, force field, QM method, or metal model.

## Current limitations

Version 0.2.0 does not support noncanonical amino acids inside peptide chains,
covalent ligands, mmCIF input, automatic loop modeling, ORCA, Multiwfn,
multiple independent bonded MCPB sites, or multi-chain MCPB protein inputs.
Unsupported chemistry fails explicitly.

## Development and release checks

```bash
mamba env create -f environment-dev.yml
conda activate mdprep-dev
pytest -q
python -m mdprep.cli --help
python -m mdprep.cli config-check examples/*.yaml
python -m mdprep.cli selftest --quick
python -m hatchling build
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [CHANGELOG.md](CHANGELOG.md), and
[CITATION.cff](CITATION.cff). mdprep is distributed under the MIT license.
