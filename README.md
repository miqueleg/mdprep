# mdprep

`mdprep` 0.4.0 prepares reproducible Amber molecular-dynamics systems from a
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
  coordinating atoms, charges, and multiplicities -- including a metal
  deposited inside a cofactor: a cytochrome P450's heme iron, bonded to the
  porphyrin, the cysteine and a water ([docs/metals.md](docs/metals.md#a-metal-inside-a-cofactor-hemes)).
- Heterogens that are not ligands, read from the file's LINK and MODRES
  records: glycans and crystallisation additives removed on one question
  each, selenomethionine replaced by methionine, and FAD, FMN, SAM and the
  nicotinamides offered with their SMILES and charge
  ([docs/heterogens.md](docs/heterogens.md)). What this means for each family
  of halogenating enzymes is in [docs/enzyme_families.md](docs/enzyme_families.md).
- Dimers, trimers, tetramers: every protein chain in one system. Copies of one
  protein can share parameters -- asked about once, matched residue for
  residue by sequence and position rather than number, and a bonded metal site
  fitted once and reused in every copy ([docs/oligomers.md](docs/oligomers.md)).
- MCPB Seminario Hessians from PySCF, GFN2-xTB, g-xTB, or MACE-POLAR-1;
  PySCF supplies the embedded large-model ESP/RESP calculation.
- Optional ASH/OpenMM active-site QM/MM refinement before final
  parameterization.
- Final `prmtop`, `inpcrd`, and PDB generation through `tleap`, followed by
  topology, coordinate, charge, and optional OpenMM energy validation.
- Optional Roe–Brooks OpenMM minimization, heating, density equilibration, and
  manifest-sized production MD with GPU-first/CPU-fallback execution.
- Predicted models: the closest X-ray structure from the RCSB, and its crystal
  waters transplanted onto the model with PyMOL -- a water on the metal moves
  with it.
- An official Colab notebook that prepares, equilibrates, and shows every step:
  charges, metal-site parameters, the equilibration, the trajectory.

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
mdprep plan input.pdb                           # what must you decide?
mdprep init input.pdb --interactive -o system.yaml
```

`mdprep plan` lists only the manifest decisions this structure actually
requires, and `--interactive` asks them and writes a validated manifest. See
[Guided manifest generation](docs/guided_manifests.md). The non-interactive
`mdprep init input.pdb -o system.yaml` still writes the commented starter
manifest.

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

For a guided workflow, use the [web input builder](https://miqueleg.github.io/mdprep/):
it shows the structure in 3D, asks only the questions your structure needs, and
writes a validated manifest and a bundle. With no structure of your own, it
loads SyrB2 (PDB 2FCT) in one click. Then open the
[official Colab notebook](https://colab.research.google.com/github/miqueleg/mdprep/blob/master/examples/mdprep_colab.ipynb)
([`examples/mdprep_colab.ipynb`](examples/mdprep_colab.ipynb)): it installs
the programs mdprep runs while you are still in the builder, and with the
bundle uploaded it prepares the system, equilibrates it, and lets you
investigate every step: the structure at each stage, the protonation states,
the charge on every atom, the metal site's force constants, how the box
equilibrated, and the MD trajectory.

No crystal structure of your protein? Predict the complex with its cofactors
(Boltz or AlphaFold3 in ColabFold), then let mdprep find the closest X-ray
structure and bring its crystal waters across -- including a water bound to
the metal, which a prediction never has:

```bash
mdprep closest-xray model.cif
mdprep transplant-waters model.cif --reference 2FCT
```

See [Predicted structures](docs/predicted_structures.md).

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
| Generating a manifest | [Guided manifest generation](docs/guided_manifests.md) | `mdprep plan`, `mdprep init --interactive` |
| Generating one in a browser | [Web input builder](web/README.md) | `python web/build.py` |
| Manifest fields | [Manifest reference](docs/manifest_reference.md) | [All examples](examples/README.md) |
| Protonation and histidines | [Protonation](docs/protonation.md) | `examples/02_manual_catalytic_protonation.yaml` |
| Ligands, AM1-BCC, RESP, QMMESP | [Ligands](docs/ligands.md) and [QMMESP](docs/qmmesp_pyscf.md) | `examples/03_multi_ligand_am1bcc.yaml`, `04_qmmesp_pyscf_ligand.yaml` |
| Metal centers and MCPB.py | [Metal centers](docs/metals.md) | `examples/08_nonbonded_zinc_1264.yaml`, `09_bonded_zinc_mcpb_prepare.yaml` |
| Active-site QM/MM refinement | [QM/MM refinement](docs/qmmm_refinement.md) | `examples/10_qmmm_refinement_ash.yaml` |
| Roe–Brooks OpenMM MD | [Molecular dynamics](docs/molecular_dynamics.md) | `examples/15_roe_brooks_openmm.yaml` |
| Predicted models: closest X-ray, crystal waters | [Predicted structures](docs/predicted_structures.md) | `mdprep closest-xray`, `mdprep transplant-waters` |
| A water bound to a metal | [Metal centers](docs/metals.md#waters-bound-in-a-metal-site) | `examples/17_syrb2_bound_water_mcpb.yaml` |
| A heme: cytochrome P450 | [Metal centers](docs/metals.md#a-metal-inside-a-cofactor-hemes) | `examples/19_cyp119_heme_mcpb.yaml` |
| Glycans, additives, modified residues: a peroxygenase | [Heterogens](docs/heterogens.md) | `examples/20_upo_5oxu_heme_glycans.yaml` |
| A flavin-dependent halogenase | [Enzyme families](docs/enzyme_families.md) | `examples/21_prna_fad_halogenase.yaml` |
| Preparing, running and investigating in Colab | [Colab example notes](examples/README.md) | `examples/mdprep_colab.ipynb` |
| Guided Colab builder (legacy) | [Colab example notes](examples/README.md) | `examples/mdprep_interactive_colab.ipynb` |
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

mdprep does not support noncanonical amino acids inside peptide chains
(a modified residue with a standard parent can be replaced by it), covalent
ligands (heme c and the bonded hemes of peroxidases among them; glycans and
other bonded heterogens can only be removed), metal clusters, oxoanions such
as vanadate, automatic loop modeling (a missing loop is left as two
termini), a metal-bound hydroxide, ORCA, Multiwfn, more than one independently
fitted bonded MCPB site (copies of one site in an oligomer share a single
fit), or a bonded MCPB site bound by two chains. Unsupported chemistry fails
explicitly.

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
