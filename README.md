# mdprep

`mdprep` is a Python package for reproducible Amber molecular-dynamics
preparation of standard proteins plus independent ligands and cofactors. A
YAML manifest is the source of truth: structure cleanup, protonation decisions,
ligand charge methods, `tleap` settings, and validation outputs are recorded so
the same preparation can be reviewed and repeated.

## What mdprep Does

mdprep automates the first usable Amber preparation workflow:

- inspects PDB files and reports protein, water, heterogen, ligand, histidine,
  titratable-residue, and disulfide content
- generates starter YAML manifests
- normalizes structures without silently deleting unknown chemistry
- applies manual protonation overrides
- runs PropKa-based pH protonation assignment when requested
- selects neutral HID/HIE histidine tautomers with xTB/GFN2 by default
- supports g-xTB single-point and optimization modes
- assigns disulfide-linked cysteines to `CYX`
- keeps or removes crystal waters according to the manifest
- extracts multiple independent ligands/cofactors
- parameterizes ligands with AM1-BCC, user mol2/frcmod, PySCF gas RESP/ESP, or
  PySCF QMMESP-style embedded charges
- builds final Amber `prmtop`, `inpcrd`, and PDB files with `tleap`
- supports solvation, neutralization, approximate salt-pair addition, and final
  sanity checks

## Supported Workflows In 0.1.0

- PDB input inspection
- starter manifest generation with `mdprep init`
- structure-only normalization
- manual protonation state overrides
- PropKa protonation
- xTB/GFN2 HID/HIE selection for neutral histidines
- g-xTB histidine tautomer ranking, including optimization mode
- disulfide detection and `CYX` assignment
- crystal-water retention/removal
- multiple independent ligands/cofactors
- GAFF/GAFF2 ligand setup with AmberTools
- AM1-BCC ligand charges
- user-provided mol2/frcmod ligand parameters
- PySCF gas-phase RESP/ESP-like ligand charges
- PySCF QMMESP-style environment-polarized ligand charges
- final `tleap` Amber build
- optional solvation, neutralization, and salt
- final validation reports

## Explicitly Unsupported

mdprep 0.1.0 intentionally does not support:

- noncanonical amino acids inside polymer chains
- covalent ligands
- bonded metal-center parameterization
- MCPB-like workflows
- ORCA backend
- Multiwfn backend
- mmCIF input
- automatic loop modeling
- production minimization or MD

Chemically sensitive states are not guessed silently. Unsupported chemistry
should fail with a clear error.

The legacy molecule-conversion toolkit identified by the dedicated ban test is
intentionally prohibited. mdprep uses direct parsers and domain-specific tools
instead of molecule-conversion fallbacks.

## Installation

The recommended path is conda or mamba:

```bash
mamba env create -f environment.yml
conda activate mdprep
pip install -e .
mdprep selftest --quick
```

For development and packaging tools:

```bash
mamba env create -f environment-dev.yml
conda activate mdprep-dev
```

The conda environment includes AmberTools, PropKa, xTB, OpenMM, ParmEd, and
PySCF. A lighter pip-only install can run the pure unit tests, but external
chemistry workflows require the corresponding command-line tools and optional
Python libraries.

## Quick Start

```bash
mdprep inspect input.pdb
mdprep init input.pdb -o system.yaml
mdprep config-check system.yaml
mdprep prepare system.yaml
mdprep validate prepared/final/system.prmtop prepared/final/system.inpcrd
```

Use `mdprep --version` to confirm the installed release.

## YAML Manifest Overview

A compact manifest:

```yaml
# mdprep reference manifest for an ff14SB/TIP3P/GAFF2 protein-substrate system.
#
# System:
#   - Protein force field: ff14SB
#   - Water model: TIP3P
#   - Ligand/substrate: 5NB
#   - Ligand net charge: +1
#   - Ligand atom types: GAFF2
#   - Ligand charges: PySCF QMMESP-like embedded RESP/ESP charges
#   - QM level for 5NB charges: HF/6-31G*
#   - Protonation: PropKa + xTB HID/HIE selection, following the protonation-optimizer logic
#   - Histidine tautomer ranking: xTB/GFN2 single-point, no implicit solvent
#   - Solvation: rectangular water box

project:
  name: 5nb_ff14sb_tip3p_gaff2_qmmesp
  input_structure: input/complex_with_5NB.pdb
  output_dir: prepared_5nb_ff14sb_tip3p_gaff2_qmmesp

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
  method: propka_xtb_his

  # PropKa handles pH-dependent protonation states.
  propka:
    executable: null
    candidate_executables:
      - propka3
      - propka
    pka_margin: 1.0
    keep_output: true

  # Neutral HIS residues are resolved as HID/HIE using xTB,
  # following the protonation-optimizer philosophy.
  #
  # Here we request GFN2 single-point calculations with no implicit solvent.
  histidine:
    neutral_tautomer_method: xtb
    xtb:
      executable: xtb
      model: gfn2
      mode: sp
      opt_level: loose
      solvent: null
      cutoff_angstrom: 5.0
      add_missing_water_hydrogens: true
      water_oh_distance_angstrom: 0.9572
      water_hoh_angle_degrees: 104.52
      extra_args: []
      energy_tie_tolerance_kcal_mol: 0.5
      low_confidence_threshold_kcal_mol: 1.0

  # Manual overrides always win over PropKa/xTB.
  # Add catalytic residues here when the automated pH-based assignment is not chemically correct.
  #
  # Example:
  # overrides:
  #   - selector:
  #       chain: A
  #       resname: ASP
  #       resid: 199
  #       icode: null
  #     state: ASH
  #     reason: "Catalytic acid; force protonated"
  #
  #   - selector:
  #       chain: A
  #       resname: HIS
  #       resid: 164
  #       icode: null
  #     state: HIE
  #     reason: "Catalytic histidine; force epsilon tautomer"
  overrides: []

disulfides:
  auto_detect: true
  detection_cutoff_angstrom: 2.2
  force: []
  forbid: []

ligands:
  - id: substrate_5nb

    # IMPORTANT:
    # Replace chain/resid/icode below with the actual residue identity of 5NB
    # in your input PDB.
    selector:
      chain: A
      resname: "5NB"
      resid: 1
      icode: null

    net_charge: 1
    multiplicity: 1
    atom_types: gaff2
    charge_method: qmmesp_pyscf

    preserve_atom_names: true
    preserve_coordinates: true
    allow_atom_renaming: false
    allow_coordinate_changes: false

    # QMMESP-like PySCF charge derivation:
    #
    # mdprep first builds a provisional Amber system.
    # 5NB is then treated as the QM region.
    # The protein, retained crystal waters, and other allowed non-target atoms
    # are used as MM point charges to polarize the 5NB QM density.
    # The RESP/ESP fit is performed only on the 5NB atoms.
    qmmesp:
      qm_engine: pyscf
      method: HF
      basis: "6-31G*"

      # Defaults to ligand net_charge and multiplicity - 1 if null.
      scf_charge: null
      scf_spin: null

      max_cycle: 100
      conv_tol: 1.0e-9

      embedding_cutoff_angstrom: 12.0

      environment:
        include_protein: true
        include_waters: true
        include_other_ligands: true
        exclude_self_ligand: true

      grid:
        type: connolly
        vdw_scale_factors:
          - 1.4
          - 1.6
          - 1.8
          - 2.0
        points_per_atom_per_shell: 60
        exclude_inside_vdw_scale: 1.2
        max_points: 8000

      resp_fitting:
        backend: native
        total_charge_constraint: true
        restraint: resp
        restraint_a: 0.0005
        restraint_b: 0.1
        max_iter: 25
        convergence: 1.0e-6
        stage_2: true

solvation:
  enabled: true
  box: rectangular
  buffer_angstrom: 12.0
  neutralize: true
  salt_concentration_molar: 0.15
  positive_ion: Na+
  negative_ion: Cl-

validation:
  run_openmm_energy_check: true
  fail_on_warnings: false
  fail_on_missing_parameters: true
  fail_on_noninteger_ligand_charge: true
```

All examples in `examples/*.yaml` are schema-validated by the test suite.

## Stop Stages

Run only part of the workflow when debugging:

```bash
mdprep prepare system.yaml --stop-after structure
mdprep prepare system.yaml --stop-after protonation
mdprep prepare system.yaml --stop-after ligands
mdprep prepare system.yaml --stop-after tleap
```

Without `--stop-after`, `mdprep prepare system.yaml` runs the full supported
workflow through final `tleap` and validation.

## Protonation

Supported protonation modes:

- `manual_only`: applies only user overrides, input-state preservation, and
  disulfide `CYX` assignment.
- `propka`: runs PropKa and assigns pH-dependent states. Neutral HIS residues
  must already be HID/HIE by input state or manual override.
- `propka_xtb_his`: runs PropKa, assigns HIP for protonated HIS, and ranks
  neutral HID/HIE tautomers with xTB.

Manual overrides always win. HIP is assigned by PropKa/pH or explicit manual
override. HID/HIE are selected by xTB only for neutral, non-overridden HIS
residues under `propka_xtb_his`. mdprep does not guess catalytic chemistry.

No hydrogens are added to the final prepared PDB. Input hydrogens can be
removed before the protonation-stage PDB is written, letting Amber assign final
hydrogens later from residue names.

For xTB histidine ranking, retained crystallographic waters in the local xTB
cluster may be oxygen-only. By default mdprep adds deterministic temporary
water hydrogens only to the HID/HIE candidate XYZ files and records them in the
protonation report. Set `protonation.histidine.xtb.add_missing_water_hydrogens:
false` to require pre-hydrogenated cluster waters instead.

## Ligands

Every ligand entry must provide a selector and `net_charge`. Ligands are
processed independently.

Supported charge methods:

- `am1bcc`: AmberTools `antechamber` plus `parmchk2`.
- `user_mol2`: validate user mol2, copy user frcmod or run `parmchk2`.
- `gas_resp_pyscf`: AmberTools for atom types/bonded terms, PySCF gas-phase
  RESP/ESP-like charges.
- `qmmesp_pyscf`: provisional Amber system, PySCF embedded single point, and
  target-ligand-only RESP/ESP-like fitting.

mdprep validates ligand atom count, atom order, atom names, element order,
coordinates, residue identity, and total charge. If a ligand operation changes
identity unexpectedly, it fails unless the manifest explicitly allows that
behavior.

## Correct QMMESP Interpretation

For `qmmesp_pyscf`, mdprep first builds a provisional Amber system. The
selected ligand is treated as the QM region. The surrounding protein, retained
waters, ions, and optionally other ligands are represented as MM point charges.
These MM point charges polarize the ligand QM density during the PySCF
calculation. The RESP/ESP fit is then performed only on the selected ligand
atoms, using the polarized ligand QM electrostatic potential. Environment point
charges are not fitted and are not written into the ligand mol2. The final
Amber system is rebuilt with the fitted ligand charges.

This is different from fitting the total protein-plus-ligand electrostatic
field into ligand charges.

## Output Layout

Typical output:

```text
prepared/
  manifest.input.yaml
  manifest.lock.yaml
  versions.json
  intermediate/
    00_input_normalized.pdb
    01_protonation_assigned.pdb
  ligands/
    <ligand_id>/
      input/
      parameters/
      qm/
  qmmesp/
    provisional_leap/
  leap/
    input/
    dry/
    solvated/
  final/
    system.prmtop
    system.inpcrd
    system.pdb
  reports/
    structure_report.*
    protonation_report.*
    ligand_report.*
    tleap_report.*
    validation_report.*
```

`manifest.lock.yaml` records resolved stage metadata and final output paths.
`versions.json` records mdprep, Python, platform, optional executable versions,
and optional Python package versions where available.

## Troubleshooting

- Unknown heterogens: add them to `ligands:` or set
  `structure.remove_unknown_heterogens: true`.
- Missing ligand net charge: set `ligands[].net_charge` explicitly.
- PropKa not found: install the conda environment or set
  `protonation.propka.executable`.
- xTB not found: install the conda environment or set
  `protonation.histidine.xtb.executable`.
- AmberTools not found: ensure `antechamber`, `parmchk2`, and `tleap` are on
  `PATH`.
- `antechamber` failed: inspect ligand-specific stdout/stderr files under
  `ligands/<id>/parameters/`.
- `tleap` unknown residue: configure the ligand or fix residue names so they
  match loaded mol2 templates.
- Missing atom type/parameter: inspect ligand mol2/frcmod consistency.
- `CYX` without disulfide pair: add the pair under `disulfides.force` or use
  `CYS`/`CYM` if the residue is not disulfide-bonded.
- OpenMM validation warning: inspect `reports/validation_report.json`.
- PySCF SCF did not converge: inspect the ligand `qm/` output and use a
  chemically valid ligand geometry/charge/multiplicity.
- Poor RESP fit: inspect `fit_report.json`; equivalent-atom constraints are
  not implemented in v0.1.
- Ambiguous QMMESP ligand mapping: use unique ligand residue names/selectors.

## Testing

```bash
pytest -q
pytest -q -m "external"
pytest -q -m "external or ambertools or tleap or pyscf or qmmesp"
python -m mdprep.cli config-check examples/*.yaml
python -m mdprep.cli selftest --quick
```

External tests skip cleanly when required tools are unavailable.

## Citation And License

See `CITATION.cff` for citation metadata and `LICENSE` for license terms.
Maintainers should update citation author metadata before archival publication.
