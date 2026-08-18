# Manifest Reference

Top-level keys:

- `project`
- `structure`
- `protein`
- `protonation`
- `disulfides`
- `ligands`
- `metals`
- `refinement`
- `solvation`
- `validation`
- `molecular_dynamics`

## project

- `name`: project identifier.
- `input_structure`: PDB input path.
- `output_dir`: mdprep output directory.

Relative file and output paths are resolved from the directory containing the
manifest, not from an installation-specific home directory. For compatibility
with older manifests, an input file that is absent beside the manifest but
already exists relative to the launch directory is accepted and recorded by
its resolved path. Plain executable names such as `xtb`, `propka3`, `MCPB.py`,
and `python` are discovered through `PATH`; users may still supply a reviewed
relative executable path when needed.

## structure

- `keep_crystal_waters`: keep waters in the input PDB.
- `altloc_policy`: `highest_occupancy`, `first`, or `fail`.
- `remove_unknown_heterogens`: remove unconfigured heterogens instead of
  failing.
- `preserve_chain_ids`: preserve chain IDs when possible.
- `remove_input_hydrogens`: remove input hydrogens before protonation-stage PDB
  output.

## protein

- `forcefield`: `ff14SB` or `ff19SB`.
- `water_model`: `TIP3P` or `OPC`.

## protonation

- `ph`: target pH.
- `method`: `manual_only`, `propka`, or `propka_xtb_his`.
- `overrides`: manual residue-state assignments.
- `histidine.xtb`: xTB/g-xTB settings for HID/HIE ranking.
  - `add_missing_protein_hydrogens`: use a validated, temporary
    PDBFixer/OpenMM hydrogenated environment when the input protein is
    incomplete (default `true`). These hydrogens never enter the prepared PDB.
  - `temporary_hydrogen_random_seed`: deterministic seed for OpenMM's initial
    temporary-protein-hydrogen placement (default `20260722`).
  - `add_missing_water_hydrogens`: add temporary xTB-only hydrogens to
    oxygen-only cluster waters.
  - `water_oh_distance_angstrom`: O-H distance for temporary cluster waters.
  - `water_hoh_angle_degrees`: H-O-H angle for temporary cluster waters.
  - `scf_iterations`: xTB SCC/SCF iteration limit for histidine tautomer jobs.
  - `electronic_temperature_kelvin`: xTB electronic temperature passed as
    `--etemp`; default is `1000.0`, set null to omit the option.

## disulfides

- `auto_detect`: detect close CYS/CYX SG-SG pairs.
- `detection_cutoff_angstrom`: SG-SG cutoff.
- `force`: manually forced disulfide pairs.
- `forbid`: pairs that must not be auto-assigned.

## ligands

Each ligand has:

- `id`
- `selector`
- `net_charge`
- `multiplicity`
- `atom_types`: `gaff` or `gaff2`
- `charge_method`: `am1bcc`, `user_mol2`, `gas_resp_pyscf`,
  `qmmesp_pyscf`, or `mcpb_resp_pyscf` for a ligand included in a bonded
  metal-site RESP fit. The latter uses AM1-BCC only provisionally and promotes
  MCPB's joint-RESP residue mol2 for the final tLEAP build.
- optional `user_mol2`
- optional `user_frcmod`
- preservation controls for names and coordinates
- optional `qmmesp` block for PySCF charge workflows

## qmmesp

- `qm_engine`: must be `pyscf`.
- `method`: `HF` or a PySCF DFT functional string.
- `basis`: PySCF basis.
- `embedding_cutoff_angstrom`: optional MM point-charge cutoff for QMMESP;
  `null` (default) uses the complete selected provisional environment.
- `num_threads` and `max_memory_mb`: PySCF resource limits.
- `grid`: deterministic layered Merz-Kollman ESP grid. The defaults use vdW
  shells at 1.4, 1.6, 1.8, and 2.0, a surface density of 1 point/A^2, and
  remove points inside 1.4 times another atom's vdW radius. `max_points` is a
  hard safety limit (maximum 99,999 for Amber's `2I5` header); mdprep fails
  instead of silently thinning the requested surface.
- `resp_fitting`: canonical AmberTools two-stage RESP (`backend: ambertools`,
  `stage_2: true`).
- `environment`: include/exclude protein, waters, and other ligands.

The target ligand is always excluded from its own MM embedding.

## metals

Each site has:

- `id`: unique site identifier.
- `model`: `nonbonded` or `bonded_mcpb`.
- `ions`: one or more explicitly selected one-atom metal residues.
  - `selector`: exact chain, residue name/number/insertion code, and atom name.
  - `element`: case-sensitive chemical symbol such as `Zn` or `Fe`.
  - `charge`: explicit +1 through +4 formal ion charge.
  - `multiplicity`: optional for ordinary metal parameterization, but mandatory
    when the ion's site is selected for QM/MM refinement.

For `model: nonbonded`, `nonbonded.parameter_set` is mandatory and must be
`12_6`, `cm`, `hfe`, `iod`, or `12_6_4`. Availability is checked against the
active AmberTools data for the configured TIP3P or OPC water model.

For `model: bonded_mcpb`, the `mcpb` block contains:

- `executable`: MCPB.py executable name or path.
- `workflow`: `prepare_inputs`, `complete`, or `pyscf`.
- `provisional_nonbonded_parameter_set`: explicit Amber ion family used for
  the pre-MCPB hydrogenation and any earlier QMMESP environment.
- `bonds`: exact metal atom/coordinator atom pairs. Every donor candidate that
  MCPB.py would discover inside the cutoff must be declared.
- `additional_residues`: reviewed residues to include even without a declared
  metal bond.
- `cutoff_angstrom`: MCPB donor search cutoff.
- `force_constant_method`: `seminario`, `modified_seminario`, `empirical`, or
  `z_matrix`.
- `charge_restraint`: `all_ligating`, `backbone_heavy`, `backbone_all`, or
  `backbone_and_cb`.
- `software_version`: `g03`, `g09`, `g16`, `gau`, or `gms`.
- `small_model_charge`, `small_model_spin`, `large_model_charge`, and
  `large_model_spin`: explicit QM charge and multiplicity for each generated
  model.
- `scale_factor` and `large_opt`: MCPB.py force-constant/large-model options.
- `artifacts`: forbidden during `prepare_inputs`; mandatory during `complete`.
  `large_mk_log` is always required. Gaussian Seminario methods require
  `small_opt_fchk`; GAMESS Seminario methods require `small_fc_log`; Z-matrix
  requires a Gaussian `small_fc_log`.
- `pyscf`: mandatory only for `workflow: pyscf`. `geometry_source` is either
  `external_pdb` or `qmmm_refinement`. The external form requires
  `optimized_small_model_pdb` and `small_model_geometry_status`; mdprep
  validates an exact coordinate-only atom-identity/order replacement.
  `qmmm_refinement` requires a completed refinement in the same preparation and
  the bonded site listed in `refinement.qm_components.metal_sites`.
  `method`, `basis`, and the SCF/resource/embedding/grid settings configure the
  PySCF large-model ESP calculation. Open-shell SCF behavior is explicit and
  reproducible through `scf_algorithm` (`diis`, `newton`, or the explicit
  `adiis_then_diis` and `adiis_then_newton` preconditioned protocols), `initial_guess`
  (`minao`, `atom`, or `huckel`), `level_shift_mode` (`static` or DIIS-only
  `dynamic`), non-negative `level_shift_hartree`,
  `damping_factor` in `[0, 1)`, and `diis_space` of at least 2. Defaults retain
  ordinary DIIS with a MINAO guess and no level shift or damping. mdprep never
  changes these controls or falls back to another QM method after failed SCF
  convergence. `adiis_precondition_cycles` controls the ADIIS phase; optional
  `adiis_precondition_dft_grid_level` uses a reviewed coarse DFT grid only for
  that phase before rebuilding the requested final grid for CDIIS or Newton.
  A reviewed large-model density restart is explicit through the paired
  `large_model_restart_checkpoint` and
  `large_model_restart_checkpoint_sha256` fields and requires
  `scf_algorithm: newton`. mdprep verifies the checksum and exact checkpoint
  atom elements/order, charge, spin, and basis dimensions before loading the
  density. Coordinates must be identical by default. A non-negative explicit
  `large_model_restart_max_displacement_angstrom` permits a nearby geometry;
  mdprep then projects the density onto the current basis and records maximum
  and RMS displacement. It never searches for or selects a checkpoint
  automatically.
  `large_model_density_fitting` explicitly enables PySCF RI-JK
  density fitting for the large-model ESP calculation only; an optional
  `large_model_auxbasis` pins its auxiliary basis. The default is the exact
  four-center SCF (`false`). `hessian_backend` is `pyscf` by default.
  `hessian_backend: xtb` requires an `xtb` block with `model: gfn2` or `gxtb`,
  `executable`, optional `release_tag` and `expected_executable_sha256`,
  positive `accuracy` (default `0.001`), `num_threads`, and optional positive
  `max_scf_iterations`. The latter is passed explicitly to every displaced
  xTB Hessian calculation; omit it to retain the executable default. The legacy
  `hessian_backend: gxtb` plus `gxtb:` block remains valid. xTB replaces only
  the small-model Hessian. `hessian_backend: mace_polar1` requires a
  `mace_polar1` block containing the isolated `python_executable`, model
  (`polar-1-s`, `polar-1-m`, or `polar-1-l`), device, thread count, optional
  checkpoint SHA-256, positive finite-difference step and error limit, and
  explicit `accept_model_license: true`. MACE runs in float64 and its
  analytical Hessian is checked against central finite differences of its
  forces. PySCF still supplies the large-model ESP for every non-PySCF Hessian
  backend. This workflow supports Hessian-based `seminario` or
  `modified_seminario`,
  `software_version: gau`, and `large_opt: 0`.
- `parameter_comparison`: optional reference comparison of MCPB-created
  BOND and ANGL Seminario terms. It requires `reference_frcmod` and accepts
  candidate/reference labels, `require_exact_term_set`, explicit bond/angle
  relative-force-constant RMSE limits, equilibrium-distance/angle RMSE limits,
  and `fail_on_thresholds`. Null thresholds report metrics without inventing an
  acceptance criterion. Relative limits are fractions (`0.20` means 20%).
  Non-positive candidate or reference force constants are explicit violations.

See [Metal centers](metals.md) for staging, protonation behavior, QMMESP
interactions, OpenMM 12-6-4 behavior, and current limits.

## refinement

- `enabled`: enable the optional pre-parameterization geometry refinement.
- `backend`: `ash`.
- `qm_method`: `gfn2_xtb`, `gxtb`, or `mace_polar1`.
- `embedding`: `electrostatic` for `gfn2_xtb`; `gxtb` and `mace_polar1`
  require the explicit value `mechanical` because neither backend receives MM
  point charges in this workflow.
- `qm_components.ligands`: explicit ligand/cofactor ids in the QM region.
- `qm_components.metal_sites`: explicit metal-site ids in the QM region.
- `qm_components.metal_coordinating_residues`: complete coordinating protein
  residues, keyed by selected metal-site id. This is mandatory for nonbonded
  metal sites; bonded MCPB sites also use their explicit `mcpb.bonds`.
- `active_region_cutoff_angstrom`: fixed at `4.0` for this protocol.
- `active_water_cutoff_angstrom`: fixed at `8.0`; this is evaluated separately
  from the non-water active-region cutoff.
- `include_contact_waters`: fixed at `true`; retained contact waters are
  protonated by the provisional Amber build and included as complete active MM
  residues.
- `movable_atoms`: `all_active_region` (default) optimizes every physical atom
  in the selected active residues. `active_region_hydrogens` retains the same
  full QM and active-residue definitions but passes only their hydrogen atoms
  to the optimizer, exactly freezing all heavy atoms. This is intended for
  proton relaxation on a trusted experimental heavy-atom structure.
- `post_refinement_protonation`: `rerun` (legacy default) or `reuse_initial`.
  Hydrogen-only refinement requires `reuse_initial`; mdprep reapplies the
  reviewed initial states without rerunning PropKa/xTB and validates/preserves
  every optimized protein/water hydrogen for later parameterization.
- `total_qm_multiplicity`: optional total spin multiplicity. It becomes
  mandatory when more than one selected component is open-shell.
- `ash.python_executable`: Python interpreter in an environment containing ASH,
  OpenMM, and geomeTRIC.
- `ash.xtb_executable`: xTB executable name/path.
- `ash.allow_unusual_link_boundaries`: explicit opt-in for ASH QM/MM link atoms
  across boundaries other than its usual C-C case. This is normally required
  when complete metal-coordinating amino-acid residues are in the QM region,
  because their peptide connections create C-N boundaries. It defaults to
  `false`; enable it only after reviewing the generated QM selection report.
- `ash.max_iterations`, `ash.num_cores`, `ash.platform`,
  `ash.xtb_max_iterations`, `ash.electronic_temperature_kelvin`, and
  `ash.accuracy`: optimizer, OpenMM, and xTB runtime controls.
- `mace_polar1`: required only for `qm_method: mace_polar1`. It pins the
  `polar-1-s`, `polar-1-m`, or `polar-1-l` checkpoint by path and mandatory
  SHA-256, selects `cpu`, `cuda`, or `mps`, accepts optional reviewed Python
  dependency search paths, and requires explicit model-license acceptance.

Selected ligands and metal ions must explicitly supply charge and multiplicity.
The total QM charge is derived from those values plus the formal charges of
protonated coordinating protein residues. See
[Active-site QM/MM refinement](qmmm_refinement.md) for region rules, staging,
files, and failure behavior.

## solvation

- `enabled`
- `box`: `truncated_octahedron` or `rectangular`
- `buffer_angstrom`
- `neutralize`
- `salt_concentration_molar`
- `positive_ion`
- `negative_ion`

## molecular_dynamics

- `enabled`: run the OpenMM MD stage after final `tleap` validation.
- `protocol`: currently `roe_brooks_2020`.
- `temperature_kelvin` and `pressure_atmosphere`: target thermodynamic state.
- `platform`: `auto`, `CUDA`, `HIP`, `Metal`, `OpenCL`, `CPU`, or `Reference`.
  `auto` tests accelerators first and falls back to CPU when necessary.
- `device_index`: optional accelerator device index.
- `precision`: `single`, `mixed`, or `double` when supported by the selected
  platform.
- `cpu_threads`: optional OpenMM CPU thread count.
- `random_seed`: deterministic OpenMM velocity, thermostat, and barostat seed.
- `step08_ensemble`: `NPT` or the stability-oriented `NVT` alternative.
- `density`: Step-10 sampling interval, 300-ps-style window, slope and
  half-window mean-difference thresholds, minimum points, and hard maximum
  duration.
- `production.steps`: mandatory positive production MD step count.
- `production.timestep_fs`: fixed at `2.0` for this protocol.
- `production.trajectory_interval_steps`, `state_interval_steps`, and
  `checkpoint_interval_steps`: positive reporting intervals no longer than the
  production run.

See [Roe--Brooks OpenMM MD](molecular_dynamics.md) and
`examples/15_roe_brooks_openmm.yaml`.

## validation

- `run_openmm_energy_check`
- `fail_on_warnings`
- `fail_on_missing_parameters`
- `fail_on_noninteger_ligand_charge`
