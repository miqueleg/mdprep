# Metal Centers

Metal chemistry is opt-in. Every metal atom, element, oxidation-state charge,
model, and bonded coordinator must be stated in the manifest. mdprep does not
infer oxidation states or decide whether a site should be bonded from geometry.

## Nonbonded Amber ion model

Use `model: nonbonded` when the ion should interact only through its formal
point charge and an Amber Li/Merz Lennard-Jones model:

```yaml
metals:
  - id: catalytic_zinc
    model: nonbonded
    ions:
      - selector:
          chain: Z
          resname: ZN
          resid: 500
          icode: null
          atom_name: ZN
        element: Zn
        charge: 2
    nonbonded:
      parameter_set: "12_6_4"
```

`parameter_set` must be one of `12_6`, `cm`, `hfe`, `iod`, or `12_6_4`.
`cm` is Amber's alias for the conventional 12-6 set. These families target
different observables; mdprep deliberately does not select one automatically.
Quote values containing underscores in YAML (for example, `"12_6_4"`) so the
YAML parser does not interpret them as underscored integers.
The model must be available for the requested element, +1 through +4 charge,
and configured TIP3P or OPC water model in the active AmberTools installation.

mdprep reads the exact MASS and NONBON entry from AmberTools, writes a one-atom
mol2 carrying the declared formal charge, and records the source frcmod,
Amber atom type, mass, Rmin/2, and epsilon in `metal_report.json`. Missing
element/charge combinations fail before `tleap`.

The 12-6-4 family needs more than its frcmod: the pair-specific C4 table must be
present in the topology. mdprep runs ParmEd `add12_6_4` after each relevant
dry or solvated `tleap` build, preserves the pre-12-6-4 topology and command
records, and verifies `%FLAG LENNARD_JONES_CCOEF` before publishing the final
`.prmtop`.

Current OpenMM reads this Amber flag and represents the total interaction as
the sum of its normal Coulomb/12-6 `NonbondedForce` and a separate
`CustomNonbondedForce` containing `-C4/r^4`. Load mdprep's `.prmtop` with
`AmberPrmtopFile`; rebuilding from force-field XML would lose the Amber C4
table. When OpenMM validation is enabled, mdprep reports whether both the
Amber flag and OpenMM r^-4 force were detected.

A nonbonded model has no metal-coordinator bonds or angle terms. It therefore
does not constrain coordination geometry; ligand exchange or site distortion
can occur during MD. That is a property of the model, not a missing `tleap`
bond.

## Bonded MCPB.py model

Use `model: bonded_mcpb` for an explicitly reviewed MCPB.py bonded site. The
site may contain multiple interacting metal ions, protein residues, retained
waters, and configured ligands/cofactors:

MCPB.py requires `AMBERHOME` even when the executable is installed in a Conda
environment. mdprep resolves the AmberTools root from the configured MCPB.py
executable (then the normal AmberTools discovery paths), validates its
`dat/leap/parm` directory, exports `AMBERHOME` only to the recorded MCPB
subprocesses, and records the resolved root in `metal_report.json`.

```yaml
metals:
  - id: catalytic_zinc
    model: bonded_mcpb
    ions:
      - selector: &zn
          chain: Z
          resname: ZN
          resid: 500
          icode: null
          atom_name: ZN
        element: Zn
        charge: 2
    mcpb:
      executable: MCPB.py
      workflow: prepare_inputs
      provisional_nonbonded_parameter_set: "12_6"
      cutoff_angstrom: 2.8
      bonds:
        - ion: *zn
          coordinator:
            chain: A
            resname: HIS
            resid: 42
            icode: null
            atom_name: NE2
        - ion: *zn
          coordinator:
            chain: L
            resname: SUB
            resid: 401
            icode: null
            atom_name: O1
      additional_residues: []
      force_constant_method: seminario
      charge_restraint: backbone_heavy
      software_version: g16
      small_model_charge: 1
      small_model_spin: 1
      large_model_charge: 1
      large_model_spin: 1
      scale_factor: 1.0
      large_opt: 0
      artifacts: null
```

The small- and large-model charges and spin multiplicities are mandatory.
Calculate them for the actual capped models written by MCPB.py; do not copy
the illustrative values above without reviewing the generated atom lists.

MCPB.py automatically treats N, O, S, F, Cl, Br, and I atoms inside its cutoff
as bonded candidates. mdprep reproduces that candidate search and requires
every candidate to appear in `mcpb.bonds`. A declared bond outside the cutoff
is passed through `add_bonded_pairs`. This makes the cutoff useful without
allowing it to silently decide the coordination sphere. Metal-metal bonds are
not inferred.

Every non-protein/non-water coordinating or additional residue must also be a
configured `ligands` entry with final mol2/frcmod parameters. This is how
substrates and cofactors enter the MCPB models. Ligands may use GAFF or GAFF2,
but one MCPB run cannot mix both families.

## Protonation coupling

Metal constraints are applied before PropKa or xTB:

- HIS coordination through ND1 requires HIE, leaving ND1 unprotonated.
- HIS coordination through NE2 requires HID, leaving NE2 unprotonated.
- Bonds requiring both neutral histidine nitrogens fail as inconsistent.
- A matching manual HID/HIE override is retained. A conflicting override and
  bond stop the run so the user can correct the chemistry.
- Metal-coordinating ASP/GLU/CYS/LYS/ARG states are not uniquely determined by
  the donor atom. They require an explicit `protonation.overrides` entry rather
  than a metal-blind PropKa result.
- Metal-bound TYR ionization is not yet represented and fails explicitly.

After all residue states and ligand parameters are final, mdprep performs a
preserved dry `tleap` pass using the explicitly chosen provisional nonbonded
ion family. This adds the actual force-field hydrogens before MCPB.py builds
its capped QM models. mdprep then checks the HID/HIE donor hydrogen pattern;
temporary xTB cluster hydrogens are never reused.

## Staged QM workflow

MCPB.py writes Gaussian or GAMESS input files but does not run those QM jobs.
The external-artifact workflow is therefore deliberately staged:

1. Set `workflow: prepare_inputs` and run through the metal stage:

   ```bash
   mdprep prepare system.yaml --stop-after metals
   ```

2. Review the numbered, hydrogenated MCPB input and generated small/large
   models. Run the generated QM inputs with the configured software.
3. Set `workflow: complete`, provide the resulting files under `artifacts`,
   and rerun mdprep with `--overwrite` for the same mdprep output directory.
   Artifact paths may point directly to the canonical files in the preserved
   MCPB work directory; mdprep detects that case without copying a file onto
   itself.

For Gaussian Seminario or modified-Seminario, provide `small_opt_fchk` after
the small frequency calculation and `large_mk_log`. For GAMESS Seminario,
provide `small_fc_log` and `large_mk_log`. The Z-matrix method requires a
Gaussian `small_fc_log`; the empirical method needs no small-model QM
artifact. Every complete workflow needs `large_mk_log` for MCPB RESP fitting.

Example completion block:

```yaml
workflow: complete
artifacts:
  small_opt_fchk: qm/catalytic_zinc_small_opt.fchk
  small_fc_log: null
  large_mk_log: qm/catalytic_zinc_large_mk.log
```

mdprep stages those files under MCPB's canonical names, runs the selected
step 2, RESP step 3, and bonded step 4b through the external-command recorder.
It preserves all MCPB inputs, outputs, stdout, and stderr. The MCPB-generated
`tleap` file is not executed as the final build; mdprep extracts its generated
templates, parameters, and required bonds into mdprep's own final `tleap`
workflow, so configured solvation and validation remain authoritative. All
MCPB residue renames are restored to original chain/residue numbering and
recorded in the report.

The same `workflow: complete` route can resume a `mcpb_resp_pyscf` ligand after
the PySCF calculation has finished but a later AmberTools step has failed. Set
`software_version: gau` and point `artifacts.small_opt_fchk` and
`artifacts.large_mk_log` at the preserved mdprep-generated files. This reuses
only the explicitly named QM artifacts; MCPB RESP fitting, bonded assembly,
final `tleap`, and validation are rerun and recorded. The artifact paths are an
explicit provenance assertion by the user and are never guessed by mdprep.

For the final LEaP input, mdprep renumbers residues sequentially in the same
deterministic order used by MCPB.py before applying MCPB's generated bond
commands. This is required when the original PDB contains negative, zero,
non-contiguous, or insertion-coded residue identifiers. The original identity,
the temporary LEaP index, and the reason are recorded in the LEaP report; atom
order and coordinates are unchanged.

Ligand-coordinate validation also follows the recorded MCPB residue rename
(for example `AKG` to `AG1`) when locating the dry LEaP output. Coordinates are
still compared against the original selected ligand, so a rename cannot hide
atom reordering or movement.

## Fixed-geometry PySCF/MCPB workflow

`workflow: pyscf` completes the Hessian and RESP stages without Gaussian or
GAMESS artifacts. The geometry source is explicit. For an externally optimized
small model, use:

```yaml
workflow: pyscf
force_constant_method: seminario
software_version: gau
large_opt: 0
pyscf:
  geometry_source: external_pdb
  optimized_small_model_pdb: qm/catalytic_zinc_small.optimized.pdb
  small_model_geometry_status: user_optimized
  method: B3LYP
  basis: "6-31G*"
  max_cycle: 150
  conv_tol: 1.0e-9
  scf_algorithm: diis
  initial_guess: minao
  adiis_precondition_cycles: 15
  adiis_precondition_dft_grid_level: null
  level_shift_mode: static
  level_shift_hartree: 0.0
  damping_factor: 0.0
  diis_space: 8
  large_model_density_fitting: false
  large_model_auxbasis: null
  dft_grid_level: 3
  num_threads: 4
  max_memory_mb: 8000
  embedding_cutoff_angstrom: null
  embedding_min_distance_angstrom: 1.2
  esp_batch_size: 256
```

On the first run, if `optimized_small_model_pdb` is absent, mdprep preserves
the generated `<site>_small.pdb` and stops with its path. Optimize that model
outside mdprep, without adding, deleting, or reordering atoms and without
changing atom names or residue identities. Then set the path and rerun with
`--overwrite`. mdprep validates the optimized file atom by atom before any QM
calculation.

For an end-to-end software test only, `small_model_geometry_status:
test_unoptimized` explicitly accepts a coordinate-identical copy of the
generated small model. mdprep propagates a `TEST ONLY
[UNOPTIMIZED_MCPB_GEOMETRY]` warning into the QM, metal, manifest-lock, and
final validation reports. Parameters from this mode are not suitable for
production MD. It is never selected automatically.

The PySCF small-model job is one converged, fixed-geometry SCF followed by an
analytic Cartesian Hessian at those same coordinates. The Hessian is not a
geometry optimization; it is required by Seminario or modified-Seminario to
derive bonded force constants. An energy-only single point cannot produce
those force constants. MCPB.py requires the conventional filename
`<site>_small_opt.fchk`; the `opt` text in that adapter filename does not mean
that mdprep performed an optimization.

The same explicit SCF controls are applied to every PySCF job configured by
this block. They are especially important for high-spin, electrostatically
embedded metal clusters. `scf_algorithm: newton` selects PySCF's second-order
solver; DIIS can instead be stabilized with a reviewed level shift, damping,
and larger DIIS subspace. `level_shift_mode: static` applies a constant shift;
`dynamic` uses PySCF's DIIS-only energy-dependent level shift and is never
selected automatically. The chosen protocol, converged energy, and available
spin-contamination diagnostics are written to the PySCF/MCPB report. Failure
to converge is fatal: mdprep does not silently remove point charges, truncate
the environment, change the electronic method, or accept the last SCF
iteration as a charge model.

`scf_algorithm: adiis_then_diis` is an explicit high-spin preconditioner. It
runs the configured number of energy-minimizing ADIIS cycles, then starts a
fresh CDIIS subspace from that density and applies the configured static level
shift. If `adiis_precondition_dft_grid_level` is set, only the ADIIS phase uses
that coarse grid; the final CDIIS phase rebuilds `dft_grid_level`. Both phases,
their cycle counts, and the final convergence are recorded. A nonconverged
preconditioning phase is allowed, but a nonconverged final phase is fatal.

`scf_algorithm: adiis_then_newton` uses the same explicit ADIIS density
preconditioner but passes its density directly to PySCF's second-order Newton
solver on the rebuilt final DFT grid. This is intended for reviewed difficult
open-shell metal models where CDIIS remains in a small nonconvergent cycle.
It is not an automatic fallback: the manifest records the choice, and failure
of the Newton phase remains fatal. The electronic method, electrostatic
embedding, charge, spin, basis, density-fitting choice, and convergence
tolerance are unchanged by the handoff.

For a reviewed recovery from a previous large-model calculation, set
`scf_algorithm: newton` and provide both `large_model_restart_checkpoint` and
its `large_model_restart_checkpoint_sha256`. The checkpoint supplies only the
initial density. mdprep rebuilds the current molecule and MM embedding, pins
the restart bytes by SHA-256, and rejects any mismatch in atom identities/order,
charge, spin, or basis dimensions. Coordinates are exact by default. An
explicit `large_model_restart_max_displacement_angstrom` can authorize a nearby
geometry; mdprep then requests PySCF basis projection and records maximum/RMS
displacement and whether projection occurred. The seed path and verified digest
are recorded in the PySCF/MCPB report. No checkpoint is discovered or trusted
implicitly; checkpoint provenance remains an explicit user-reviewed input.

`large_model_density_fitting: true` selects PySCF's RI-JK implementation only
for the embedded large-model ESP SCF. It leaves the selected B3LYP/6-31G*
orbital method, QM atoms, MM point charges, and RESP centers unchanged while
approximating the Coulomb/exchange integrals in an auxiliary basis. Pin
`large_model_auxbasis` when exact auxiliary-basis reproducibility is required;
otherwise PySCF records its automatically selected bases. Because this is a
scientific approximation, it is disabled by default and never activated as an
automatic response to slow or failed SCF convergence.

For a configured cofactor with `charge_method: mcpb_resp_pyscf`, its GAFF/AM1-BCC
mol2 is provisional and is used only to construct the refinement and MCPB
models. After MCPB's joint two-stage RESP fit, mdprep validates the generated
renamed-residue mol2 atom by atom, records its (not necessarily integer)
fragment charge, and promotes that mol2 as the final charge artifact. The
provisional mol2 is explicitly excluded from the final tLEAP ligand list;
MCPB's generated tLEAP setup loads the fitted residue mol2 together with the
coordinating amino-acid and metal templates. A missing or mismatched fitted
mol2 is fatal.

Alternatively, `geometry_source: qmmm_refinement` consumes the completed ASH
QM/MM relaxation from the same preparation. The bonded site must be explicitly
selected in `refinement.qm_components.metal_sites`. MCPB.py builds its capped
small model from that relaxed full-system structure, and the refinement PDB and
hash are recorded as geometry provenance. Supplying
`optimized_small_model_pdb` or `small_model_geometry_status` in this mode is an
error.

### GFN2-xTB or g-xTB small-model Hessian

The small-model Hessian may be evaluated with standard GFN2-xTB or a g-xTB
enabled executable. This changes only the Hessian backend: PySCF remains
mandatory for the electrostatically embedded large-model ESP calculation.
Consequently, `method`, `basis`, and the PySCF resource/grid settings still
configure that large-model calculation.

```yaml
pyscf:
  geometry_source: qmmm_refinement
  hessian_backend: xtb
  xtb:
    model: gfn2  # or gxtb
    executable: xtb
    release_tag: null
    expected_executable_sha256: null
    accuracy: 0.001
    num_threads: 1
    # Optional for difficult open-shell displaced Hessian points.
    max_scf_iterations: 1000
  method: B3LYP
  basis: "6-31G*"
```

mdprep runs `xtb ... --gfn 2 --hess` or `xtb ... --gxtb --hess` at the selected
coordinates and converts the full Cartesian Hessian to MCPB.py's
formatted-checkpoint adapter. The executable version, SHA-256 digest, complete
command record, raw output, energy, gradient norm, Hessian digest, and runtime
are preserved. Set `expected_executable_sha256` to the digest of the reviewed
binary when a byte-identical executable is required.

xTB obtains this Hessian by numerical differentiation of analytic gradients.
The default `accuracy: 0.001` is intentionally tighter than xTB's normal
single-point setting. mdprep rejects malformed, non-finite, wrong-sized, or
materially asymmetric Hessians. For `small_model_geometry_status:
user_optimized` and for QM/MM-refined production geometries, it also rejects
raw numerical asymmetry warnings instead of silently symmetrizing a noisy
production result. The official g-xTB macOS binary is restricted to
`num_threads: 1` because its parallel numerical Hessian is not reliable.

MCPB.py diagonalizes non-symmetric 3x3 interatomic Hessian blocks. A real
Cartesian Hessian can therefore give complex-conjugate eigenpairs whose final
projected force constant is real but retains NumPy's complex scalar type;
unpatched MCPB.py then fails while rounding that value. mdprep runs Seminario
steps through a recorded compatibility adapter that converts only negligible
imaginary residuals (relative tolerance `1e-10`) to real scalars. A materially
complex result still fails. The Hessian and numerical force constant are not
altered, and the conversion count is written to the MCPB stdout log.

This backend is a speed/accuracy tradeoff, not a claim of DFT equivalence.
GFN2-xTB, g-xTB, and B3LYP can produce materially different metal-ligand force
constants at identical coordinates. Compare the resulting Seminario parameters
with an independent method and chemical knowledge before production use,
especially for a metal/oxidation-state combination outside the method's
demonstrated domain. The 7E07 benchmark compares both xTB variants with a
conventional UB3LYP/6-31G* MCPB reference and with MACE-POLAR-1. The fixed
capped cluster is not a stationary point for the reference methods, so these
data measure method sensitivity rather than establish an acceptance reference.

### MACE-POLAR-1 small-model Hessian

MACE-POLAR-1 can supply the small-model analytical Cartesian Hessian while
PySCF continues to supply the embedded large-model ESP used for MCPB RESP:

```yaml
pyscf:
  geometry_source: qmmm_refinement
  hessian_backend: mace_polar1
  mace_polar1:
    python_executable: python
    model: polar-1-m
    device: cpu
    num_threads: 4
    expected_model_sha256: null
    finite_difference_step_angstrom: 0.001
    max_hessian_vector_relative_error: 0.01
    accept_model_license: true
  method: B3LYP
  basis: "6-31G*"
```

MACE-POLAR-1 uses the manifest's exact small-model charge and spin
multiplicity. mdprep evaluates the model in float64, rejects unsupported
elements, validates dimensions/finiteness/symmetry, and checks one
deterministic Hessian-vector product against central finite differences of
MACE forces. It records the model checkpoint path and SHA-256, Python
executable SHA-256, package versions, charge, multiplicity, energy, gradient
norm, validation error, runtime, and both external-command records. Set
`expected_model_sha256` when a byte-identical checkpoint is required.

The `polar-1-s`, `polar-1-m`, and `polar-1-l` checkpoints are supported. The
default medium model is a compromise between cost and receptive field.
Checkpoint use is subject to MACE-POLAR-1's Academic Software License, so the
manifest must explicitly set `accept_model_license: true` after the user has
reviewed it. mdprep does not silently accept that license.

MACE-POLAR-1 is a learned potential trained against hybrid-DFT data, not a
drop-in experimental truth standard. Its speed does not remove the need to
inspect stationarity, all generated MCPB terms, coordination chemistry, and
short MD behavior. The controlled 7E07 comparison is documented in
[the MCPB Hessian benchmark](mcpb_xtb_benchmark.md).

The large model uses a separate fixed-geometry, electrostatically embedded
PySCF SCF. MM charges from the provisional Amber topology polarize the QM
density, but are not RESP centers. mdprep evaluates the QM ESP and feeds it to
MCPB.py's standard two-stage RESP calculation. For a coordinating ligand,
use `charge_method: mcpb_resp_pyscf`: its AM1-BCC charges are provisional and
MCPB RESP supplies the final site charges, including any charge redistribution
across the explicitly bonded metal environment.

## Comparing MCPB parameter sets

Generate the reference with the same MCPB small-model geometry, atom typing,
declared bonds, charge, multiplicity, force-constant method, and scale factor.
Only the QM Hessian method should differ for a controlled comparison. Then add:

```yaml
parameter_comparison:
  reference_frcmod: reference/catalytic_zinc_gfn2_mcpbpy.frcmod
  candidate_label: g-xTB Seminario
  reference_label: GFN2-xTB Seminario
  require_exact_term_set: true
  max_bond_force_constant_relative_rmse: null
  max_angle_force_constant_relative_rmse: null
  max_bond_equilibrium_distance_rmse_angstrom: null
  max_angle_equilibrium_value_rmse_degrees: null
  fail_on_thresholds: true
```

The comparison reads only BOND and ANGL entries explicitly marked by MCPB.py as
Seminario-created; inherited protein or GAFF terms are excluded. It writes
JSON, CSV, and Markdown reports with term-by-term signed/relative differences,
MAE, RMSE, maxima, missing terms, file hashes, and configured-limit violations.
Null thresholds report data without inventing a pass criterion. Once reviewed,
set explicit system-specific limits if automated failure is desired.
Relative thresholds are fractions: for example, `0.20` means 20% relative
RMSE.

Non-positive MCPB bond or angle force constants are always reported as invalid.
They give the comparison status `FAIL`; `--report-only` preserves the full
diagnostic without turning the invalid parameter set into an accepted result.
Likewise, a report-only result with positive terms is not an automatic
production acceptance: large method sensitivity or a non-stationary extracted
cluster still requires scientific review.

The same comparison can be run independently:

```bash
mdprep compare-mcpb candidate_mcpbpy.frcmod reference_mcpbpy.frcmod \
  --output-dir comparison
```

Agreement with a selected reference tests method sensitivity for that geometry
and electronic state; it does not replace charge validation, vibrational
inspection, final topology checks, or restrained minimization/short MD tests.

## Interactions with QMMESP

Configured metals are included in the provisional QMMESP Amber environment as
their explicit formal charges and selected nonbonded ion models, so they can
polarize a target ligand's PySCF density. A bonded site uses
`provisional_nonbonded_parameter_set` at that earlier stage because MCPB
parameters do not exist yet.

A QMMESP target ligand cannot itself be a bonded coordinator or MCPB
`additional_residue`: MCPB RESP step 3 would replace the QMMESP charges on
that same residue. mdprep rejects this conflicting combination rather than
silently choosing one charge model.

## Current bonded-workflow limits

- One bonded MCPB site per preparation; a single site may contain multiple
  interacting ions.
- One protein chain in the MCPB input. MCPB.py discards chain identifiers,
  making automatic multi-chain peptide reconstruction unsafe.
- Metal charges are limited to +1 through +4 because the mandatory provisional
  hydrogenation/QMMESP environment uses Amber Li/Merz ion models.
- No metal-metal bond inference, covalent ligands, noncanonical polymer
  residues, ORCA backend, or Multiwfn dependency.
- No small-model optimization inside the MCPB Hessian stage. Geometry must come
  from a completed ASH QM/MM refinement or an explicitly supplied optimized PDB.
- MCPB.py and its AmberTools RESP dependencies are always required. The
  `complete` workflow additionally requires user-generated Gaussian or GAMESS
  output appropriate to the selected method; the `pyscf` workflow requires
  PySCF and either a completed selected-site QM/MM refinement or a user-supplied
  optimized small-model PDB.
