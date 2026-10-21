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

## Deriving the charges with GFN2 instead of DFT

The large model exists to be fitted to an ESP, and that fit sets the charges
the finished system carries. `esp_backend` chooses what produces it:

```yaml
pyscf:
  hessian_backend: xtb     # small-model Hessian
  esp_backend: xtb         # large-model ESP; default is pyscf
  xtb: {model: gfn2, executable: xtb, num_threads: 1}
```

On the BesD site -- a 74-atom open-shell Fe(II) large model with ~4,600
embedding charges -- GFN2 takes about a minute against tens of minutes for
B3LYP/6-31G*. The embedding, the Merz-Kollman grid and the Gaussian log
MCPB.py reads are identical either way, so the two are directly comparable.

What changes is the density underneath. GFN2 is a semi-empirical valence
method, and charges fitted to its ESP are not RESP charges in the usual
sense. Every report says so, and `esp_backend: pyscf` remains the default.

### Charges the manifest has to declare

MCPB.py has no ion parameters of its own: the Amber protein force fields
carry none, so an ion's charge reaches it only from a mol2 that mdprep
writes. The oxidation states in the manifest are what those files say, which
is why they have to be right:

```yaml
metals:
  - id: a_fe2_302_site
    ions:
      - selector: {chain: A, resname: FE2, resid: 302, atom_name: FE}
        element: Fe
        charge: 2          # becomes chargedict["FE"] for MCPB.py
        multiplicity: 5
    coordinating_ions:
      - selector: {chain: A, resname: CL, resid: 303, atom_name: CL}
        element: Cl
        charge: -1         # becomes chargedict["CL"]
    mcpb:
      small_model_charge: -1   # Fe(2+) + AKG(2-) + Cl(1-) + 2 x His(0)
      large_model_charge: -1
```

mdprep renames an ion's residue and atom to the bare element symbol inside
the MCPB working directory, because MCPB.py reads the charge by atom name in
one place and by residue name in another and pairs it with the element symbol
to find van der Waals parameters, so all three have to agree. A coordinating
halide also gets a two-character atom type carrying Amber's own Li/Merz
parameters, because MCPB.py reads a frcmod atom type from a two-character
field and cannot represent `Cl-` at all. Both renames are confined to that
directory; the prepared system keeps the names the input used.

### A zero force constant is reported, not shipped

A Seminario force constant of zero means the Hessian has no restoring force
along that bond, and mdprep stops rather than writing it out.

On BesD with `hessian_backend: xtb` and `xtb.model: gfn2`, Fe-O5 of the
bidentate alpha-ketoglutarate does exactly that. Projecting the GFN2 Hessian
onto the five Fe-ligand bonds shows why: that one comes out **negative**, and
MCPB.py clamps it to zero.

| Fe bond | r (A) | GFN2 | g-xTB |
|---|---|---|---|
| Fe-NE2 (His137) | 2.247 | 44.1 | 78.7 |
| Fe-NE2 (His204) | 2.224 | 28.4 | 84.7 |
| Fe-Cl | 2.392 | 46.6 | 112.4 |
| Fe-O1 (aKG) | 2.161 | 51.1 | 99.3 |
| Fe-O5 (aKG) | 2.179 | **-16.5** | 86.5 |

(Raw Seminario projections in kcal/mol/A^2; MCPB.py reports half these, using
the `E = k(r-r0)^2` convention.)

GFN2 is parameterised for main-group chemistry and describes transition-metal
coordination poorly, which is what the negative value is. The geometry is not
at fault: every distance is normal.

g-xTB has its own case: the dehaloperoxidase of 2QFK, a high-spin aquo
Fe(III) on a histidine. In the lowest SCF state the search finds, the angle
His89 NE2-Fe-NC of the porphyrin comes out near -6.6 kcal/mol/rad^2, and the
run stops, naming it. In another state of the same family, 1.3 kcal/mol lower,
which a multi-threaded SCF can stumble into, the same angle is +15.9 and the
run validates. A metal's angles are the softest terms of its site, and taken
at the crystal geometry with only the hydrogens relaxed -- not a minimum of
g-xTB, whose Hessian there has seven or eight negative eigenvalues -- their
sign hangs on which of these near-degenerate states the Hessian is taken in.
mdprep does not prepare this site, rather than ship either; a Hessian at a
relaxed geometry or from DFT (`hessian_backend: pyscf`, expensive at 91
atoms) is the way to parameterise it.

### g-xTB is the default, and why

A bonded metal site takes `hessian_backend: xtb` with `xtb.model: gxtb`
unless told otherwise. On this site that Hessian took 8.1 s against GFN2's
2.2 s, and the whole BesD preparation -- structure through solvated Amber
system -- completes in about a minute. The defaults need no block at all:

```yaml
mcpb:
  pyscf:
    geometry_source: qmmm_refinement
    esp_backend: xtb      # GFN2 is fine here; the Hessian is where it is not
```

g-xTB is a separate build from the GFN-only xtb conda installs. Download it
from the [official g-xTB releases](https://github.com/grimme-lab/g-xtb/releases)
and point mdprep at it once:

```bash
export MDPREP_GXTB_EXECUTABLE=/path/to/g-xtb/bin/xtb
```

or give `mcpb.pyscf.xtb.executable` the path. mdprep looks for
`MDPREP_GXTB_EXECUTABLE`, then `g-xtb`, then `xtb`, and `mdprep doctor
<manifest>` says whether a usable one was found -- it runs the candidate's
`--help` rather than trusting its name, because the conda `xtb` is called
`xtb` too and would otherwise look like a pass.

`xtb.model: gfn2` runs on the ordinary conda xtb and needs no download. It is
the right choice only once you have checked the force constants for the site
in hand; on a transition metal it may well give one that mdprep refuses.

If you want a DFT reference instead, `hessian_backend: pyscf` with a pure
functional and a small basis (for example `method: TPSS`, `basis: def2-SVP`)
is far cheaper than the B3LYP/6-31G* default, because the cost of a hybrid is
dominated by exact exchange. Expect minutes to hours for a 46-atom analytic
Hessian rather than seconds, and check the result against the tight-binding
one with `mdprep compare-mcpb`.

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

- One bonded MCPB fit per preparation; a single site may contain multiple
  interacting ions. The same site in other copies of the protein -- a dimer,
  a tetramer -- reuses that fit through `parameters_from` rather than being
  fitted again; see [oligomers.md](oligomers.md). Two sites that would each
  need a fit of their own are refused.
- A bonded site sits within one protein chain. The structure may hold any
  number of chains, but MCPB.py caps each coordinating residue from its
  neighbours in one numbering, which is wrong across a chain boundary, so a
  metal bound between two chains has to be modelled nonbonded.
- Metal charges are limited to +1 through +4 because the mandatory provisional
  hydrogenation/QMMESP environment uses Amber Li/Merz ion models.
- No metal-metal bond inference, metal clusters, covalent ligands (heme c
  among them), noncanonical polymer residues (a modified residue with a
  standard parent can be replaced by it; see [heterogens.md](heterogens.md)),
  ORCA backend, or Multiwfn dependency. A metal inside a cofactor is supported; see
  [A metal inside a cofactor](#a-metal-inside-a-cofactor-hemes).
- No small-model optimization inside the MCPB Hessian stage. Geometry must come
  from a completed ASH QM/MM refinement or an explicitly supplied optimized PDB.
- MCPB.py and its AmberTools RESP dependencies are always required. The
  `complete` workflow additionally requires user-generated Gaussian or GAMESS
  output appropriate to the selected method; the `pyscf` workflow requires
  PySCF and either a completed selected-site QM/MM refinement or a user-supplied
  optimized small-model PDB.
- A water bound to the metal is always built as H2O, and only with TIP3P. A
  bound hydroxide needs a residue template this version does not build, and a
  four-point water's extra point cannot join the QM model. See
  [Waters bound in a metal site](#waters-bound-in-a-metal-site).

## Ions bound in a metal site

A halide, azide or hydroxide bound to the metal is part of the site, and where
it is declared depends on whether a force field already knows it.

**Monatomic ions** -- F-, Cl-, Br-, I- -- go in `metals[].coordinating_ions`:

```yaml
metals:
  - id: catalytic_fe
    ions:
      - selector: {chain: A, resname: FE2, resid: 302, icode: null, atom_name: FE}
        element: Fe
        charge: 2
        multiplicity: 5
    coordinating_ions:
      - selector: {chain: A, resname: CL, resid: 303, icode: null, atom_name: CL}
        element: Cl
        charge: -1
```

They belong nowhere else. `ions[]` would have MCPB treat the halide as a metal
centre, and `ligands[]` would send it to antechamber, whose semi-empirical step
refuses a single atom. Amber's Li/Merz sets parameterise all four halides
directly, in the same file as the monovalent cations, so the force-field
relaxation has what it needs.

**Polyatomic ions** -- azide, for instance -- need none of this. antechamber
handles them, so configure one under `ligands[]` as a cofactor and name it as
an `mcpb.bonds` coordinator. That is what alpha-ketoglutarate already does in
[the BesD example](../examples/16_besd_halogenase_gfn2_mcpb.yaml).

Either way the ion is part of the metal site's QM region, contributes its
declared formal charge to any xTB cluster it falls into, and is kept by
structure normalization rather than treated as an unknown heterogen.

Only the metal contributes unpaired electrons; a closed-shell halide beside it
does not.

## Waters bound in a metal site

The resting iron of an alpha-ketoglutarate enzyme often holds a water where O2
later binds. SyrB2 ([PDB 2FCT](https://www.rcsb.org/structure/2FCT)) is the
reference case: Fe(II), two histidines, a chloride, bidentate
alpha-ketoglutarate and one water. In a predicted model there is no water at
all, so one has to be brought in from a crystal structure first.

The decision engine finds a water within bonding distance of a metal and
proposes it as a donor with the rest of the shell. Confirm the shell and the
water becomes an `mcpb.bonds` coordinator like any other:

```yaml
    mcpb:
      bonds:
        - ion: {chain: A, resname: FE2, resid: 900, icode: null, atom_name: FE}
          coordinator: {chain: A, resname: HOH, resid: 1072, icode: null, atom_name: O}
```

What mdprep does with it:

- **Kept whatever `keep_crystal_waters` says.** Dropping it with the other
  waters would leave an MCPB bond pointing at a residue that is gone. (A
  water on a nonbonded metal is kept with it too: it is the ion's first
  shell.)
- **Given to MCPB.py as a TIP3P template.** MCPB.py asks for a mol2 for every
  non-amino-acid residue bonded to the metal, a water included. mdprep writes
  it from the water's own coordinates with TIP3P's atom types and charges
  (OW -0.834, HW +0.417), which MCPB.py then refits with the rest of the site.
- **Given the H-O-H angle TIP3P never had.** TIP3P is rigid in Amber: its
  frcmod holds the shape with an H-H pseudo-bond and defines no angle. Bound to
  the metal the water is a flexible ligand, so mdprep supplies parm99's
  `HW-OW-HW` angle (100 kcal/mol/rad^2, 104.52 degrees) and MCPB.py carries it
  over to the oxygen's new atom type. Without it tleap stops on
  `HW-Y6-HW`.
- **Counted as neutral in the QM/MM refinement.**
- **Requires TIP3P.** A four-point water such as OPC has an extra point with no
  nucleus, which cannot be part of a QM model. The decision engine refuses the
  combination before a manifest is written.

### 1-4 pairs through the metal

Amber adds a scaled nonbonded interaction between the two ends of every
dihedral. Through a bonded metal that pairs atoms of different ligands: with a
chloride and a water on the same iron, Cl-Fe-O-H makes the chloride and the
water's hydrogen a 1-4 pair. MCPB.py zeroes the torsions through the metal, the
O...Cl repulsion is excluded because the two are 1-3 through the iron, and a
TIP3P hydrogen has no Lennard-Jones radius. That leaves a Coulomb attraction
between -0.95 and +0.45 with nothing to stop it: in SyrB2 the first
minimisation put the hydrogen 0.14 A from the chloride, and heating went to
NaN.

So after tleap, mdprep switches off exactly those 1-4 terms: through a metal,
between different residues, where one atom has no repulsion at all. It sets the
dihedral's "ignore end group" flag in the prmtop, which every Amber engine and
OpenMM honour, and changes nothing else. The coordination geometry is still
held by the angles MCPB.py fitted around the metal. Every pair switched off is
listed in `reports/tleap_report.md` under *1-4 Pairs Through The Metal Switched
Off*.

With the fix, SyrB2 runs the whole Roe-Brooks protocol and production with
Fe-N(His) 2.0-2.35 A, Fe-Cl 2.2-2.5 A, Fe-O(alpha-KG) 1.9-2.2 A, Fe-O(water)
2.1-2.5 A, and the water hydrogens never closer than 3 A to the chloride.

## A metal inside a cofactor: hemes

A heme is deposited as one residue. Heme b is `HEM`, 43 atoms with the iron
among them, and a cytochrome P450's iron is bonded to the porphyrin's four
nitrogens, to the proximal cysteine's sulfur and, at rest, to a water across
the heme plane. Neither half can be parameterised with the other inside it:
antechamber's semi-empirical charges do not treat iron, and MCPB.py builds a
site around a one-atom metal residue.

So mdprep moves the metal into a residue of its own as the structure is read,
from a PDB or an mmCIF file, a deposition or a prediction:

- The new residue is named after the element, keeps the cofactor's chain and
  residue number, and comes right after it. In CYP119
  ([PDB 1IO7](https://www.rcsb.org/structure/1IO7)) `A:HEM1401` becomes
  `A:HEM1401`, 42 atoms, and `A:FE1401`.
- The name is recorded as assigned, so no oxidation state is read into it:
  `FE` is Fe(III) in the PDB dictionary, and a heme's iron is either.
- The reader says so in the structure's warnings, and selectors in the manifest
  use the new residue.

From there the iron is an ordinary metal site and the porphyrin an ordinary
ligand that coordinates it. The decision engine proposes the bonded model with
six donors, NA, NB, NC and ND, the cysteine's SG and the water's O, and asks:

| Question | CYP119 at rest | Why it is asked |
| --- | --- | --- |
| Oxidation state | +3 | `HEM` does not say: Fe(III) in a resting P450, Fe(II) in deoxymyoglobin |
| Cysteine protonation | `CYM` | The proximal thiolate. Asked before the model charge, which counts it |
| Spin of the QM region | 2 | Six-coordinate low-spin Fe(III). Five-coordinate, with the water gone, it is high-spin: 6 |

It also proposes, filled in and changeable:

- **The porphyrin's SMILES and net charge**: protoporphyrin IX as it binds the
  iron, with both pyrrole nitrogens deprotonated and both propionates ionised,
  -4. Change it if a propionate is protonated in your structure.
- **The model charge**, -2: Fe +3, porphyrin -4, thiolate -1, water 0.

The porphyrin's charges come from the site's fit (`mcpb_resp_pyscf`), as an
alpha-ketoglutarate's on a non-heme iron do. A proposal is not a default: a
manifest written without a front end still has to state each value.

A predicted P450 has no water on its iron. `mdprep closest-xray` and
`mdprep transplant-waters` bring it in from a crystal structure in the same
state, as for any other metal: for a Boltz model of CYP119 they find 1IO7 and
put its HOH1402 2.27 A from the predicted iron
(see [predicted_structures.md](predicted_structures.md)).

### Smearing the xTB occupations

An open-shell porphyrin has iron d orbitals close enough together that xTB's
SCF oscillates at its own electronic temperature of 300 K and does not
converge: on the CYP119 small model it was still oscillating after 1,000
cycles. At 1000 K it converges in seconds. The builder therefore writes this
for a site whose donors include a heme:

```yaml
      pyscf:
        xtb:
          model: gxtb
          electronic_temperature_kelvin: 1000.0
```

It applies to both the Hessian and the GFN2 ESP of the large model, and it is
in every command record. A site that fails this way without it says which
setting to add.

### More than one SCF state

Smearing makes an open-shell heme's SCF converge; it does not make the state
it converges to unique. On the high-spin aquo Fe(III) of a dehaloperoxidase
(2QFK: histidine, heme and water, spin 6), g-xTB at 1000 K reaches two kinds
of solution with the same charge and spin: one with 3.1 unpaired electrons on
the iron, and a family 27 kcal/mol lower with 3.6-3.7. Which one a cold start
finds depends on as little as the number of threads -- with one to four it
settled in the first, with eight in either from one run to the next, after up
to 1,100 cycles -- and the Fe-N force constants of the two differ by up to 1.7
times, while those within the lower family agree to about 10%.

So for a heme site the builder also writes

```yaml
          max_scf_iterations: 3000
          scf_annealing_kelvin: [2000.0, 3000.0, 5000.0]
```

and mdprep converges the SCF four times before the Hessian: once straight at
1000 K, and once from a converged SCF at each of these temperatures, cooled to
1000 K. Each is single-threaded, so the thread count cannot decide. The lowest
energy wins, the Hessian starts from its wavefunction, and the run stops if
the Hessian's own SCF ends in another state. Every start, converged or not, is
in `gxtb_hessian.json` and among the small model's command records, with its
files in the `scf_start_*` folders beside them.

On 2QFK the search takes about two and a half minutes and finds the lower
family -- where the site then stops, at a negative angle force constant (see
below). On a peroxygenase's low-spin Fe(III) (5OXU) every start reaches the
same state, and on chloroperoxidase's five-coordinate high-spin one (1CPO) the
direct start is the lowest -- the annealed ones settle 18-28 kcal/mol above
it -- so there the search changes nothing. No set of starts is guaranteed to
find the lowest state there is: the energies each reached are recorded so you
can see how far apart they were.

### The iron's force constants

The Seminario method reads a bond's force constant from the 3 x 3 block of the
Hessian between its two atoms. For a heme's iron the result depends on which of
the two it starts from, because that block mixes the bond's stretch with the
porphyrin's ring modes, and MCPB.py takes one order. On CYP119's crystal
geometry it writes 24, 61, 91 and 93 kcal/mol/A^2 for the four Fe-N bonds;
projected from the iron instead, the same four come out 76-107. On a Boltz model
of the same enzyme MCPB.py writes 52-82.

Either way the site holds. Through the Roe-Brooks equilibration and 200 ps of
production on the crystal parameters, the four Fe-N bonds average 2.00-2.04 A,
Fe-S 2.29 A and Fe-O 2.25 A (crystal: 1.98-2.01, 2.27 and 2.27 A), the softest
Fe-N fluctuating only slightly more than the others. The water, at 22
kcal/mol/A^2, moves most. Sections E7 and H of the official notebook show your
own site's constants and its distances after every step: look at both.

### What is not supported

- **A residue holding more than one metal**, an iron-sulfur cluster for
  instance, is refused before any question is asked: no single site describes
  a cluster.
- **Heme c** (`HEC`) is bonded to two cysteines through thioethers, which
  makes it a covalent ligand, and it is refused by name. A bond cannot be told
  from a clash by distance alone, and predicted complexes have clashes that
  short.
- **A heme bonded to the protein** -- the heme of a myeloperoxidase or a
  lactoperoxidase, held by ester and sulfonium bonds the file declares in its
  LINK records -- is refused the same way: it cannot be built bonded, and
  cannot be left out without its metal site. See
  [heterogens.md](heterogens.md).
- **An oxoanion** is not split: the vanadate of a vanadium haloperoxidase
  would leave four oxygens bonded to nothing, so it stays whole and is
  refused, being neither a metal site nor a ligand.
- Heme b is the only metal cofactor whose SMILES is proposed. Another
  single-metal cofactor is split the same way, and its metal-free SMILES has
  to be typed.

[examples/19_cyp119_heme_mcpb.yaml](../examples/19_cyp119_heme_mcpb.yaml) is
the manifest for CYP119's chain A.

## Magnesium, calcium, and metals that are not part of the system

Magnesium and calcium are proposed nonbonded even when the planner finds a
coordination shell: they are closed-shell ions held by electrostatics, which a
Li/Merz ion model describes, and a QM calculation would add cost and nothing
else. 5OXU's magnesium, held by the heme's propionate, is one. A nonbonded ion
next to a bonded site's QM region is part of the MM environment of its QM/MM
refinement, like a crystal chloride.

A metal that is not inside a cofactor can also be left out: its model
question offers **Remove it**, for a nickel from a His-tag purification or a
cadmium from the crystallisation buffer, and the metal goes into
`structure.remove_residues` with its reason. Nothing else is asked about it.

## Free ions in the crystal

A chloride or sodium the crystal resolved away from any metal is a *crystal
ion*, not a ligand: tleap builds it from its ion library like a counterion.
`structure.keep_crystal_ions` (default `true`) decides whether it stays. A
halide bound to a bonded metal site is kept with the site whatever the answer;
one bound to a nonbonded metal follows the answer like any other crystal ion.

