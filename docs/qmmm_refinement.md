# Active-site QM/MM refinement

`mdprep` can optionally refine an active-site geometry before the final ligand,
cofactor, metal, and Amber parameterization stages. The refinement uses ASH as
the QM/MM interface and the provisional Amber topology through OpenMM for the
MM calculation. GFN2-xTB uses electrostatic embedding. g-xTB and
MACE-POLAR-1 use mechanical embedding because these interfaces do not accept
the differentiable MM point-charge environment required by ASH electrostatic
embedding. MACE-POLAR-1 is a machine-learned high-layer potential rather than
an electronic-structure method, so that variant is scientifically an ML/MM
refinement even though it reuses the same ASH QM-region machinery.

This is a geometry-refinement stage. It is separate from the PySCF QMMESP
workflow that derives final ligand charges. A ligand can therefore be refined
with provisional AM1-BCC charges and subsequently receive `qmmesp_pyscf`,
`gas_resp_pyscf`, `mcpb_resp_pyscf`, user-supplied, or final AM1-BCC parameters
as requested in its normal ligand configuration.

## Workflow

With `refinement.enabled: true`, preparation proceeds in this order:

1. Normalize the input structure without silently removing heterogens.
2. Apply the requested amino-acid protonation protocol and manual overrides.
3. Generate provisional AM1-BCC parameters for every configured ligand or
   cofactor and build an unsolvated Amber topology with `tleap`.
4. Protonate every retained crystal water through that provisional `tleap`
   build.
5. Select the explicit QM components and the complete active residues.
6. Optimize the active region with ASH, OpenMM, and the explicitly selected QM
   method/embedding. Everything outside the active region remains frozen.
7. Restore stable input chain/residue identities, then rerun the requested
   amino-acid protonation protocol on the optimized geometry.
8. Run the normal final ligand, cofactor, metal, solvation, `tleap`, and
   validation stages.

For a bonded MCPB site, the refined geometry can be consumed automatically by
the metal stage. Set `mcpb.workflow: pyscf` and
`mcpb.pyscf.geometry_source: qmmm_refinement`. Manifest validation requires
that exact site under `refinement.qm_components.metal_sites`; runtime validation
requires a completed refinement result and its preserved PDB. MCPB.py step 1
then constructs the capped small model from the relaxed full-system geometry.
No manual small-model PDB copy is used in this mode.

The generated ASH driver, JSON input/output, standard output/error, provisional
AM1-BCC files, provisional `tleap` script/log, optimizer files, and refinement
reports are retained below the preparation output directory.

ASH/geomeTRIC writes full-system, active-region, and QM-region XYZ optimizer
artifacts. The optional ASH OpenMM PDB-trajectory hook is disabled because an
energy/gradient `OpenMMTheory` does not create the MD `simulation` object that
hook requires; the final coordinates are transferred from the converged ASH
fragment and validated against the provisional topology atom count.

`structure.keep_crystal_waters` must be `true` when refinement is enabled.
This prevents the structure stage from deleting waters that should participate
in the reviewed active-site model.

## QM and active regions

The QM region contains:

- every ligand/cofactor id listed under `qm_components.ligands`;
- every metal ion in a site listed under `qm_components.metal_sites`; and
- each complete protein residue explicitly declared as a coordinator of those
  metals. For a bonded MCPB site, coordinator residues in `mcpb.bonds` are also
  used.

The active region contains the complete QM region plus every complete non-water
residue with at least one physical atom within 4.0 Å of a QM atom. Retained
crystallographic waters use a separate 8.0 Å cutoff and enter as complete,
protonated water residues. Water virtual sites, when present, are controlled by
their parent atoms and are not independent optimizer coordinates.

All non-protein, non-water molecules contacting the selected QM region must be
configured and selected explicitly. `mdprep` fails rather than treating an
unreviewed nearby cofactor or substrate as MM.

## Required charge and spin data

Every selected ligand/cofactor must explicitly define both `net_charge` and
`multiplicity`. Although ordinary ligand setup defaults multiplicity to one,
that implicit default is rejected for refinement.

Every selected metal ion must explicitly define `charge` and `multiplicity`.
Nonbonded metal sites must also list their full coordinating residues under
`metal_coordinating_residues`; no coordination chemistry is inferred from a
distance cutoff.

The total QM charge is derived and reported as:

- selected ligand/cofactor net charges;
- selected metal formal charges; and
- formal charges of the protonated coordinating protein residues, including
  recognized charged termini.

Multiplicity is assigned as follows:

- all components singlet: total multiplicity 1;
- exactly one open-shell component: that component's multiplicity;
- multiple open-shell components: `total_qm_multiplicity` is mandatory because
  the spin coupling cannot be inferred safely.

## Manifest example

```yaml
ligands:
  - id: substrate
    selector: {chain: B, resname: SUB, resid: 501, icode: null}
    net_charge: 0
    multiplicity: 1       # must be explicit for selected QM components
    atom_types: gaff2
    charge_method: qmmesp_pyscf
    qmmesp:
      qm_engine: pyscf
      method: HF
      basis: 6-31G*

refinement:
  enabled: true
  backend: ash
  qm_method: gfn2_xtb
  embedding: electrostatic
  qm_components:
    ligands: [substrate]
    metal_sites: []
    metal_coordinating_residues: {}
  active_region_cutoff_angstrom: 4.0
  active_water_cutoff_angstrom: 8.0
  include_contact_waters: true
  movable_atoms: all_active_region
  total_qm_multiplicity: null
  ash:
    python_executable: python
    xtb_executable: xtb
    allow_unusual_link_boundaries: false
    max_iterations: 250
    num_cores: 4
    platform: CPU
```

To relax proton positions while preserving a trusted crystallographic
heavy-atom model, use the same full QM-region declaration with:

```yaml
refinement:
  enabled: true
  qm_method: gfn2_xtb
  embedding: electrostatic
  movable_atoms: active_region_hydrogens
  post_refinement_protonation: reuse_initial
```

The Fe/ligand/cofactor/coordinating-residue QM region is unchanged and still
defines the GFN2 electronic calculation. Only hydrogens in the selected 4 Å
protein/non-water and 8 Å water active residues are optimizer variables. QM
heavy atoms, MM heavy atoms, and hydrogens outside the active region remain
fixed. The output selection report records the policy and every zero-based
movable atom index; heavy-atom invariance should be audited before accepting
the refined coordinates.

`reuse_initial` is mandatory for this hydrogen-only policy. The residue states
selected before refinement are restored after coordinate transfer without a
second PropKa/xTB decision, and no refined hydrogen is stripped. mdprep checks
HID/HIE/HIP, acid, lysine, cysteine, and TIP3P water hydrogen counts, then
requires pre-MCPB tleap and MCPB.py to preserve every supplied coordinate.
This makes the g-xTB Hessian and the embedded large-model ESP use the actual
proton-relaxed structure instead of a freshly rebuilt hydrogen geometry.

For method-specific mechanically embedded refinements, retain the same QM and
active-region declarations and change only the method block:

```yaml
refinement:
  enabled: true
  backend: ash
  qm_method: gxtb
  embedding: mechanical
  # qm_components, cutoffs, charge/multiplicity, and ASH settings as above
  ash:
    python_executable: python
    xtb_executable: xtb
    # mdprep's default; tighter values can make difficult g-xTB SCFs less robust.
    accuracy: 0.1
```

or:

```yaml
refinement:
  enabled: true
  backend: ash
  qm_method: mace_polar1
  embedding: mechanical
  # qm_components, cutoffs, charge/multiplicity, and ASH settings as above
  mace_polar1:
    model: polar-1-m
    model_path: models/MACEPOLAR1Mmodel
    expected_model_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    device: cpu
    python_search_paths: []
    # Set true only after reviewing and accepting the model license.
    accept_model_license: true
```

Mechanical embedding evaluates the QM-region energy and gradients with the
selected method while OpenMM supplies QM-MM electrostatics and van der Waals
terms. ASH removes the provisional MM description of internal QM-region bonded
and nonbonded interactions, so they are not double counted. It does not
polarize the QM density or MACE prediction with MM point charges. Comparisons
between methods must start from the same provisional structure and use the same
QM atom list, active atom list, charge, multiplicity, protonation, and cutoffs.
The resulting total energies belong to different Hamiltonians and must not be
compared as absolute electronic energies; compare structures and observables.
Both methods retain the same default ASH/ORCA optimizer convergence criteria.

For a nonbonded metal site:

```yaml
metals:
  - id: catalytic_zinc
    model: nonbonded
    ions:
      - selector: {chain: Z, resname: ZN, resid: 1, atom_name: ZN}
        element: Zn
        charge: 2
        multiplicity: 1
    nonbonded: {parameter_set: 12_6}

refinement:
  enabled: true
  qm_components:
    ligands: [substrate]
    metal_sites: [catalytic_zinc]
    metal_coordinating_residues:
      catalytic_zinc:
        - {chain: A, resname: HIS, resid: 64}
        - {chain: A, resname: HIS, resid: 116}
        - {chain: A, resname: ASP, resid: 148}
  ash:
    python_executable: python
    # Full coordinating residues make reviewed peptide C-N QM/MM boundaries.
    allow_unusual_link_boundaries: true
```

Explicitly selected metal-ion atoms are excluded from ASH's distance-based
QM/MM link-boundary discovery. Their coordinating sphere remains in the QM
region as configured, while peptide bonds at the edges of complete coordinating
residues retain their reviewed link atoms. This prevents a nearby relaxed MM
water from being misclassified as a new covalent metal boundary on an optimizer
restart; the excluded zero-based atom indices are recorded in the selection and
ASH input reports.

Manual protonation overrides still take precedence in both protonation passes.
For example, a catalytic aspartate that must remain deprotonated should use an
explicit `state: ASP` override.

## Integrated metal-cluster completion

The fast bonded-metal route is:

```yaml
metals:
  - id: catalytic_metal
    model: bonded_mcpb
    ions:
      - selector: {chain: Z, resname: ZN, resid: 500, atom_name: ZN}
        element: Zn
        charge: 2
        multiplicity: 1
    mcpb:
      workflow: pyscf
      # Explicit bonds, model charges/spins, and other MCPB settings omitted.
      pyscf:
        geometry_source: qmmm_refinement
        hessian_backend: xtb
        xtb:
          model: gfn2       # or gxtb with the appropriate executable
          executable: xtb
          accuracy: 0.001
          num_threads: 1
        method: B3LYP       # embedded large-model ESP/RESP calculation
        basis: 6-31G*

refinement:
  enabled: true
  qm_components:
    ligands: [substrate, cofactor]
    metal_sites: [catalytic_metal]
  ash:
    python_executable: python
    xtb_executable: xtb
    # Required here because the complete coordinating residues are QM.
    allow_unusual_link_boundaries: true
```

GFN2-xTB/g-xTB supplies the fixed-geometry small-model Hessian used by
Seminario. PySCF still supplies the electrostatically embedded large-model ESP,
and MCPB.py performs its normal RESP and bonded-parameter steps. After an
optional MCPB-reference comparison passes, mdprep continues to the final
solvation, `tleap`, and topology validation stages.

## Runtime requirements and failures

ASH may live in a separate Python environment. Set `ash.python_executable` to
that environment's Python interpreter. `ash.xtb_executable` must resolve to an
executable named `xtb` for GFN2-xTB and g-xTB, because that is the path
convention used by ASH's xTB interface. The ASH environment must also provide
OpenMM and geomeTRIC. MACE-POLAR-1 must be importable there; its checkpoint is
mandatory and SHA-256 pinned, inference uses float64, and license acceptance is
never inferred. `python_search_paths` may expose dependencies from another
reviewed environment, but an isolated ASH/MACE environment is preferable.

ASH normally rejects QM/MM link boundaries other than its standard C-C case.
Selecting complete coordinating amino-acid residues generally cuts the two
adjacent peptide C-N bonds. After reviewing the exact boundary atoms in the
preserved ASH output and refinement selection report, set
`ash.allow_unusual_link_boundaries: true` to authorize those link atoms. The
choice is written to `ash_refinement_input.json`; mdprep never enables it
implicitly.

Use `mdprep prepare system.yaml --stop-after refinement` to generate and inspect
the refined, post-protonation intermediate without running final ligand and
metal parameterization.

If a downstream audit fails after ASH has converged, rerun `mdprep prepare`
with `--resume`. At successful ASH completion, mdprep writes immutable cache
metadata containing the ASH payload, an exact SHA-256 of the provisional
`.inpcrd`, and a SHA-256 of the `.prmtop` scientific content. The topology
fingerprint excludes only Amber's wall-clock `%VERSION` timestamp. Cached ASH
output, stdout, stderr, and command-record hashes are also pinned. It reuses
optimized coordinates only when those artifacts and the regenerated scientific
inputs match; otherwise the resume fails explicitly. Legacy results without
immutable metadata are never reused. `--resume` and `--overwrite` are mutually
exclusive.

Refinement fails clearly when executables are unavailable, charge/spin data are
missing, region mapping changes residue identity or count, retained waters are
not protonated, ASH exits unsuccessfully, or optimized coordinates are
non-finite or do not match the provisional topology.
