# Developer Notes

## Architecture

The workflow is stage-based:

1. config loading and validation
2. structure inspection/normalization
3. protonation assignment
4. optional provisional AM1-BCC/ASH active-site refinement and a second
   protonation assignment on the optimized geometry
5. ligand extraction and final parameterization
6. explicit metal-site preparation
7. final `tleap` build
8. validation and reports

Each stage writes machine-readable reports and should fail before silently
changing unsupported chemistry.

## Future Noncanonical Residues

Add noncanonical residue support at the structure classification and `tleap`
template-loading boundary. Do not treat noncanonical polymer residues as simple
independent ligands.

## Metal Centers

Metal logic lives under `mdprep.metals`. Nonbonded sites load exact Amber
Li/Merz parameters and 12-6-4 sites receive a verified C4 post-processing
step. Bonded sites use a preserved, staged MCPB.py workflow. Keep metal
identity, oxidation state, parameter family, coordinator bonds, model charges,
spins, and QM artifacts explicit. Distance searches may validate the declared
coordination sphere but must not choose it.

The protonation stage runs before MCPB model generation. Atom-specific
histidine constraints may assign HID/HIE; other titratable donors require an
explicit manual state. MCPB must consume a `tleap`-hydrogenated structure, and
all generated residue renames must be mapped back and reported.

## Adding A Ligand Charge Method

Add a manifest enum value, extraction/parameter workflow branch, mol2
validation, reports, examples, and tests. Preserve atom names, atom order,
coordinates, residue identity, and total charge unless the manifest explicitly
allows a change.

## External Tools

All external commands must use `mdprep.external.runner` and preserve command,
working directory, return code, stdout, stderr, and runtime.

## QMMESP Correctness

MM point charges enter the target-ligand SCF Hamiltonian through
`pyscf.qmmm.mm_charge` and polarize only the target-ligand density. Fit only
target-ligand atom charges. Never include environment point charges as fitted
centers or write them to ligand mol2 files.

Configured metal ions enter the provisional QMMESP topology with explicit
formal charges and a selected nonbonded parameter family. Never fit those
environment charges into the target ligand. A ligand cannot simultaneously be
a QMMESP target and an MCPB-refitted site residue.

## Active-site refinement

Refinement logic lives under `mdprep.refinement`. Keep manifest component ids,
metal coordinators, charge derivation, spin-coupling choices, residue-distance
selection, and coordinate mapping deterministic and auditable. ASH runs in the
user-selected Python environment through `mdprep.external.runner`; do not
import ASH into the main process. Provisional AM1-BCC charges must never replace
a requested final RESP/QMMESP/user charge model.

The generated ASH driver supports electrostatically embedded GFN2-xTB and
mechanically embedded g-xTB or MACE-POLAR-1. Mechanical runs must never be
reported as receiving MM point charges. MACE inference is float64,
license-gated, and checkpoint-hash pinned. Keep method selection out of region
construction so controlled comparisons preserve identical atom lists, charge,
multiplicity, protonation, and 4/8 Å cutoffs.
MACE and xTB refinements retain the same default ASH/ORCA geomeTRIC
convergence criteria so method comparisons are controlled.
Pass explicitly selected metal-ion atom indices to ASH's
`excludeboundaryatomlist`; never infer or add link atoms between a selected
metal and nearby MM waters during a restart.

For `mcpb.pyscf.geometry_source: qmmm_refinement`, pass the in-memory completed
`RefinementResult` through the preparation and metal stages. Do not infer
completion from a stale PDB on disk. The MCPB step-1 small model inherits the
relaxed coordinates; record both the full refinement PDB and the generated
small-model hashes.

GFN2-xTB and g-xTB Hessians share the validated runner in
`mdprep.metals.gxtb_hessian`; model-specific flags and output markers remain
explicit. MACE-POLAR-1 runs in an isolated configured environment through
`mdprep.metals.mace_polar1_hessian` and the self-contained
`mace_polar1_worker.py`; never import its optional PyTorch/MACE stack into the
mdprep process. Preserve float64 inference, exact charge/multiplicity,
checkpoint and interpreter hashes, and the independent force finite-difference
check. MCPB-reference comparison logic lives in
`mdprep.metals.parameter_comparison` and must compare only MCPB-labeled
Seminario BOND/ANGL terms. It must not treat inherited GAFF/protein terms as QM
validation data. Production MCPB completion must reject non-positive Seminario
force constants. Diagnostic comparisons may parse them only to preserve and
report the failed terms explicitly.

Seminario steps are launched through `mdprep.metals.mcpb_compat`. The adapter
addresses only MCPB.py's inability to round an exactly-real NumPy complex
scalar produced by conjugate eigenpairs. Keep its imaginary-residual tolerance
strict and fail rather than discarding a material imaginary component. All
other MCPB steps continue to execute directly through the external runner.
