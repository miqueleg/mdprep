# Changelog

## Unreleased

### Added

- The official Colab notebook, `examples/mdprep_colab.ipynb`: takes a bundle
  from the web builder (or runs the SyrB2 example), prepares and equilibrates
  the system, and shows every step -- the structure after each stage, the
  protonation states, the charge on every atom in tables and in 3D, the metal
  site's MCPB force constants, how temperature, density and volume
  equilibrated, the structure after each Roe-Brooks step, and the trajectory
  with its RMSD, RMSF and metal-site distances. The web builder links to it.
- `mdprep closest-xray`: the X-ray structures closest to a predicted model,
  from the RCSB sequence search, with resolution, waters and bound ligands;
  those holding the model's metal come first.
- `mdprep transplant-waters`: a reference structure's crystal waters on a
  predicted model. PyMOL pairs the residues (by sequence, or by structure for a
  distant homologue); each model chain takes its best reference chain; a water
  bound to a metal moves with it onto the model's metal; clashing and bulk
  waters are dropped; a reference of another fold is refused. Every decision
  is in a JSON report.
- `mdprep.analysis`: reads a prmtop, inpcrd and OpenMM DCD with numpy alone,
  and provides charge tables, the metal site's force constants, imaged and
  superposed trajectories, RMSD, RMSF and distances, and the equilibration and
  production logs.
- `equilibration.csv` in every Roe-Brooks run: every stage on one time axis,
  with temperature, density, box volume and energies every 0.5 ps.
- `docs/predicted_structures.md`: from a Boltz or AlphaFold3 prediction to a
  model with its crystal waters, with the worked SyrB2 example.
- `examples/17_syrb2_bound_water_mcpb.yaml` and the vendored 2FCT deposition.
- When a structure with a metal has no waters at all, the metal question says
  so and points to `mdprep transplant-waters`.

- Waters bound to a bonded metal site, as the resting iron of an
  alpha-ketoglutarate enzyme carries (SyrB2, PDB 2FCT). The decision engine
  proposes the water as a donor; mdprep writes the TIP3P template MCPB.py
  requires, supplies the H-O-H angle rigid TIP3P lacks, and counts the water as
  neutral in the QM/MM refinement.
- `structure.keep_crystal_ions` and its question: free monatomic ions resolved
  in the crystal (Cl-, Na+, Br-, ...) are kept or removed as a group, and
  listed in the structure report either way.

### Fixed

- A residue name mdprep assigned (a predicted iron called `LIG1` read as `FE`)
  lost that record when the structure was written and read back, and `FE` read
  cold is the CCD code for Fe(III): the planner then proposed +3 for an Fe(II)
  site. mdprep's PDB writer records assigned names in a remark its reader
  honours.
- The model-charge proposal ignored an oxidation state accepted as proposed,
  so a student added the terms up by hand.
- The old Colab run notebook ran the MD twice: `mdprep prepare` already runs it
  when the manifest enables it. It now points to the official notebook, which
  stops `prepare` before the MD.
- A free crystal chloride was offered as a ligand and then refused as a
  reserved residue name, telling the user to rename it. tleap builds it from
  its ion library like any counterion; it is no longer a ligand candidate, and
  a histidine xTB cluster that contains one counts its charge.
- A topology with a water and a halide on the same metal went to NaN in
  heating: their 1-4 pair through the metal had a Coulomb attraction and no
  repulsion. Those 1-4 terms are now switched off in the prmtop and listed in
  the tleap report.
- tleap bonded the two ends of an unresolved loop into one peptide bond, 6.08 A
  long in 2FCT. A TER is now written wherever C(i)-N(i+1) exceeds 2 A.
- The crystal-water and unknown-heterogen questions disappeared whenever any
  water coordinated any metal -- in 6NIE, three waters on a surface iron of
  chain B hid the question for all 847. Both are asked now, and the
  crystal-water question names the bound waters that are kept regardless.
- A water bound to a metal was dropped with the other crystal waters when
  those were removed: for a bonded site that left an MCPB bond to a residue
  that no longer existed, and for a nonbonded one it stripped the ion's first
  shell, though the question had said it would stay. The question and
  normalisation now share one definition of a bound water
  (`mdprep.structure.coordination`).
- Declining a ligand while keeping unknown heterogens, dropping crystal waters
  with a bonded site, or choosing OPC with a water bound to a bonded metal now
  fail in the builder with what to change, instead of later in Colab.
- The QM-region spin proposal ignored defaults the user had accepted without
  typing them, so a bonded iron with the default model got no proposal.

## 0.3.0 - 2026-10-19

### Added

- `mdprep plan` and `mdprep init --interactive`: a structure-aware decision
  engine that reduces a 250+ setting schema to the questions one structure
  actually needs, and asks them. Chemistry-sensitive values have no default and
  cannot be skipped.
- Each decision carries viewer hints (`focus`, `targets`, `target_resnames`) so
  a 3D front end can frame exactly the residues a question is about.
- `mdprep inspect` reports free monatomic metal ions.
- A static web input builder (`web/`) that runs mdprep in the browser under
  Pyodide, shows a 3D view framed on whatever each question is about, and
  produces a validated manifest plus runnable Colab cells.
- `mdprep.workflows.runtime.conda_packages_for` derives the external toolchain a
  given manifest needs, so a generated setup cell installs PropKa, xTB or PySCF
  only when the run will actually invoke them.

- `mdprep doctor` checks that the external tools a given manifest will invoke
  are installed, and names the one command that installs what is missing.
- mmCIF and ModelCIF input: a structure predicted by Boltz or AlphaFold is read
  directly, with no conversion step and no loss of the chain and residue
  identifiers the manifest selects on.
- Four-character residue names, hybrid-36 serial and residue numbers, and
  residues whose name does not fit a PDB column, which a predicted structure
  written to PDB reaches routinely. A name mdprep has to assign is recorded as
  assigned, so nothing downstream reads chemistry into a name it invented.
- `metals[].coordinating_ions` declares a non-metal ion bound in a metal site,
  such as a coordinating halide, which previously had nowhere to live: it is
  not a metal, and antechamber cannot treat a single atom.
- The setup cell downloads the g-xTB build of xtb from the Grimme group's
  releases and checks it against the published checksum, because conda-forge
  carries only the GFN-only build.

- `structure.keep_chains` restricts a build to named chains, which a bonded
  MCPB site requires and a multi-copy crystal structure needs.
- The decision engine proposes a metal coordination sphere from interatomic
  distances, requires it to be confirmed, and generates the bonded MCPB block
  with a GFN2 Seminario Hessian and GFN2 QM/MM refinement geometry.
- `examples/16_besd_halogenase_gfn2_mcpb.yaml`, the BesD halogenase reference
  case.
- Detection of incomplete residues and chain breaks, asked about before any
  expensive stage runs, and protection for waters coordinating a metal.
- `examples/mdprep_colab_run.ipynb`, a Colab notebook that runs a manifest and
  installs only the toolchain that manifest needs.

- `metals[].coordinating_ions` declares a non-metal ion bound in a metal site,
  such as a coordinating halide, which previously had nowhere to live: it is
  not a metal and antechamber cannot treat a single atom.

### Changed

- The refinement stage defaults to `backend: openmm`, a force-field hydrogen
  relaxation, instead of GFN2 QM/MM. The stage only ever moved hydrogens, and
  a force field settles them in seconds. `backend: ash` keeps the QM/MM route
  and is required to relax heavy atoms. `movable_atoms` and
  `post_refinement_protonation` default to match the hydrogen-only backend.
- Histidine HID/HIE selection stays with xTB: force-field energies of the two
  tautomers are not comparable, because they are different residue templates
  rather than one molecule with a proton moved.

- Dropped the unused `scipy` and `pandas` runtime dependencies.
- A ligand that coordinates a bonded metal site now defaults to taking its
  charges from that site's MCPB fit instead of AM1-BCC.
- A metal site's Hessian defaults to g-xTB rather than GFN2. GFN2 has no
  parameters for most transition metals, and said so only after the run had
  got as far as the Hessian.
- **A ligand whose residue name an Amber force field already defines is now
  refused, where it used to be renamed for the build.** `LYS` became `LY1`,
  which satisfied tleap and moved the problem: the prepared system then
  described a residue under a name the input never used, and every later
  reader had to know about the substitution. A manifest that declares such a
  ligand now fails, in the GUI before it asks anything and on the command line
  before any quantum chemistry, and says which residue to rename. The web
  builder offers to make the change in the input.
- **`refinement.total_qm_multiplicity` is now required whenever any component
  of the QM region is open-shell**, where it used to be taken from the first
  open-shell component found. For a lone high-spin Fe(II) that is the right
  answer and it was right by accident: it holds only while the ion is the
  single unpaired centre and nothing couples to it, which nothing in a
  manifest can check. Existing manifests that leave it out of an open-shell
  region will fail validation and need the value added.
- Preparing the same system the same way twice now yields the same atom count.
  MD starting velocities stay random and the seed drawn for each run is
  recorded, so a replicate can be repeated by putting that number back.

### Fixed

- Per-step MD snapshots are written in the box the manifest asked for. They
  carried the pre-NPT cell, because `PDBFile` takes CRYST1 from the topology
  and the topology was the one loaded at the start, and they were wrapped into
  the parallelepiped OpenMM spans rather than imaged about the solute -- so a
  truncated octahedron came back as a sheared prism with the protein off to
  one side and its water drawn 70 A away instead of 54 A.
- The ESP grid survives an xTB crash on its last point, and the fit uses the
  grid that was actually evaluated rather than the one that was requested.
- A metal in a predicted structure is named after its element, so a site is
  not identified by a name the prediction never meant as one.
- CONECT records that cover only part of a ligand are ignored rather than
  treated as its whole connectivity.
- tleap writes its own atom types into the element column of the PDB it
  outputs; the reader no longer believes them.
- The setup cell installs PDBFixer whenever a run will need it, not only when
  structure repair happens to pull it in, and the resolver is checked against
  every combination of options rather than the one that was reported.
- A titratable side chain coordinating a metal is asked about, from the same
  contact list the bonds come from, instead of refusing the run later.
- Guessed force-field parameters (`ATTN, need revision`) and the OpenMM energy
  check are both checked when the manifest asks for them.
- Many GUI fixes: the Finish button reports what went wrong instead of doing
  nothing, re-uploading a structure cannot leave the bundle describing the
  previous one, a file that parses to no atoms is rejected, Start over clears
  the answers, Enter fetches a PDB id, and an entry that is mmCIF-only is
  fetched as `.cif`.

## 0.2.1 - 2026-10-01

### Fixed

- A ligand whose residue name is one the Amber force fields already define is
  now built under a name they do not. `tleap` resolves residues by name and
  maps an amino-acid name to its terminal form first, so the BesD free-lysine
  substrate was built from the `NLYS` protein template and the run died with
  "Atom .R<NLYS 252>.A<OXT 25> does not have a type" for every one of its
  ligand-named atoms; binding the ligand unit to `LYS` instead would have
  pointed every lysine in the protein at the ligand's parameters. The leap
  input PDB, a copy of the fitted mol2, the `tleap` script and the final
  validation all use the new name, and every rename is reported.
- The temporary xTB hydrogen environment no longer hydrogenates a residue
  twice. OpenMM matches the hydrogens a residue already has through the bonds
  its residue templates define, so hydrogens mdprep placed from
  `ligands[].smiles`, named after their parent heavy atom, were invisible to it
  and it built a second complete set beside them: the BesD free-lysine
  substrate came out carrying both HB2/HB3 and HB/HBA. Such a residue is now
  passed through untouched and reported, unless a protonation state was decided
  for it, in which case OpenMM still rebuilds it in that state.
- Amber ion atom types were built as `element + charge + "+"`, which produced
  `Na1+` for every monovalent cation and could not express an anion at all.

- Roe--Brooks stage 2 no longer fails with "Particle coordinate is NaN" on
  solvated systems. Free ions were positionally restrained because Amber names
  them `Na+`/`Cl-`, which match no bare element symbol, and coordinates handed
  between stages were wrapped into the periodic box while the restraint
  references stayed unwrapped, placing every wrapped restrained particle a full
  box length from its reference. Stage hand-off and XML restarts now stay in
  the unwrapped frame; only the per-step PDBs and the trajectory are wrapped.
- Final-output validation no longer rejects its own PDB with "PDB atom serials
  must be unique when present" above 99,999 atoms. Serials are decoded for the
  column-12 overflow `tleap`/`ambpdb` write, for hybrid-36, and for `*****`; a
  genuine wrap-around is renumbered in file order with a warning, and stays
  fatal only when `CONECT` records make the connectivity ambiguous. The PDB
  writer uses the same convention it reads, so structures of that size
  round-trip unchanged.
- `mdprep inspect --json` no longer emits unparseable JSON on a narrow
  terminal, where rich reflowed long strings to the console width.

### Changed

- xTB histidine calculations pin `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` to the new `protonation.histidine.xtb.num_threads`
  (default `1`), with `OMP_STACKSIZE` from `omp_stacksize` (default `1G`).
  xTB otherwise opened one thread per hardware thread for every tautomer
  cluster, spending large amounts of CPU time without reducing wall time.
- `mdprep inspect` also reports free monatomic metal ions.

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
