# Changelog

## Unreleased

### Added

- A metal's bonds are a checklist in the web builder and `mdprep init
  --interactive`: every residue with a donor atom within 2.6 A is a box, all
  ticked, with its atoms and their distances. Unticking one -- a substrate
  docked too close to an iron that has no water of its own -- leaves it
  unbonded and fits the site without it: its bonds are left out of
  `mcpb.bonds`, MCPB.py's `cutoff_angstrom` drops just below its nearest
  atom, the bonds kept beyond that are handed to MCPB.py by name, and the
  proposed model charge no longer counts it. A ligand unticked while its
  charges still come from the site's fit is stopped at the checklist with
  what to change. It replaces the single yes/no on the whole shell; an
  answer given to that still reads as all ticked or none.
- `noncanonical_residues`: modified residues kept in their chain rather than
  replaced by their parent -- a phosphoserine with its phosphate, an
  acetyl-lysine, a p-azido-phenylalanine -- and bonded into it by tleap. One
  block per residue name, every residue named. `parameters: amber_library`
  builds them from a template AmberTools ships for the protein force field
  (SEP, S1P, TPO, T1P, PTR from phosaa14SB/phosaa19SB; ALY, MSE, AZF, CYF from
  ff14SB_modAA/ff19SB_modAA; HYP from the force field itself), the template's
  name being the protonation. `parameters: am1bcc` parameterises them from
  their SMILES, capped or as the free amino acid: the residue is cut out
  between an acetyl and a methylamide built from its neighbours' atoms, given
  the SMILES's hydrogens, AM1-BCC charges with the backbone's fixed to the
  force field's for a residue of its sign, the protein's atom types on the
  backbone and CB and GAFF2's beyond, parmchk2's estimates for the junction
  only, ff19SB's CMAP of its parent, and a library whose head and tail bond
  it into the chain. Every topology is read back: each kept residue is there,
  with its charge, and peptide-bonded on both sides. A residue report records
  how each was built. See [docs/noncanonical_residues.md](docs/noncanonical_residues.md)
  and `examples/22_spoiiaa_phosphoserine.yaml` (SpoIIAA, PDB 1H4X).
- `parameters: qmmesp_pyscf` for a kept residue: the same parameterisation,
  with its charges fitted by Amber's two-stage RESP to the PySCF ESP of the
  capped residue polarised by the rest of the system. The provisional system
  QMMESP ligands are fitted in -- which the ligand stage now builds whenever a
  ligand or a residue asks for QMMESP -- holds the residue with its AM1-BCC
  parameters, written apart under `residues/<NAME>/provisional_am1bcc/`. Its
  MM point charges leave out the residue, the two residues the caps stand in
  for, and the peptide bonds joining those to the chain beyond; an MM atom
  within 1.0 A of the QM model stops the fit as a clash. RESP holds the
  residue's backbone N, H, C and O at the force field's charges and each cap
  at zero, through respgen's predefined charges and charge groups; only the
  residue's own charges are kept, in a library with the provisional one's
  types and frcmod, and the final build takes only that one. Any charge more
  than 0.5 e from its AM1-BCC value is reported. A kept residue counts as
  protein in any QMMESP embedding, a ligand's included. The planner and the
  web builder offer it beside AM1-BCC, written with HF/6-31G*, and the Colab
  setup then installs PySCF -- and RDKit for any residue parameterised from
  its SMILES, which it did not before.
- A kept residue may coordinate a nonbonded metal; with QMMESP the metal's
  formal charge is among those that polarise the fit.
  `examples/23_osteocalcin_gla_calcium.yaml` keeps osteocalcin's three
  gamma-carboxyglutamates on its three Ca2+ (PDB 1Q8H, shipped as a
  tutorial), and the SMILES and charge of Gla (CGU) are proposed.
- Bonded MCPB sites and QM/MM refinement in builds that keep residues,
  including a kept residue on the metal. MCPB.py tells an amino acid by its
  atoms, so it models the residue as one -- side chain in the small model,
  backbone capped from its neighbours in the large -- reading its types and
  charges from a mol2 mdprep stages for every kept residue: the residue's own
  for one parameterised from its SMILES, or the template's unit written out
  by tleap with its family's frcmod. MCPB.py's rename of the residue on the
  metal is carried through: its template and peptide bonds reach the final
  build, every other residue of the name keeps its library (no unit of the
  residue's own name is taken from MCPB.py's commands), and the topology
  checks count it under the new name with its refitted charge reported. The
  planner counts a kept residue's charge in the model charge it proposes,
  and refusals now cover only QMMESP charges for a residue on a bonded metal,
  which MCPB.py refits. The pre-MCPB hydrogenation build loads the residues'
  libraries, and the refinement's provisional system looks for them under the
  project's output directory rather than its own; its QM region counts a kept
  residue as protein, at its block's charge. A residue fitted with QMMESP after a refinement keeps the
  refinement's hydrogens.
- A metal deposited inside a modified residue of a chain -- a cysteine and its
  zinc as one residue -- is read the way a heme's iron is: moved into a
  residue of its own, a metal site, and written after the chain rather than
  inside it, where tleap would have ended the chain. A residue whose metal is
  bonded to more than the amino acid, as S-(methylmercury)cysteine's is, is
  left whole as an organometallic group and can only be replaced by its
  parent. Replaced, a residue on a bonded metal has its parent's protonation
  asked, as a cysteine's on a zinc is.
- `run_amber_resp_fit` takes charges to hold fixed and groups of atoms whose
  charges sum to a given value, handed to respgen as its additional input and
  checked in the fitted charges.
- The planner, and so the web builder, asks one question per modified residue
  name: replace it, keep it with AmberTools' template, or keep it
  parameterised from its SMILES with AM1-BCC or QMMESP charges, offering what
  applies. Kept, the template's protonation, or the SMILES and charge, follow;
  for the common modifications they are proposed, with the reason for the
  protonation.
- The xTB histidine embedding takes a kept residue's charges, its template's
  or those mdprep fitted, instead of leaving it out; a residue fitted with
  QMMESP takes part with its provisional AM1-BCC ones.
- A residue parameterised from its SMILES may hold only what GAFF2 types: H,
  C, N, O, F, P, S, Cl, Br and I. antechamber types a selenium as GAFF2's
  thioether sulfur without complaint, which would build a selenomethionine
  with sulfur's mass and radius; it is refused with the atom named, and MSE
  is kept only from AmberTools' template.

### Changed

- `structure.replace_modified_residues`, a yes/no question for all modified
  residues together, is replaced by `residue.<NAME>.treatment`. Keeping a
  modified residue no longer stops the build, and one with no known parent
  inside its chain is no longer refused: it can be kept.
- A modified residue of a chain that the manifest neither keeps, replaces nor
  removes stops the build as the structure is read, even with
  `remove_unknown_heterogens: true`. It used to be removed with the unknown
  heterogens, which cut its chain in two.
- The web builder and the official Colab notebook install mdprep 0.5.1, so a
  structure that arrives with hydrogens and a docked ligand with only its
  polar ones reach the students as they now prepare.

### Fixed

- Choosing gas-phase RESP or QMMESP for a ligand in the planner or the web
  builder wrote a manifest without the `qmmesp` block those methods require,
  so it failed validation at Finish. The block is written, with HF/6-31G*.
- The RESP fit read antechamber's AC file by splitting on spaces, so a
  four-letter atom name, written run together with the residue name
  (`OE11CGU`), failed the atom-order check. The fixed columns are read.
- RDKit 2026.03, which the Colab runtime installs, left some of the
  hydrogens it places from a SMILES at the origin when the molecule had not
  been sanitised first: one or two on a saturated ligand (ethanol, methanol,
  isopropanol, propane, an alkylamine), none on one with a double bond or a
  ring. sqm then failed to converge, or a build could have carried them. The molecule is
  now sanitised before they are placed, and every placed hydrogen must sit
  at a bond's length from an atom of its own or the run stops.
- A ligand written after the crystal waters -- a docked substrate appended to
  the end of the file -- stopped the pre-MCPB tleap step as an unexpected
  rename. tleap writes every water after everything else, and its output is
  matched back by position, so it is now matched in that order. Its input
  keeps the file's order, which MCPB.py's bond commands count residues in.
- A noncanonical residue kept from its SMILES whose conformation is strained
  stopped the build in RDKit, which read its bonds from its coordinates, with
  the SMILES blamed. 1H4X's chain A phosphoserine, in the alternate location
  the file lists first, has P and CB both on OG and 2.04 A apart: read as a
  three-membered ring. Such a residue is now named, with its atoms and their
  alternate location, at the planner's question, which refuses the SMILES
  routes while its chain is built -- AmberTools' template, which bonds by
  name, still builds it -- and a hand-written manifest stops with the same
  reason before RDKit runs. Likewise a copy that cannot be kept, at the end
  of its chain, rules out keeping only while its chain is built: the
  question offers keeping when the other chains allow it, and names them.

## 0.5.1 - 2026-10-26

### Added

- `ligands[].keep_input_hydrogens`: what to do with input hydrogens that
  disagree with the ligand's SMILES. Unset, they are replaced and reported;
  `true` makes the disagreement an error (a `user_mol2` implies it); `false`
  replaces them without comparing.
- The structure report's input-hydrogen record (how many were removed from
  protein, water and each other residue, and which ligands kept theirs) and
  its ligand-hydrogen record (placed from the SMILES, kept because they agree
  with it, or replaced, with the atoms that disagreed). The builder's findings
  list the removed hydrogens too.

### Changed

- A ligand's SMILES is used whether or not the input has hydrogens on it.
  Hydrogens it has are compared with the SMILES heavy atom by heavy atom and
  kept only if they agree; otherwise they are replaced by the SMILES's and the
  disagreeing atoms are reported (`LIGAND_HYDROGENS_REPLACED`). A ligand with
  any hydrogens at all used to be left as it was and its SMILES ignored, and
  the builder did not ask for one. In consequence a ligand that arrives with
  hydrogens needs a SMILES, unless a `user_mol2` describes it: one without is
  refused as the structure is read, as is one with neither hydrogens nor a
  SMILES (a hydrogen-free molecule can be declared with `expected_formula`,
  as before, or with a SMILES that gives it none). The 7E07 tutorial's
  ligands carry their SMILES now.
- Every ligand whose hydrogens come from or are checked against a SMILES has
  its electrons counted against its net charge and multiplicity as the
  structure is read; a count the multiplicity cannot have stops the run.
- The web builder and the official Colab notebook install a pinned mdprep
  release (0.5.0) instead of git master, so everyone in a workshop runs the
  same code and the manifest the builder writes is read by the version that
  wrote it. CONTRIBUTING's release steps raise the pin after the tag has
  published, and a test checks that the page and the notebook name the same
  release.

### Fixed

- A structure that arrives with hydrogens -- a deposition refined with riding
  hydrogens, or a model run through PyMOL, reduce, PDBFixer or a format converter --
  no longer stops the run. `structure.remove_input_hydrogens` used to act only
  at the end of protonation, so the hydrogens reached everything before it:
  PyMOL and format converters protonate chlorides, and the builder refused the HCl as
  a ligand named after an Amber ion; reduce numbers every hydrogen it adds 0,
  and the reader refused the file for its CONECT records; a lysine's riding
  HZ3 stopped the temporary hydrogenation for xTB, which built the neutral
  lysine PropKa asked for; and PDBFixer's alpha-ketoglutarate, protonated as
  the neutral acid, was kept and gave MCPB negative angle force constants.
  The hydrogens are now removed as the structure is read, in the builder,
  `mdprep plan`, `mdprep init` and the structure stage alike, so the plan is
  made about the structure that is built and PDBFixer repairs that structure
  too. A ligand's are compared with its SMILES (see Changed).
- Structure repair lost atoms without saying so when several in a residue
  shared a name: PDBFixer keeps one of them, and the check that no atom went
  missing looked atoms up by name. A format converter names every hydrogen H,
  so six of an alpha-ketoglutarate's went in and one came out. A ligand's
  hydrogens now go round PDBFixer, which has nothing to do with them, and any
  other merge stops the repair with the atoms it affected.
- Atoms written more than once at partial occupancy without altloc labels --
  a format converter keeps both conformers of a residue and drops the A and B,
  leaving two OD2s in an aspartate at 0.50 each -- are read as the alternate
  locations they are: labelled in file order, chosen by the altloc policy, and
  reported. PDBFixer used to keep one of each pair without saying which.
- A docked ligand carrying only its polar hydrogens stopped antechamber with
  an odd number of electrons, after the protonation stage had run: having
  hydrogens at all made the builder skip its SMILES question and the
  preparation keep them. It now gets the SMILES's hydrogens, and a manifest
  without its SMILES is refused at once with the reason.
- CONECT records in a file whose atom serials repeat are followed to the
  atoms' new numbers when the serial they name is unique, and dropped with a
  warning when it is not, instead of the file being refused.
- The temporary hydrogenation for the xTB histidine comparison stops only
  when a heavy atom goes missing. A hydrogen the residue had and its assigned
  state does not is dropped, as the hydrogens are temporary anyway.
- One test of what a hydrogen is, the element the reader settled on, instead
  of five that disagreed. A mercury ion written without an element column was
  a hydrogen to two of them, and deuterium was one to only two.
- A four-character atom name starting with H in a residue without element
  columns (HG21, HO2') is read as a hydrogen, not as mercury or holmium.
- The official notebook's 3D views hid the metal site they were drawn to
  show: an Fe/alpha-KG enzyme holds its iron inside a beta-barrel, and the
  opaque cartoon covered it. With a metal present the cartoon is now
  translucent, every residue within the site radius is drawn as sticks (it
  was only those within 3 A), and the view is framed on the site. Section
  H's `SITE_RADIUS` reached the atoms it selected but not the drawing, so
  raising it changed nothing visible; it now does both.

## 0.5.0 - 2026-10-23

### Added

- The web builder's **Load SyrB2 (PDB 2FCT)** button: the workshop's worked
  example, for a student with no structure of their own, fetched from RCSB as
  a typed ID would be.
- `mdprep.workflows.runtime.full_toolchain_manifest()`: a manifest that calls
  every program mdprep installs, for an environment built before the real
  manifest exists.
- Hemes and other metals deposited inside a cofactor. The PDB writes heme b as
  one residue, HEM, with its iron among its atoms; mdprep now moves the metal
  into a residue of its own as any structure is read, named after the element
  with the cofactor's chain and number, and the porphyrin left behind is a
  ligand that coordinates it. A cytochrome P450 becomes an ordinary bonded
  iron site: the four porphyrin nitrogens, the cysteine and the water. The
  oxidation state is asked, not read off the assigned name. Heme b's SMILES
  and net charge (-4, both propionates ionised) are proposed with the
  assumption stated. See `docs/metals.md`.
- `examples/19_cyp119_heme_mcpb.yaml` and the vendored 1IO7 deposition:
  CYP119 at rest, low-spin Fe(III).
- `mcpb.pyscf.xtb.electronic_temperature_kelvin`: Fermi smearing for the xTB
  Hessian and the GFN2 ESP of the large model. An open-shell heme's SCF does
  not converge at xTB's own 300 K; the builder sets 1000 K for a site whose
  donors include a heme, and an SCF failure without it names the setting.
- A residue still holding several metal atoms (an iron-sulfur cluster) and
  heme c, a covalent ligand, are refused with the reason before any question
  is asked.
- Heterogens that are not ligands. mdprep reads the bonds a deposition
  declares (PDB LINK, mmCIF `_struct_conn`) and its modified residues (MODRES,
  `_pdbx_struct_mod_residue`), and sorts the heterogens before asking about
  ligands: modified residues in a chain, glycans, other heterogens bonded to
  the protein, and crystallisation additives. Each group is one question with
  every member listed. Glycans and additives are proposed for removal, a
  selenomethionine for replacement by methionine; anything else is asked
  without a proposal, and keeping what cannot be built stops at the question.
  See `docs/heterogens.md`.
- `structure.remove_residues`: residues left out on purpose, each with its
  reason, applied before anything else looks at the structure and reported.
- `structure.repair.replace_modified_residues`: modified residues replaced by
  their standard parents through PDBFixer, exactly the ones listed.
- A metal that is not inside a cofactor can be removed on its model question
  (a nickel from a His-tag purification). Magnesium and calcium are proposed
  nonbonded.
- The SMILES and net charge of FAD, FMN, SAM, SAH, NAD+, NADH, NADP+ and NADPH
  are proposed, at pH 7, with the protonation stated; each is tested against
  a deposited cofactor.
- `examples/20_upo_5oxu_heme_glycans.yaml` (an unspecific peroxygenase:
  heme-thiolate, magnesium, N-glycans, additives) and
  `examples/21_prna_fad_halogenase.yaml` (a flavin-dependent halogenase),
  with their depositions. `docs/enzyme_families.md` lists what mdprep does
  with each family of halogenating and dehalogenating enzymes.
- Refused when the structure is loaded, with the reason: a cofactor bonded to
  the protein that holds a metal (myeloperoxidase's heme), an oxoanion such as
  vanadate, a modified residue with no standard parent, and a declared bond
  between protein residues that is neither a peptide bond nor a disulfide.

### Changed

- The spin is asked once, for the QM region. The per-metal multiplicity
  question is gone; the QM region's answer is also the bonded site's
  `ions[].multiplicity` and MCPB model spins.
- Crystallisation additives are no longer offered one by one as ligands
  unless their removal is declined.
- The official Colab notebook takes its structure only from the builder's
  bundle. Its predicted-model section (closest X-ray structure, crystal
  waters), the SyrB2 example and the upload of an earlier run's results are
  gone; `mdprep closest-xray` and `mdprep transplant-waters` still bring
  waters across on the command line. It installs every program mdprep runs
  first, before there is a bundle, so the install runs while the builder's
  questions are answered; the MD follows the preparation directly, and the
  investigation of both comes after. Sections, old to new: C to A, D to C,
  F to D, G to F, H to G, I to H, J to I; B and E1-E8 keep their letters.

- The protonation of a titratable side chain on a metal is asked right after
  the coordination shell is confirmed, before the spin and the model charge.
- Proposals carry their own explanation, shown by the builder and the
  `mdprep init --interactive` wizard alike.

### Fixed

- A metal-bound cysteine stated as CYM stopped the protonation stage: OpenMM
  calls the thiolate's hydrogen variant CYX. Every cysteine donor -- a zinc
  finger's, a heme's -- reached this.
- The proposed model charge read a donor's charge from its residue name, so a
  cysteine stated as CYM counted as neutral and an aspartate stated as ASH as
  -1. It now counts the state given, and proposes nothing until it is given.
- The builder named a residue once per highlighted atom.
- The ligand bond-order check measured conjugated single bonds against
  saturated lengths, so every amide, amino-heterocycle and flavin looked like
  a wrong tautomer and a SMILES for FAD, NAD or SAM could not be applied.
- A heme on both a bonded iron and a nonbonded magnesium, as in a
  peroxygenase, was refused for taking its charges from the magnesium's
  non-existent fit.
- QM/MM refinement stopped on a nonbonded metal or a crystal ion next to the
  QM region; both are MM environment with known parameters.
- A lysine PropKa made neutral lost a hydrogen's charge in the embedding of
  the xTB histidine comparison: OpenMM names its amine hydrogens HZ1 and HZ2,
  ff14SB's LYN HZ2 and HZ3. The histidines around it were compared beside a
  lysine at -0.39 e, and the report asked for the lysine to be declared as a
  ligand. PrnA's Lys79 reached this: its four histidines keep their states,
  but their HID-HIE gaps moved by 1 to 3 kcal/mol, enough to flip a closer
  call.
- A vanadate was split into a vanadium and four unbonded oxygens.
- Replacing a modified residue stopped the run unless
  `structure.repair.add_missing_heavy_atoms` was set: PDBFixer keeps a blank
  insertion code as a space, so the replaced residue's own new atoms counted
  as unrequested. Any replacement in a structure missing nothing else reached
  this, 1CPO's pyroglutamate among them. A chain's terminal OXT, which the
  planner never asks about, stopped such a run too; it is left to tleap.
- A pyroglutamate replaced by glutamine kept its lactam's CD 1.5 A from its
  own backbone nitrogen. A replaced side chain bonded to its own backbone is
  rebuilt beyond CB from the parent's template, listed under
  `side_chain_rebuilt`, with a warning naming its closest contact.
- The g-xTB Hessian of a high-spin heme site could fail to converge, or
  converge in whichever of two states the thread count led it to: on 2QFK's
  aquo Fe(III), 27 kcal/mol apart, with Fe-N force constants up to 1.7 times
  different. `mcpb.pyscf.xtb.scf_annealing_kelvin` converges the SCF straight
  and from each hotter temperature listed, single-threaded, and takes the
  Hessian from the lowest; the builder sets it, with `max_scf_iterations:
  3000`, for every heme site. Examples 19 and 20 carry both. 2QFK itself now
  stops every time at the same place, a negative His-Fe-N angle constant,
  instead of passing or failing by the thread count. See `docs/metals.md`.
- A Seminario term MCPB.py found negative and wrote as zero was reported only
  as a 0.0 in the frcmod; the error now names it, its atoms and its value.
- The bond-order check read a porphyrin bond by bond, so 2QFK's heme, refined
  with one pyrrole's beta-beta bond at single-bond length, was refused as a
  wrong tautomer of heme b. A porphyrin's core is skipped like a carboxylate;
  a chlorin's reduced ring is still checked.

## 0.4.0 - 2026-10-20

### Added

- Dimers, trimers and tetramers. Every protein chain can be prepared in one
  system, and when the chains are copies of one protein the builder asks
  whether every copy should get the same parameters. If so it asks about the
  first copy only and gives each answer to the matching residue of the others,
  matched by sequence and by position rather than by residue number, which
  copies often do not share (2FCT numbers its two alpha-ketoglutarates 897 and
  898). A ligand is parameterised once; a bonded metal site is fitted once with
  MCPB.py, and each copy's residues take the fit's templates and its metal the
  fit's bonds. A copy whose site differs -- 6NIE's chains B and C bind
  alpha-ketoglutarate through the other carboxylate oxygen -- is reported and
  asked about on its own. See `docs/oligomers.md`.
- `parameters_from` on ligands and metal sites: a block built with another
  block's parameters, checked by `config-check` to repeat its settings and, for
  a bonded site, to list its residues in the reference's order.
- `examples/18_syrb2_dimer_shared_mcpb.yaml`: both copies of SyrB2, one fit.
- `mdprep.analysis.charges.metal_sites`: each metal site's charge on its own,
  which the official notebook now shows one table per site.
- The web builder lists which chains are copies of one protein, shows the
  chain and copy questions with each chain in its own colour, and names
  residues with their chain.
- Refusals the answers already make certain are given at the question that
  causes them, in the web builder and in `mdprep init --interactive`: a second
  bonded metal site needing a fit of its own, or a cofactor whose charges were
  to come from a site made nonbonded. Every `DecisionError` names the question
  to go back to (`decision_id`), and the page goes there.

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

### Changed

- **Free crystal ions are kept by default.** A free monatomic ion in the input
  used to count as an unknown heterogen, so a manifest with
  `structure.remove_unknown_heterogens: true` dropped it. It is now governed by
  `structure.keep_crystal_ions`, which defaults to `true`, so the same manifest
  builds with the ion in place: 2FCT's free chloride, for one. Add
  `keep_crystal_ions: false` to keep the old result. Every ion kept or removed
  is listed in `reports/structure_report.md`.
- Two bonded metal sites are refused by `mdprep config-check`, where 0.3.0
  refused them only once a run had started.

### Fixed

- CUDA did not start on Colab's T4 GPUs (OpenCL did):
  - conda-forge installed a CUDA runtime newer than the GPU driver, whose
    kernels the driver refused (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`);
  - the setup the notebook and the web builder run now caps `cuda-version`
    at the driver's version, and rerunning section C repairs an existing
    environment;
  - `mdprep doctor` starts a CUDA system to check;
  - a CUDA failure in the MD says what to do, and `auto` says which
    platform it passed over.
- The official notebook drew the final structure (E3) from tleap's PDB, which
  has no elements, so the viewer read hydrogens such as HG21 as mercury and
  bonded them to everything near. It is now drawn from the topology, as E8
  and H are. Each protein molecule is now its own chain in every view, so no
  cartoon runs from one copy of a dimer into the next or across a missing
  loop. A chloride on the iron, one atom with no bond to draw, now shows as a
  sphere.
- Keeping one chain of a predicted oligomer kept every ligand chain too, so the
  other copy's iron and cofactor floated in the box. A chain holding only
  heterogens now goes with the protein chain it sits against.
- The spin of the QM region was asked of any structure with two metals, even
  when neither turned out bonded; it is now asked once a bonded site is
  confirmed, whichever site that is.
- `DecisionPlan.applicable` counted an answer whose own question was no longer
  reachable -- a shell confirmed before the site was switched to nonbonded.
- The ligand coordinate check after tleap could not tell two copies of one
  cofactor apart, since tleap renumbers residues; it now takes the copy nearest
  the ligand it checks.
- The web builder and the wizard told a student to add a SMILES to a ligand
  without hydrogens after asking for it and being given it.
- `mdprep init --interactive` re-asked a rejected answer without saying why.

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
