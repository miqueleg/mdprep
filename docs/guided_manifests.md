# Guided Manifest Generation

A manifest exposes several hundred settings, but a given structure only needs a
few dozen of them, and only a handful are decisions a researcher has to make.
`mdprep plan` works out which, and `mdprep init --interactive` asks them.

On a solvated protein--ligand complex this is the difference between 250+
available settings and **19 questions, 3 of which need real thought**.

## See what a structure requires

```bash
mdprep plan input.pdb
```

This reports what mdprep found (chains, ligands, metal ions, histidines,
possible disulfides) and then every decision that follows from it, marking the
ones that cannot be defaulted and which earlier answer each one depends on.

## Answer the questions and write a manifest

```bash
mdprep init input.pdb --interactive -o system.yaml
```

Each question shows what was found, why the setting matters, and the available
values. Press Enter to accept the value shown; there is no value to accept for
a chemistry-sensitive question, so it is asked until you answer it.

The manifest is validated against the full schema before it is written, so a
file produced this way always passes `mdprep config-check`. Nothing is
generated when the wizard is interrupted.

The non-interactive `mdprep init` is unchanged and still produces the commented
starter manifest.

## What is never defaulted

mdprep does not guess chemistry, and the wizard must not become a way around
that. These have no suggested value, and a front end cannot skip them:

- **Ligand net charge.** It cannot be read from a PDB file, and a wrong value
  produces a plausible-looking topology with the wrong electrostatics.
- **Metal site model and oxidation state.** Neither the residue name nor the
  element implies an oxidation state. Zn is almost always +2; Fe and Cu are not.
- **Removing unknown heterogens.** Normalization must never discard a residue
  that was not explicitly configured, so the question names every residue at
  stake.
- **Production length**, when MD is enabled. There is no sensible default.

Free monatomic metals are offered as metal sites, never as GAFF ligands. Bulk
counterions are treated as solvent and are not asked about at all.

## Metal sites

Choosing `bonded_mcpb` proposes a coordination sphere from interatomic
distances: atoms of a donor element (N, O, S, P, Se, halide) within 2.6 A of
the metal. Carbon and hydrogen are excluded, which is what keeps the carbons
sitting behind a coordinating carboxylate oxygen out of the shell.

The proposal is **not** a decision. You are shown every candidate bond with its
measured distance and must confirm the set before anything is written; decline,
and the site is reported for manual work rather than invented. The oxidation
state, the spin multiplicity and the capped-model charge have no defaults
either, because none of them can be read off a structure.

A ligand that coordinates the metal takes its charges from that site's MCPB fit
rather than an independent AM1-BCC calculation, since fitting a bonded donor
separately leaves the site electrostatically inconsistent. A ligand that does
not coordinate keeps AM1-BCC.

The Hessian defaults to GFN2, which is orders of magnitude cheaper than DFT and
is what makes a bonded site tractable outside a cluster; g-xTB and PySCF DFT
are offered too. The geometry comes from the QM/MM refinement stage, which is
enabled automatically for a bonded site, so nothing has to be optimised by hand
first.

[examples/16_besd_halogenase_gfn2_mcpb.yaml](../examples/16_besd_halogenase_gfn2_mcpb.yaml)
is the worked reference: BesD, an alpha-ketoglutarate-dependent halogenase with
Fe(II) in a 2-His triad, chloride in the carboxylate position, and a lysine
substrate.

## What the wizard will not write for you

A bonded site whose proposed coordination shell you decline is left out of the
generated file with a note; add it by hand following [metals.md](metals.md).

The RESP charge methods likewise add a `qmmesp` block you must complete. See
[ligands.md](ligands.md) and [qmmesp_pyscf.md](qmmesp_pyscf.md).

## The web input builder

[`web/`](../web/README.md) is a static page that does all of the above in a
browser, with a 3D viewer that frames the residues each question is about. It
runs this same package under Pyodide rather than reimplementing any of it, and
hands you a manifest plus the Colab cell that runs it.

## Building another front end

The decision plan is the contract. `mdprep plan --json` emits it in full:

```bash
mdprep plan input.pdb --json
```

Each decision carries a stable `id`, the `question`, its `kind`
(`choice`, `integer`, `number`, `text`, `boolean`), the allowed `choices`, the
`default` (`null` when there is none), `requires_user_input`, the `evidence`
from the structure, and a `when` list naming the earlier answers that make it
relevant. A decision is asked when every condition in `when` holds and, if its
`when_any` list is not empty, at least one condition there does too; an answer
counts only while its own question is still asked.

In a structure with several copies of one protein, a copy's questions carry
`copy_of`: the id of the same question about the counterpart residue in the
first copy. They are skipped when the copies share parameters
(`structure.same_parametrization`), and `build_manifest` gives them that
question's answer. A front end needs to do nothing about this beyond
honouring `when_any`; see [oligomers.md](oligomers.md).

### Driving a 3D viewer

Each decision also says what a viewer should put on screen while it is being
answered:

- `focus` is `none` (no structural aspect, such as a project name),
  `targets` (emphasise and frame the residues below), `structure` (show the
  whole system, as for the force field or the solvent box), or `chains` (the
  whole system with each chain in its own colour and the residues below
  emphasised in every copy, as for which chains to keep and whether copies
  match).
- `targets` lists the residues to highlight, each with `chain`, `resname`,
  `resid`, `icode`, and `element`/`atom_name` where they apply. **`targets[0]`
  is the subject of the decision** -- the ligand being charged, the ion being
  modelled -- and any later entries are surrounding context.
- `target_resnames` stands in for a set too large to enumerate usefully, such
  as every crystallographic water; a viewer selects on the names directly.

So a ligand charge question frames that ligand, a disulfide question frames
both cysteines of every candidate pair, the unknown-heterogen question frames
every residue at stake, and a metal question frames the ion together with
whatever lies within 3 A of it.

That contact search exists only to frame the view. It is **not** a
determination of coordination: mdprep requires coordinating bonds to be
declared explicitly, and the evidence string says "within 3.0 A" rather than
claiming a coordination shell.

A web form, a notebook or another CLI consumes that JSON and never needs to
re-implement any of the logic. In Python:

```python
from mdprep.config.decisions import plan_decisions, build_manifest

plan = plan_decisions("input.pdb")
for decision in plan.applicable(answers_so_far):
    ...  # render it however you like

manifest = build_manifest(plan, answers)  # raises if a blocking answer is missing
```

`build_manifest` returns a plain dictionary, and raises `DecisionError` rather
than defaulting anything a user was supposed to decide. That guarantee is what
makes it safe to put a graphical front end in front of mdprep at all.

A `DecisionError` carries `decision_id`, the question whose answer would
resolve it, when there is one; the web builder takes the user back to it.
Some refusals are certain before the last question -- a second bonded metal
site that would need an MCPB fit of its own, or a cofactor whose charges were
to come from a site that is no longer bonded. `answer_conflicts(plan,
answers)` returns that refusal as soon as the answers given make it certain,
so a front end can say so at the question that caused it, as the web builder
and `mdprep init --interactive` do.
