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

## What the wizard will not write for you

A bonded MCPB metal site needs external QM artifacts that cannot be invented
from a structure. Choosing `bonded_mcpb` records the decision, leaves the site
out of the generated file, and prints a note; add the block by hand following
[metals.md](metals.md).

The RESP charge methods likewise add a `qmmesp` block you must complete. See
[ligands.md](ligands.md) and [qmmesp_pyscf.md](qmmesp_pyscf.md).

## Building another front end

The decision plan is the contract. `mdprep plan --json` emits it in full:

```bash
mdprep plan input.pdb --json
```

Each decision carries a stable `id`, the `question`, its `kind`
(`choice`, `integer`, `number`, `text`, `boolean`), the allowed `choices`, the
`default` (`null` when there is none), `requires_user_input`, the `evidence`
from the structure, the `target` residue where one applies, and a `when` list
naming the earlier answers that make it relevant.

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
