# mdprep input builder (static site)

A browser front end for `mdprep plan` and `mdprep init --interactive`. Load a
structure, answer only the decisions that structure actually needs while a 3D
viewer shows you what each question is about, and leave with a validated
manifest plus the Colab cell that runs it.

A structure comes from RCSB by ID, from a file of your own, or -- with no
structure of your own -- from the **Load SyrB2 (PDB 2FCT)** button, the
worked example of the workshop: an Fe(II)/alpha-ketoglutarate halogenase with
a water on its iron. The button fetches 2FCT as a typed ID would, so the
bundle and the Colab cells treat it as one;
`tests/test_web_builder_example_button.py` drives it in a real browser.

## Why there is no JavaScript copy of the logic

The page runs the **real mdprep package** under
[Pyodide](https://pyodide.org): `plan_decisions` and `build_manifest` are the
same functions the CLI calls, so the site cannot drift from the manifest schema
and a manifest it produces always passes `mdprep config-check`.

That only works while the planning path stays importable in a browser, which
means pydantic and PyYAML and nothing heavier. `tests/test_browser_planning_path.py`
fails the build if anything else creeps in — if you add an `import numpy` to,
say, `mdprep/structure/inspect.py`, that test tells you before the site breaks.

Pyodide 0.29.5 ships prebuilt wheels for every one of those packages, so the
only thing the page has to fetch itself is the ~250 KB mdprep wheel.

## When the page refuses a structure

A ligand whose residue name an Amber force field already defines cannot be
built: `tleap` resolves residues by name, so it would use the protein, water,
ion or nucleic template instead of the ligand's parameters. mdprep refuses it
and will not rename it silently, which is what it used to do.

The page shows the refusal and offers the rename, because the structure this
matters most for is 6NIE -- fetched by PDB id, substrate deposited as
`LYS A 301`. `mdprep_rename` in `app.js` applies it through mdprep's own
reader and writer, by residue identity and never by name, so the protein's own
lysines are untouched and the text the page then plans, draws and ships is one
file rather than three descriptions of one. Accepting it also clears the
stored PDB id: the Colab cell re-downloads by id when there is one, which
would fetch the deposition that was just refused.

`tests/test_web_builder_reference_case.py` drives this in a real browser over
the untouched deposition.

## Dimers and tetramers

When all chains of an oligomer are kept and some are copies of one protein,
the page asks once whether every copy should share the first copy's
parameters, and then asks about the first copy only. The skipped questions
carry `copy_of` and a `when_any` condition; `isApplicable` in `app.js` honours
`when_any` exactly as `Decision.is_applicable` does, and `build_manifest`
fills in the copies. See [docs/oligomers.md](../docs/oligomers.md);
`tests/test_web_builder_oligomer_case.py` drives the SyrB2 dimer through the
page.

## When the 3D view misbehaves

The protein's cartoon is the one large thing the viewer draws -- 40 MB of
geometry for an octamer of 4,400 residues, against a few kilobytes for the
ligands -- so a browser short of graphics memory drops it and draws the
ligands alone. That happened on one laptop at a workshop. 3Dmol 2.5 draws
every viewer through one WebGL context it keeps for the page, and its
handler for a lost context listens on the wrong canvas, so the page looks
after its own:

- a context the browser resets (a driver hiccup, a laptop switching
  graphics cards) rebuilds the view on a new one;
- out of memory, which WebGL reports only when asked and the page asks after
  every draw, redraws the protein as lines, a fortieth of the geometry, and
  the caption says so;
- loading a structure again deletes the last view's buffers at once;
- after three rebuilds the view stops and says what to do, and **Redraw the
  3D view** rebuilds it by hand.

`tests/test_web_builder_viewer.py` forces each of these in a real browser.

## Running it locally

```bash
python web/build.py                        # stage the mdprep wheel
python -m http.server --directory web 8000
```

Then open <http://localhost:8000>. Rebuild the wheel after changing any Python.

`web/vendor/` is a build artifact and is not committed.

## Publishing it

The `web` workflow builds the site on every push and pull request, and
publishes it from `master`. Publishing needs GitHub Pages enabled on the
repository: **Settings -> Pages -> Source: GitHub Actions**. Until that is
done the build still runs and uploads the site as an artifact, and the deploy
step skips with a notice rather than failing -- a missing repository setting
is not something a commit can fix, so it should not turn the workflow red.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure |
| `app.js` | Loads Pyodide, renders the plan, drives the 3Dmol viewer |
| `zip.js` | Minimal stored-ZIP writer for the download bundle |
| `style.css` | Styling |
| `build.py` | Stages the mdprep wheel into `web/vendor/` |

## How the structure reaches Colab

Two paths, because they have different constraints:

- **Fetched from RCSB** — Colab re-downloads it by ID, so the generated cell is
  self-contained and nothing is uploaded anywhere.
- **Uploaded by you** — the structure never leaves your browser, so the site
  gives you a bundle (`structure` + `system.yaml` + a README) to upload to
  Colab. Shipping them together is deliberate: it is what stops Colab preparing
  a different file from the one this page planned.

## Deployment

`.github/workflows/pages.yml` builds the wheel and publishes `web/` to GitHub
Pages on every push to `master`. Pull requests build but do not deploy.

## The generated Colab cells

The page emits two cells, both runnable in a stock Colab notebook.

**Cell 1 builds the runtime.** `mdprep prepare` shells out to `tleap` and
`antechamber`, which pip cannot provide — but conda-forge can, so the cell
installs them rather than treating them as a prerequisite you have to arrange.
It uses micromamba in its own prefix rather than `condacolab`, because
condacolab replaces the notebook's interpreter and forces a kernel restart.

The package list is derived from your manifest by
`mdprep.workflows.runtime.conda_packages_for`, so a job is not made to wait on
a quantum chemistry package it never calls:

| Manifest | Installed |
| --- | --- |
| manual protonation, AM1-BCC ligand | `ambertools openmm parmed` |
| PropKa + xTB histidines | adds `propka xtb` |
| RESP/QMMESP charges, or a bonded MCPB site | adds `pyscf` |

It is still a large conda solve and takes several minutes, but only once per
session. The page says so.

**Cell 2 puts the inputs in place and runs**, invoking mdprep through the
micromamba environment since it is not on the notebook's own Python.

## Things worth knowing

- The generated cell installs the release `MDPREP_INSTALL` in `app.js` pins,
  from PyPI, so everyone in a workshop installs the same code. It moves only
  once a release is on PyPI (CONTRIBUTING.md, Releasing).
