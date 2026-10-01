# mdprep input builder (static site)

A browser front end for `mdprep plan` and `mdprep init --interactive`. Load a
structure, answer only the decisions that structure actually needs while a 3D
viewer shows you what each question is about, and leave with a validated
manifest plus the Colab cell that runs it.

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

## Running it locally

```bash
python web/build.py                        # stage the mdprep wheel
python -m http.server --directory web 8000
```

Then open <http://localhost:8000>. Rebuild the wheel after changing any Python.

`web/vendor/` is a build artifact and is not committed.

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

## Things worth knowing

- `mdprep prepare` needs AmberTools (`tleap`, `antechamber`), which pip cannot
  install, so the generated cell is meant for a Colab runtime that already has
  them. The page says so rather than handing you a command that fails.
- The generated cell installs mdprep **from the repository**, not PyPI: the
  published 0.2.0 wheel predates the Roe--Brooks NaN and large-PDB fixes.
  Change `MDPREP_INSTALL` in `app.js` once a release carrying them is on PyPI.
