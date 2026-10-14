# Installation

One command, then check it:

```bash
conda env create -f environment.yml   # or: mamba env create -f environment.yml
conda activate mdprep
mdprep doctor
```

`environment.yml` installs everything the tutorials need, mdprep included.
`mdprep doctor` then reports what is present and what is not, and prints the
command to install anything missing. Given a manifest it reports only what
that manifest will actually use, so nobody installs a quantum chemistry
package for a job that never calls it:

```bash
mdprep doctor examples/16_besd_halogenase_gfn2_mcpb.yaml
```

To write and validate manifests without preparing anything, the much smaller
`environment-minimal.yml` is enough.

## What gets installed, and why

| | Needed for |
|---|---|
| AmberTools | `tleap`, `antechamber`, `parmchk2`, `MCPB.py` |
| PropKa | `protonation.method: propka` and `propka_xtb_his` |
| xTB | histidine tautomers, GFN2 charges, GFN2 Hessians and ESP |
| OpenMM | force-field refinement, the MM point charges behind the embedded histidine comparison, validation |
| PDBFixer | `structure.repair`, temporary hydrogenation |
| RDKit | placing ligand hydrogens from `ligands[].smiles` |
| ParmEd | topology and charge validation |
| PySCF | QMMESP and DFT MCPB RESP fitting |

## ASH

`refinement.backend: ash` is the one thing `environment.yml` cannot provide.
ASH is a manual checkout from
[RagnarB83/ash](https://github.com/RagnarB83/ash), put on `PYTHONPATH`; it
also wants `geometric`, which `environment.yml` does install.

**Do not `pip install ash`.** The `ash` distribution on PyPI is unrelated
software, and installing it produces an environment that looks correct and is
not. The default `refinement.backend: openmm` needs none of this, holds every
heavy atom fixed, and relaxes the hydrogens in seconds.

All distributed manifests use relative file paths or executable names.
Activate the environment containing the requested backend (or put its
executable on `PATH`) before running mdprep. Runtime reports record resolved
executables, but no developer-machine path is embedded in the package. Bonded MCPB completion additionally requires user-run Gaussian or
GAMESS calculations matching the selected MCPB input format when using
`workflow: complete`. `workflow: pyscf` instead uses either a user-supplied,
already optimized MCPB small-model PDB or the geometry inherited from a
completed ASH QM/MM refinement. It then performs fixed-geometry Hessian and
embedded ESP calculations.

The MCPB `hessian_backend: xtb` supports `xtb.model: gfn2` with a standard xTB
binary and `xtb.model: gxtb` with a g-xTB-enabled binary. The legacy
`hessian_backend: gxtb`/`gxtb:` spelling remains supported. For g-xTB, download
a reviewed binary from the
[official g-xTB releases](https://github.com/grimme-lab/g-xtb/releases).
Point the applicable `executable` setting to the binary and optionally pin its
SHA-256 in the manifest. mdprep does not download or replace scientific
executables during a preparation run.

MACE-POLAR-1 is best installed in a dedicated Python 3.11 environment because
its PyTorch stack need not be shared with mdprep. One reproducible CPU setup
used by the external test is:

```bash
python3.11 -m venv mace-polar1-env
mace-polar1-env/bin/python -m pip install "mace-torch==0.3.16"
mace-polar1-env/bin/python -m pip install \
  "graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git@66ef8753f1675ed77af10d9a03401d827cd3d188"
```

Set `mcpb.pyscf.mace_polar1.python_executable` to that environment's Python.
The MACE factory downloads the selected official checkpoint into its cache on
first use; mdprep records and can enforce its SHA-256. Review the
[MACE-POLAR-1 documentation](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)
and Academic Software License before setting `accept_model_license: true`.
MACE 0.3.16 or newer is required because that release added PolarMACE Hessian
support. Unit tests use fakes; set `MDPREP_MACE_POLAR1_PYTHON` to run the real
external test.

Active-site QM/MM refinement additionally requires an ASH Python environment
with OpenMM and geomeTRIC. ASH can remain isolated from the main `mdprep`
environment: set `refinement.ash.python_executable` to that environment's
Python. GFN2-xTB and g-xTB refinement also use
`refinement.ash.xtb_executable`. MACE-POLAR-1 refinement requires MACE and
PolarMACE's dependencies in the ASH environment, or explicitly reviewed
directories under `refinement.mace_polar1.python_search_paths`. See
[Active-site QM/MM refinement](qmmm_refinement.md).

For development and release checks:

```bash
mamba env create -f environment-dev.yml
conda activate mdprep-dev
```

Pip-only installs can run the pure Python tests, but real preparation workflows
that call external chemistry tools require those executables or libraries.

## Quick Verification

```bash
mdprep --version
mdprep config-check examples/*.yaml
mdprep selftest --quick
pytest -q
```
