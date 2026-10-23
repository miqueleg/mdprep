# Roe--Brooks OpenMM MD

mdprep can continue a validated `tleap` build through the staged Roe and
Brooks minimization, heating, and density-equilibration protocol and then run a
user-sized production segment. The implementation is part of the package; the
root-level `Roe_and_Brooks.py` file is only a compatibility entry point and
does not contain a separate protocol copy.

Enable the stage in the manifest:

```yaml
molecular_dynamics:
  enabled: true
  protocol: roe_brooks_2020
  temperature_kelvin: 300.0
  pressure_atmosphere: 1.0
  platform: auto
  device_index: null
  precision: mixed
  cpu_threads: null
  random_seed: 20260817
  step08_ensemble: NPT
  density:
    increment_ns: 1.0
    report_interval_ps: 1.0
    window_ps: 300.0
    slope_threshold_g_ml_ps: 1.0e-6
    mean_difference_threshold_g_ml: 0.02
    minimum_points: 50
    maximum_duration_ns: 10.0
  production:
    steps: 50000000
    timestep_fs: 2.0
    trajectory_interval_steps: 10000
    state_interval_steps: 5000
    checkpoint_interval_steps: 50000
```

`production.steps` is mandatory when the stage is enabled. At the fixed 2 fs
time step, 50,000,000 steps correspond to 100 ns. Reporting intervals are also
given in steps and may not exceed the total production length.

Run the complete preparation and MD workflow with:

```bash
mdprep prepare system.yaml
```

To use an already generated Amber topology and coordinates without repeating
parameterization:

```bash
mdprep run-md system.yaml \
  --prmtop prepared/example/final/system.prmtop \
  --inpcrd prepared/example/final/system.inpcrd
```

`platform: auto` tries CUDA, HIP, Metal, and OpenCL before CPU and Reference.
Each backend must successfully initialize the actual system and return a finite
energy before it is selected. Failed candidates and the selected platform
properties are recorded in `md_report.json`. An explicit platform does not
fall back silently. `device_index` selects an accelerator device;
`cpu_threads` controls the OpenMM CPU platform.

### CUDA and the GPU driver

OpenMM's CUDA platform compiles its kernels when the MD starts, with the NVRTC
compiler that comes with the conda CUDA runtime, and the GPU driver then loads
them. A driver older than that runtime refuses them:
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The CUDA platform loads but cannot start,
while OpenCL, compiled by the driver itself, works. conda-forge installs the
newest runtime the driver's major version allows, which can be newer than
the driver itself: on Colab's T4 runtime, CUDA never started.

The environments mdprep sets up cap the runtime at what the driver supports:
the official notebook's section A, and the setup cell the web builder writes.
They read the driver's CUDA version from `nvidia-smi` and install with
`cuda-version<=X.Y`. To repair an environment by hand:

```bash
micromamba install -n mdprep-runtime -c conda-forge "cuda-version<=12.4"
```

using the CUDA version `nvidia-smi` prints. `mdprep doctor` starts a small
CUDA system to check, and both `doctor` and the MD say what to do when it
fails. With `platform: auto`, the MD says which platform it passed over, and why.


Step 10 stops only after its final density window passes both the configured
slope and half-window mean-difference tests. `maximum_duration_ns` prevents an
unbounded calculation: failure to converge by that limit is an explicit error,
and production is not started.

The stage writes every step PDB, density samples, step-2/step-10/final XML and
binary restarts, a periodic production checkpoint, DCD trajectory, state log,
and `md_report.json` under `md/roe_brooks_2020/`. Random seeds, platform,
thermodynamic settings, convergence statistics, energies, volumes, production
steps, and output paths are recorded.

`equilibration.csv` puts the whole protocol on one time axis: one row per
minimisation, and one every 0.5 ps of dynamics, with the stage, the simulated
time, the potential and kinetic energy, the temperature, the box volume and
the density. It is what shows the system heating in step 2, the box shrinking
from tleap's loosely packed water once the barostat starts in step 7, and the
density levelling off in step 10. `mdprep.analysis.equilibration` reads it,
and the official Colab notebook plots it; the stages are described in
`mdprep.md.roe_brooks.STAGES`.

## Restraints and the coordinate frame

Steps 1-4 restrain solute heavy atoms to the coordinates of the input
structure, and steps 6-8 restrain them to the step-5 minimum. Hydrogens, water
and free monatomic ions are solvent and are never restrained. Ion residues are
matched after stripping the charge decoration Amber writes, so `Na+`, `Cl-`,
`Mg2+` and the plain `NA`/`CL` spellings are all recognised; only single-atom
residues qualify, so a ligand whose name happens to normalise onto an element
symbol stays solute.

Coordinates passed from one stage to the next, and those written into the XML
restarts, are deliberately **not** wrapped into the periodic box. The restraint
reference coordinates live in the unwrapped frame of the input, so wrapping a
restrained particle that has crossed a periodic face would place it a full box
length from its reference. Only the per-step PDBs and the DCD trajectory are
wrapped, because those are read by humans and viewers rather than fed back into
a restrained stage.

## What the per-step PDBs are imaged into

A truncated octahedron is stored as a triclinic cell, and OpenMM's
`enforcePeriodicBox` wraps molecules into the parallelepiped spanned by that
cell's three vectors, with a corner at the origin. That is the same lattice,
but it is not the shape the manifest asked for: the solvent fills a sheared
prism, the solute sits wherever it happened to be rather than in the middle,
and water that was beside the protein is wrapped to the far corner. On a real
run, water reached 54 A from the solute in tleap's octahedron and 70 A in the
snapshot written after the MD.

So each per-step PDB is imaged the way `cpptraj`'s `autoimage` images one:
the solute is put at the origin and every other molecule is given the
periodic image nearest to it (`mdprep.md.imaging`). For a truncated
octahedron that is the octahedron back, and it is the convention tleap
already wrote `final/system.pdb` in, so the system looks the same before and
after the MD. The solute is everything that is not water or a monatomic ion
tleap added.

Each PDB also carries the box **as it was at that stage**. The cell shrinks
through the NPT steps -- 83.4 A to 79.9 A on the run above -- and a snapshot
whose `CRYST1` still claimed the starting cell made anything that rebuilt
periodic images from the header get them wrong.

`production.dcd` is left as OpenMM writes it: whole molecules wrapped into
the parallelepiped, with the correct box stored per frame. Trajectories are
conventionally imaged by whatever analyses them, and doing it at write time
would both cost wall-clock on every frame and take the choice away. With
cpptraj:

```
parm final/system.prmtop
trajin md/roe_brooks_2020/production.dcd
autoimage
trajout production.imaged.nc
```

The protocol checks numerical stability and completion, but a short smoke test
does not establish scientific convergence. Select production length and
analysis requirements appropriate for the system.
