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

## NVIDIA cuPotential (`engine: cupotential`)

```yaml
molecular_dynamics:
  engine: cupotential   # default: openmm
```

[cuPotential](https://github.com/NVIDIA/cuPotential) is NVIDIA's nonbonded
engine for OpenMM's CUDA platform, shipped as the `nv-openmm-plugin` wheel.
With `engine: cupotential` the dynamics -- the heating, every restrained and
unrestrained stage, step 10 and the production -- run with three of its
parts in place of OpenMM's:

- its `NonbondedForce`, copied from the one OpenMM builds from the prmtop:
  charges, Lennard-Jones, the 1-4 exceptions and the dispersion correction.
  Its Coulomb lattice sum is NVIDIA's own splitting, not OpenMM's PME;
- its `LangevinMiddleIntegrator`, which removes the centre-of-mass motion
  itself (OpenMM's `CMMotionRemover` comes out of the System, and the
  temperatures in the logs still count those three degrees of freedom out);
- its `MonteCarloBarostat`, with the pressure, frequency and seed OpenMM's
  would have had.

Bonded terms, the positional restraints and the 12-6-4 C4 term stay OpenMM's.

What changes besides the engine:

- **The minimisations stay on OpenMM.** The plugin is made for MD on a
  relaxed box, not for the atoms tleap leaves all but on top of one another,
  so steps 1, 1b, 3, 4, 5 and 5b use OpenMM's force as with `engine: openmm`,
  and the dynamics start from what they leave.
- **The Ewald error tolerance of the dynamics is 5e-4**, OpenMM's default and
  the only one the plugin accepts; mdprep's OpenMM runs use 1e-4. The report
  records both.
- **CUDA only, single or mixed precision.** Any other `platform` or
  `precision: double` is refused when the manifest is read, and a CUDA
  platform that does not start is an error rather than a fall back to OpenCL.
- **SHAKE and SETTLE only.** The plugin's integrator has no CCMA. A system
  that needs it still gets the plugin's forces, under OpenMM's integrator;
  the MD says so and `md_report.json` records why.

`md_report.json` has `engine`, the tolerances, the plugin's version and
which integrator ran. The trajectory, restarts and logs are the same files as
with OpenMM, and the analysis reads them the same way.

### Installing it

The plugin is **NVIDIA proprietary software under
[NVIDIA's licence](https://github.com/NVIDIA/cuPotential/blob/main/LICENSE-cupo.txt)**,
early access, which mdprep cannot redistribute: it is never a dependency and
is never installed unasked. Anyone who installs it accepts that licence, which
among other things restricts publishing benchmark results obtained with it.

Its wheel is for Python 3.12, OpenMM 8.4 from PyPI, CUDA 13.4 and glibc 2.38
or newer, so it gets an environment of its own:

```bash
python3.12 -m venv cupotential && . cupotential/bin/activate
pip install mdprep nv-openmm-plugin
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
mdprep run-md system.yaml
```

The `LD_LIBRARY_PATH` line puts the CUDA 13 libraries pip installed before
any system CUDA, whose NVRTC the plugin would otherwise load. NVIDIA asks for
driver R615 or newer. With an older CUDA 13 driver (CUDA 13.0-13.3, as
`nvidia-smi` prints), install that release's NVRTC in place of the plugin's,
whose kernels the driver cannot load:

```bash
pip install --no-deps "nvidia-cuda-nvrtc==13.0.*"
```

A CUDA 12 driver cannot run it.

In the official Colab notebook, choose `ENGINE = "cupotential"` in section D.
The first run builds the `mdprep-cupotential` environment
(`mdprep.workflows.runtime.cupotential_setup_commands`): it checks glibc and
the driver first and stops with the reason on a machine that cannot run the
plugin, installs the same mdprep as section A beside the plugin, and matches
NVRTC to the driver as above.

## The 12-6-4 C4 term

OpenMM builds a prmtop's C4 table into a `CustomNonbondedForce` over every
pair in the box, though the table is zero unless one atom of the pair is a
12-6-4 ion. Both engines evaluate it as an interaction group of the ions'
atoms against all atoms instead, which covers each nonzero pair and nothing
else: the forces are the same, and the energy differs only in how OpenMM
averages the force's long-range correction, by about 1e-3 kJ/mol on a box of
27,000 atoms. `md_report.json` lists the ions' atoms under
`c4_restricted_to_ion_atoms`.


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
