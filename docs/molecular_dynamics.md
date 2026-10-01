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

Step 10 stops only after its final density window passes both the configured
slope and half-window mean-difference tests. `maximum_duration_ns` prevents an
unbounded calculation: failure to converge by that limit is an explicit error,
and production is not started.

The stage writes every step PDB, density samples, step-2/step-10/final XML and
binary restarts, a periodic production checkpoint, DCD trajectory, state log,
and `md_report.json` under `md/roe_brooks_2020/`. Random seeds, platform,
thermodynamic settings, convergence statistics, energies, volumes, production
steps, and output paths are recorded.

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

The protocol checks numerical stability and completion, but a short smoke test
does not establish scientific convergence. Select production length and
analysis requirements appropriate for the system.
