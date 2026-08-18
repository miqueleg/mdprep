#!/usr/bin/env python3
"""
Roe & Brooks (2020) 10-step equilibration protocol in OpenMM (hardened + window-based Step10 stop)

What’s included / fixed:
- Step counts: int(round(time/dt)) using OpenMM Quantities
- Restraints do NOT accumulate across steps (fresh System each step)
- Restraints are added before Context creation and reference coordinates are
  converted explicitly to OpenMM's nanometer unit
- "Heavy" restraints exclude water/ions (prevents huge solvent restraint energies / NaNs)
- Step 5b constrained minimization (HBonds + rigid water) to make SHAKE stable
- Step 1b constraint-compatible minimization before the first heating stage
- Step 5c short, very gentle NVT settle before entering Step 6
- Step 6 runs as NVT (no barostat) to avoid early NPT instability; Step 7+ are NPT
- Updated periodic box vectors propagate between NPT stages
- Production progress includes the already completed equilibration steps
- Density conversion fixed: dalton/nm^3 -> g/cm^3 (uses exact Avogadro constant)
- Sanity checks:
    * prmtop atom count == inpcrd positions count
    * periodic box vectors exist and volume is finite/positive
- Step10 convergence STOP is WINDOW-based (point-based, robust):
    * last window_ps (default 300 ps) must satisfy:
        - |slope| < slope_thresh (default 1e-6 g/cc/ps)
        - |mean(second half) - mean(first half)| < mean_diff_thresh (default 0.02 g/cc)
- Writes PDB at the end of EVERY step (step01.pdb ... step10.pdb) + production_last.pdb
- Writes restart files (.chk + .xml) after Step 2, after Step 10, and after production
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import openmm as mm
from openmm import unit, Platform
from openmm.app import DCDReporter, StateDataReporter
from openmm.app import (
    AmberPrmtopFile,
    AmberInpcrdFile,
    Simulation,
    PDBFile,
    PME,
    HBonds,
)

# Backbone-ish names (protein + nucleic acids)
BACKBONE_NAMES = {
    "N", "CA", "C", "O", "OXT",
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "O2'",
}

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TP3", "SPC", "SPCE"}
ION_RESNAMES = {"NA", "CL", "K", "RB", "CS", "LI", "MG", "CA", "ZN", "MN", "FE", "CU", "CO", "NI"}


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def steps_from_time(time_q, dt_q) -> int:
    """Compute integer number of steps from two OpenMM Quantities."""
    return int(round(time_q / dt_q))


def write_pdb(sim: Simulation, topology, out_pdb: Path):
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(out_pdb, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f, keepIds=True)


def save_restart(sim: Simulation, prefix: Path):
    sim.saveCheckpoint(str(prefix.with_suffix(".chk")))
    st = sim.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True,
        enforcePeriodicBox=True
    )
    with open(prefix.with_suffix(".xml"), "w") as f:
        f.write(mm.XmlSerializer.serialize(st))


def add_positional_restraints(system: mm.System, topology, ref_positions,
                             k_kcal_per_A2: float, selection: str):
    """
    selection:
      - "heavy": restrain solute heavy atoms (non-H), excluding water/ions
      - "backbone": restrain solute backbone-ish heavy atoms (non-H), excluding water/ions
    """
    selection = selection.strip().lower()

    k = (k_kcal_per_A2 * unit.kilocalories_per_mole / (unit.angstrom**2)).in_units_of(
        unit.kilojoules_per_mole / (unit.nanometer**2)
    )
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", k)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    # CustomExternalForce per-particle coordinates are unitless and interpreted
    # in nm. AmberInpcrdFile positions can retain an angstrom display unit.
    ref_positions_nm = ref_positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        name = atom.name.strip()
        is_h = (atom.element is not None and atom.element.symbol == "H") or name.startswith("H")
        if is_h:
            continue

        resname = atom.residue.name.strip().upper()
        # IMPORTANT: exclude water/ions from restraints to avoid solvent pinning instabilities
        if resname in WATER_RESNAMES or resname in ION_RESNAMES:
            continue

        if selection == "heavy":
            p = ref_positions_nm[atom.index]
            force.addParticle(atom.index, [p.x, p.y, p.z])

        elif selection == "backbone":
            if name in BACKBONE_NAMES:
                p = ref_positions_nm[atom.index]
                force.addParticle(atom.index, [p.x, p.y, p.z])

        else:
            raise ValueError(f"Unknown restraint selection: {selection!r}")

    system.addForce(force)
    return force


def build_system(prmtop: AmberPrmtopFile, constraints_setting, rigid_water: bool,
                 cutoff_nm=0.8, ewald_tol=1e-4, use_barostat=False, P=None, T=None):
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=cutoff_nm * unit.nanometer,
        constraints=constraints_setting,
        rigidWater=rigid_water,
        ewaldErrorTolerance=ewald_tol,
    )

    # Enable dispersion correction if NonbondedForce present
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(True)

    if use_barostat:
        if P is None or T is None:
            raise ValueError("P and T required for barostat")
        system.addForce(mm.MonteCarloBarostat(P, T, 100))  # attempt every 100 steps

    return system


def make_sim(topology, system, integrator, platform, properties,
             positions, box_vectors=None, velocities=None, set_temp=None):
    sim = Simulation(topology, system, integrator, platform, properties)
    sim.context.setPositions(positions)
    if box_vectors is not None:
        sim.context.setPeriodicBoxVectors(*box_vectors)
    if velocities is not None:
        sim.context.setVelocities(velocities)
    if set_temp is not None:
        sim.context.setVelocitiesToTemperature(set_temp)
    return sim


def print_pre_step_diagnostics(sim: Simulation, label: str):
    st = sim.context.getState(getEnergy=True, enforcePeriodicBox=True)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    ke = st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    V = st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    print(f"{label} pre-step: PE={pe:.3e} kJ/mol  KE={ke:.3e} kJ/mol  V={V:.6f} nm^3")


def compute_density_g_per_cm3(state: mm.State, system: mm.System) -> float:
    """
    Convert density from (dalton / nm^3) to (g / cm^3) explicitly.
    1 dalton = 1 g/mol; grams = (Da) / N_A; cm^3 = nm^3 * 1e-21
    """
    mass_da = 0.0
    for i in range(system.getNumParticles()):
        mass_da += system.getParticleMass(i).value_in_unit(unit.dalton)

    vol_nm3 = state.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)

    NA = 6.02214076e23  # 1/mol (exact)
    mass_g = mass_da / NA
    vol_cm3 = vol_nm3 * 1e-21
    return mass_g / vol_cm3


def density_plateau_check_window_points(times_ps, dens_g_cm3,
                                        window_ps=300.0,
                                        report_ps=1.0,
                                        slope_thresh=1e-6,
                                        mean_diff_thresh=0.02,
                                        min_points=50):
    """
    Robust window plateau test using the last N points, where
    N ~= window_ps / report_ps.

    Criteria on the last window:
      - |linear slope| < slope_thresh   (g/cm^3/ps)
      - |mean(second half) - mean(first half)| < mean_diff_thresh (g/cm^3)
    """
    t = np.asarray(times_ps, dtype=float)
    D = np.asarray(dens_g_cm3, dtype=float)

    n = len(D)
    if n < min_points:
        return False, {"reason": f"Need >= {min_points} total points (have {n})"}

    # number of samples corresponding to window_ps
    nwin = int(round(float(window_ps) / float(report_ps))) + 1
    nwin = max(min_points, nwin)

    if n < nwin:
        return False, {
            "reason": f"Need >= {nwin} points for window {window_ps} ps at {report_ps} ps/report (have {n})"
        }

    tw = t[-nwin:]
    Dw = D[-nwin:]

    # Linear fit
    m, b = np.polyfit(tw, Dw, 1)

    mid = len(Dw) // 2
    mean_first = float(np.mean(Dw[:mid]))
    mean_second = float(np.mean(Dw[mid:]))
    dmean = abs(mean_second - mean_first)
    stdw = float(np.std(Dw))

    ok = (abs(m) < slope_thresh) and (dmean < mean_diff_thresh)
    info = {
        "npoints_window": int(nwin),
        "window_ps": float(tw[-1] - tw[0]),
        "t_start_ps": float(tw[0]),
        "t_end_ps": float(tw[-1]),
        "slope": float(m),
        "slope_thresh": float(slope_thresh),
        "dmean": float(dmean),
        "mean_diff_thresh": float(mean_diff_thresh),
        "std_window": float(stdw),
        "mean_first": mean_first,
        "mean_second": mean_second,
    }
    return ok, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prmtop", required=True)
    ap.add_argument("--inpcrd", required=True)
    ap.add_argument("--outprefix", default="run")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--pressure", type=float, default=1.0)
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--device", default=None)
    ap.add_argument("--production_ns", type=float, required=True)
    ap.add_argument("--prod_dcd_ps", type=float, default=20.0,help="Write production DCD every N ps")

    # Step 10 density reporting / chunking
    ap.add_argument("--step10_increment_ns", type=float, default=1.0,
                    help="Step10 runs in chunks of this size (ns); convergence is checked after each density sample.")
    ap.add_argument("--density_report_ps", type=float, default=1.0,
                    help="Write a density sample every N ps during Step10.")

    # Window-based convergence stop (Step10)
    ap.add_argument("--density_window_ps", type=float, default=300.0,
                    help="Window length (ps) for Step10 convergence check.")
    ap.add_argument("--density_window_slope_thresh", type=float, default=1e-6,
                    help="Slope threshold (g/cm^3/ps) for Step10 window convergence.")
    ap.add_argument("--density_window_mean_diff_thresh", type=float, default=0.02,
                    help="Mean-diff threshold (g/cm^3) for Step10 window convergence.")
    ap.add_argument("--density_window_min_points", type=int, default=50,
                    help="Minimum points required in the window test (and total).")

    # Debug option
    ap.add_argument("--write_pre_pdb", action="store_true",
                    help="Write *_pre.pdb before each MD stage step() call (useful for debugging).")
    ap.add_argument("--step08_nvt", action="store_true",
                    help="Run step08 without barostat (NVT) for stability.")
    args = ap.parse_args()

    outdir = Path(args.outprefix)
    ensure_dir(outdir)

    T = args.temperature * unit.kelvin
    P = args.pressure * unit.atmosphere

    # Load inputs
    prmtop = AmberPrmtopFile(args.prmtop)
    inpcrd = AmberInpcrdFile(args.inpcrd)
    topology = prmtop.topology
    positions0 = inpcrd.positions
    box_vectors = inpcrd.boxVectors

    # Platform/properties
    platform = Platform.getPlatformByName(args.platform)
    properties = {}
    if args.device is not None and args.platform in ("CUDA", "OpenCL"):
        properties["DeviceIndex"] = str(args.device)
        if args.platform == "CUDA":
            properties["Precision"] = "mixed"

    # ---------------- Sanity checks
    n_top = topology.getNumAtoms()
    n_pos = len(positions0)
    print("Topology atoms:", n_top, "Coordinate atoms:", n_pos)
    if n_top != n_pos:
        raise RuntimeError("prmtop and inpcrd atom counts differ -> mismatch (will crash).")

    if box_vectors is None:
        raise RuntimeError("inpcrd/rst7 has no box vectors. PME/NPT requires periodic box vectors.")

    tmp_sys = build_system(prmtop, constraints_setting=HBonds, rigid_water=True, use_barostat=False)
    tmp_sim = make_sim(topology, tmp_sys, mm.VerletIntegrator(1.0 * unit.femtoseconds),
                       platform, properties, positions0, box_vectors=box_vectors)
    st_tmp = tmp_sim.context.getState(enforcePeriodicBox=True)
    V0 = st_tmp.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)
    print(f"Initial box volume: {V0:.6f} nm^3")
    if (not np.isfinite(V0)) or V0 < 1e-6:
        raise RuntimeError(f"Bad box volume ({V0} nm^3). Check inpcrd box vectors.")

    # Time steps
    dt_dummy = 1.0 * unit.femtoseconds  # for Verlet holder/minimization
    dt1 = 1.0 * unit.femtoseconds
    dt2 = 2.0 * unit.femtoseconds

    # ---------------- Step 1: min 1000 iters, heavy 5.0, NO constraints
    system1 = build_system(prmtop, constraints_setting=None, rigid_water=False, use_barostat=False)
    ref0 = positions0
    add_positional_restraints(system1, topology, ref0, 5.0, "heavy")
    sim1 = make_sim(topology, system1, mm.VerletIntegrator(dt_dummy), platform, properties,
                    positions0, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim1.context, maxIterations=1000)
    write_pdb(sim1, topology, outdir / "step01.pdb")
    pos1 = sim1.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    # Constraint-compatible handoff before SHAKE/rigid-water heating.
    system1b = build_system(prmtop, constraints_setting=HBonds, rigid_water=True, use_barostat=False)
    add_positional_restraints(system1b, topology, ref0, 5.0, "heavy")
    sim1b = make_sim(topology, system1b, mm.VerletIntegrator(dt_dummy), platform, properties,
                     pos1, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim1b.context, maxIterations=500)
    write_pdb(sim1b, topology, outdir / "step01b.pdb")
    pos1 = sim1b.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    # ---------------- Step 2: 15 ps NVT, dt=1 fs, heavy 5.0, constraints ON, velocities at T
    system2 = build_system(prmtop, constraints_setting=HBonds, rigid_water=True, use_barostat=False)
    add_positional_restraints(system2, topology, ref0, 5.0, "heavy")  # reference = initial coords per paper
    integrator2 = mm.LangevinMiddleIntegrator(T, 5.0 / unit.picosecond, dt1)
    integrator2.setConstraintTolerance(1e-6)
    sim2 = make_sim(topology, system2, integrator2, platform, properties,
                    pos1, box_vectors=box_vectors, set_temp=T)
    if args.write_pre_pdb:
        write_pdb(sim2, topology, outdir / "step02_pre.pdb")
    print_pre_step_diagnostics(sim2, "step02")
    sim2.step(steps_from_time(15.0 * unit.picoseconds, dt1))
    write_pdb(sim2, topology, outdir / "step02.pdb")
    save_restart(sim2, outdir / "after_step02")
    st2 = sim2.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos2, vel2 = st2.getPositions(), st2.getVelocities()

    # ---------------- Step 3: min 1000 iters, heavy 2.0, NO constraints
    system3 = build_system(prmtop, constraints_setting=None, rigid_water=False, use_barostat=False)
    add_positional_restraints(system3, topology, ref0, 2.0, "heavy")
    sim3 = make_sim(topology, system3, mm.VerletIntegrator(dt_dummy), platform, properties,
                    pos2, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim3.context, maxIterations=1000)
    write_pdb(sim3, topology, outdir / "step03.pdb")
    pos3 = sim3.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    # ---------------- Step 4: min 1000 iters, heavy 0.1, NO constraints
    system4 = build_system(prmtop, constraints_setting=None, rigid_water=False, use_barostat=False)
    add_positional_restraints(system4, topology, ref0, 0.1, "heavy")
    sim4 = make_sim(topology, system4, mm.VerletIntegrator(dt_dummy), platform, properties,
                    pos3, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim4.context, maxIterations=1000)
    write_pdb(sim4, topology, outdir / "step04.pdb")
    pos4 = sim4.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    # ---------------- Step 5: min 1000 iters, no restraints, NO constraints
    system5 = build_system(prmtop, constraints_setting=None, rigid_water=False, use_barostat=False)
    sim5 = make_sim(topology, system5, mm.VerletIntegrator(dt_dummy), platform, properties,
                    pos4, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim5.context, maxIterations=1000)
    write_pdb(sim5, topology, outdir / "step05.pdb")
    pos5 = sim5.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()

    # ---------------- Step 5b: constrained minimization (HBonds + rigid water) to repair SHAKE geometry
    system5b = build_system(prmtop, constraints_setting=HBonds, rigid_water=True, use_barostat=False)
    sim5b = make_sim(topology, system5b, mm.VerletIntegrator(dt_dummy), platform, properties,
                     pos5, box_vectors=box_vectors)
    mm.LocalEnergyMinimizer.minimize(sim5b.context, maxIterations=500)
    pos5 = sim5b.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions()
    write_pdb(sim5b, topology, outdir / "step05b.pdb")

    # Reference for steps 6-8 per paper (now constraint-consistent)
    ref5 = pos5

    # ---------------- Step 5c: short restrained NVT settle (very stable) before Step 6
    system5c = build_system(prmtop, constraints_setting=HBonds, rigid_water=True, use_barostat=False)
    add_positional_restraints(system5c, topology, ref5, 1.0, "heavy")
    dt_settle = 0.25 * unit.femtoseconds
    integ5c = mm.LangevinMiddleIntegrator(T, 20.0 / unit.picosecond, dt_settle)
    integ5c.setConstraintTolerance(1e-6)
    sim5c = make_sim(topology, system5c, integ5c, platform, properties,
                     pos5, box_vectors=box_vectors, set_temp=T)
    if args.write_pre_pdb:
        write_pdb(sim5c, topology, outdir / "step05c_pre.pdb")
    print_pre_step_diagnostics(sim5c, "step05c")
    sim5c.step(steps_from_time(0.2 * unit.picoseconds, dt_settle))
    st5c = sim5c.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    pos5 = st5c.getPositions()
    vel5 = st5c.getVelocities()
    write_pdb(sim5c, topology, outdir / "step05c.pdb")

    # -------- Steps 6-10 helper
    def run_md(step_label, start_pos, start_vel, start_box, dt, duration_ps,
               restraint_k=None, restraint_sel=None,
               assign_velocities=False, use_barostat=True):
        system = build_system(prmtop, constraints_setting=HBonds, rigid_water=True,
                              use_barostat=use_barostat, P=P, T=T)
        if restraint_k is not None and restraint_sel is not None:
            add_positional_restraints(system, topology, ref5, restraint_k, restraint_sel)

        integrator = mm.LangevinMiddleIntegrator(T, 5.0 / unit.picosecond, dt)
        integrator.setConstraintTolerance(1e-6)

        sim = make_sim(topology, system, integrator, platform, properties,
                       start_pos, box_vectors=start_box,
                       velocities=None if assign_velocities else start_vel,
                       set_temp=T if assign_velocities else None)

        if args.write_pre_pdb:
            write_pdb(sim, topology, outdir / f"{step_label}_pre.pdb")
        print_pre_step_diagnostics(sim, step_label)

        sim.step(steps_from_time(duration_ps * unit.picoseconds, dt))
        write_pdb(sim, topology, outdir / f"{step_label}.pdb")
        st = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
        return st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors()

    # ---------------- Step 6: 5 ps NVT (NO barostat), 1 fs, heavy 1.0
    pos6, vel6, box6 = run_md("step06", pos5, vel5, box_vectors, dt1, 5.0,
                        restraint_k=1.0, restraint_sel="heavy",
                        assign_velocities=False, use_barostat=False)

    # ---------------- Step 7: 5 ps NPT, 1 fs, heavy 0.5
    pos7, vel7, box7 = run_md("step07", pos6, vel6, box6, dt1, 5.0,
                        restraint_k=0.5, restraint_sel="heavy",
                        assign_velocities=False, use_barostat=True)

    # ---------------- Step 8: 10 ps, 1 fs, backbone 0.5 (optional NVT)
    pos8, vel8, box8 = run_md("step08", pos7, vel7, box7, dt1, 10.0,
                        restraint_k=0.5, restraint_sel="backbone",
                        assign_velocities=False, use_barostat=(not args.step08_nvt))

    # ---------------- Step 9: 10 ps NPT, 2 fs, unrestrained
    pos9, vel9, box9 = run_md("step09", pos8, vel8, box8, dt2, 10.0,
                        restraint_k=None, restraint_sel=None,
                        assign_velocities=False, use_barostat=True)

    # ---------------- Step 10: NPT, dt=2 fs, run in increments until WINDOW plateau
    system10 = build_system(prmtop, constraints_setting=HBonds, rigid_water=True,
                            use_barostat=True, P=P, T=T)
    integrator10 = mm.LangevinMiddleIntegrator(T, 5.0 / unit.picosecond, dt2)
    integrator10.setConstraintTolerance(1e-6)
    sim10 = make_sim(topology, system10, integrator10, platform, properties,
                     pos9, box_vectors=box9, velocities=vel9)

    density_csv = outdir / "density_step10.csv"
    with open(density_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["time_ps", "density_g_cm3", "box_volume_nm3"])

    times_ps, dens = [], []
    elapsed_ps = 0.0

    # how often we record density (in steps)
    report_every = steps_from_time(args.density_report_ps * unit.picoseconds, dt2)
    report_every = max(1, report_every)

    # chunk size (in steps)
    chunk_steps = steps_from_time(args.step10_increment_ns * unit.nanoseconds, dt2)

    plateaued = False
    info = {}
    if args.write_pre_pdb:
        write_pdb(sim10, topology, outdir / "step10_pre.pdb")
    print_pre_step_diagnostics(sim10, "step10")

    # We keep the chunk loop for efficiency, but we CHECK convergence after EVERY density sample.
    while not plateaued:
        steps_left = chunk_steps
        while steps_left > 0:
            now = min(report_every, steps_left)
            sim10.step(now)
            steps_left -= now

            st = sim10.context.getState(enforcePeriodicBox=True)
            d = compute_density_g_per_cm3(st, system10)
            V = st.getPeriodicBoxVolume().value_in_unit(unit.nanometer**3)

            elapsed_ps += (now * dt2).value_in_unit(unit.picoseconds)

            times_ps.append(elapsed_ps)
            dens.append(d)

            with open(density_csv, "a", newline="") as fcsv:
                csv.writer(fcsv).writerow([elapsed_ps, d, V])

            # ---- WINDOW convergence check (point-based, robust)
            plateaued, info = density_plateau_check_window_points(
                times_ps, dens,
                window_ps=args.density_window_ps,
                report_ps=args.density_report_ps,
                slope_thresh=args.density_window_slope_thresh,
                mean_diff_thresh=args.density_window_mean_diff_thresh,
                min_points=args.density_window_min_points,
            )

            # Debug print occasionally so you can see it approaching convergence
            if len(dens) % 50 == 0:
                if "slope" in info:
                    print(f"[Step10] t={elapsed_ps:.1f} ps | "
                          f"slope={info['slope']:.3e} (thr {info['slope_thresh']:.1e}) | "
                          f"dmean={info['dmean']:.6f} (thr {info['mean_diff_thresh']:.2f}) | "
                          f"std={info['std_window']:.6f} | plateaued={plateaued}")
                else:
                    print(f"[Step10] t={elapsed_ps:.1f} ps | {info.get('reason', 'no info')}")

            if plateaued:
                print(f"[Step10] WINDOW converged at t={elapsed_ps:.1f} ps | "
                      f"slope={info.get('slope', float('nan')):.3e} | "
                      f"dmean={info.get('dmean', float('nan')):.6f} | "
                      f"std={info.get('std_window', float('nan')):.6f}")
                break

        if plateaued:
            break

    write_pdb(sim10, topology, outdir / "step10.pdb")
    save_restart(sim10, outdir / "after_step10")

    # ---------------- Production for user-provided X ns (continue from Step 10 end)

    # DCD reporter
    prod_dcd_steps = steps_from_time(args.prod_dcd_ps * unit.picoseconds, dt2)
    sim10.reporters.append(
        DCDReporter(str(outdir / "production.dcd"), prod_dcd_steps)
    )

    # Optional: text log (energies, temp, density)
    sim10.reporters.append(
        StateDataReporter(
            str(outdir / "production.log"),
            prod_dcd_steps,
            step=True, time=True,
            potentialEnergy=True, kineticEnergy=True,
            temperature=True, density=True,
            progress=True, remainingTime=True,
            speed=True,
            totalSteps=(
                sim10.currentStep
                + steps_from_time(args.production_ns * unit.nanoseconds, dt2)
            ),
            separator="\t"
        )
    )

    prod_steps = steps_from_time(args.production_ns * unit.nanoseconds, dt2)
    sim10.step(prod_steps)

    write_pdb(sim10, topology, outdir / "production_last.pdb")
    save_restart(sim10, outdir / "after_production")

    print("DONE")
    print(f"Step10 WINDOW plateau satisfied at ~{elapsed_ps:.1f} ps with info: {info}")
    print(f"Outputs in: {outdir}")


if __name__ == "__main__":
    main()
