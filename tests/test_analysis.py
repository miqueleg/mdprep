"""Reading charges, logs and trajectories the way the Colab notebook does.

A hand-written five-atom topology and a DCD written byte for byte as OpenMM
writes one, so none of this needs ParmEd, OpenMM or AmberTools.
"""

import math
import struct

import numpy as np
import pytest

from mdprep.analysis import charges, equilibration
from mdprep.analysis import trajectory as tr
from mdprep.analysis.dcd import cell_vectors, read_dcd
from mdprep.analysis.prmtop import AMBER_CHARGE_UNIT, PrmtopError, read_prmtop


# Residues: a bonded site (FE1 bound to HD1's NE2), a water, a chloride ion.
NAMES = ["NE2", "CE1", "FE", "O", "H1", "H2", "Cl-"]
TYPES = ["Y1", "CR", "M1", "OW", "HW", "HW", "Cl-"]
CHARGES = [-0.52, 0.30, 1.41, -0.834, 0.417, 0.417, -1.0]
MASSES = [14.01, 12.01, 55.85, 16.0, 1.008, 1.008, 35.45]
NUMBERS = [7, 6, 26, 8, 1, 1, 17]
RESIDUES = ["HD1", "FE1", "WAT", "Cl-"]
POINTERS_RESIDUE = [1, 3, 4, 7]
BONDS_H = [3 * 3, 3 * 4, 1, 3 * 3, 3 * 5, 1]  # O-H1, O-H2
BONDS = [0, 3 * 1, 2, 0, 3 * 2, 3]  # NE2-CE1, NE2-FE


def _section(flag, fmt, values, per_line, width, kind):
    lines = [f"%FLAG {flag}", f"%FORMAT({per_line}{kind}{width}{'.8' if kind == 'E' else ''})"]
    for start in range(0, len(values), per_line):
        chunk = values[start : start + per_line]
        if kind == "a":
            lines.append("".join(f"{value:<{width}s}" for value in chunk))
        elif kind == "E":
            lines.append("".join(f"{value:{width}.8E}" for value in chunk))
        else:
            lines.append("".join(f"{value:{width}d}" for value in chunk))
    if not values:
        lines.append("")
    return "\n".join(lines)


def write_prmtop(path, *, with_atomic_numbers=True):
    pointers = [0] * 32
    pointers[0], pointers[11], pointers[27] = len(NAMES), len(RESIDUES), 2
    parts = [
        "%VERSION  VERSION_STAMP = V0001.000",
        _section("TITLE", None, ["test"], 20, 4, "a"),
        _section("POINTERS", None, pointers, 10, 8, "I"),
        _section("ATOM_NAME", None, NAMES, 20, 4, "a"),
        _section("CHARGE", None, [q * AMBER_CHARGE_UNIT for q in CHARGES], 5, 16, "E"),
        _section("MASS", None, MASSES, 5, 16, "E"),
        _section("AMBER_ATOM_TYPE", None, TYPES, 20, 4, "a"),
        _section("RESIDUE_LABEL", None, RESIDUES, 20, 4, "a"),
        _section("RESIDUE_POINTER", None, POINTERS_RESIDUE, 10, 8, "I"),
        _section("BONDS_INC_HYDROGEN", None, BONDS_H, 10, 8, "I"),
        _section("BONDS_WITHOUT_HYDROGEN", None, BONDS, 10, 8, "I"),
    ]
    if with_atomic_numbers:
        parts.append(_section("ATOMIC_NUMBER", None, NUMBERS, 10, 8, "I"))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def write_dcd(path, frames, cells, *, dt_ps=0.002, interval=500):
    """The bytes OpenMM's DCDFile writes."""

    natoms = frames.shape[1]
    out = bytearray()
    out += struct.pack("<i4c9if", 84, b"C", b"O", b"R", b"D", len(frames), 0, interval, 0, 0, 0, 0, 0, 0,
                       dt_ps / 0.04888821)
    out += struct.pack("<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24, 84, 164, 2)
    out += struct.pack("<80s", b"Created by OpenMM") + struct.pack("<80s", b"Created now")
    out += struct.pack("<4i", 164, 4, natoms, 4)
    for frame, (a, b, c, alpha, beta, gamma) in zip(frames, cells):
        out += struct.pack(
            "<i6di", 48, a, math.cos(math.radians(gamma)), b,
            math.cos(math.radians(beta)), math.cos(math.radians(alpha)), c, 48,
        )
        for axis in range(3):
            out += struct.pack("<i", 4 * natoms)
            out += np.asarray(frame[:, axis], dtype="<f4").tobytes()
            out += struct.pack("<i", 4 * natoms)
    path.write_bytes(bytes(out))
    return path


# --------------------------------------------------------------------------
# prmtop
# --------------------------------------------------------------------------


def test_charges_residues_and_bonds_are_read(tmp_path):
    topology = read_prmtop(write_prmtop(tmp_path / "s.prmtop"))

    assert [atom.name for atom in topology.atoms] == NAMES
    assert [round(atom.charge, 4) for atom in topology.atoms] == CHARGES
    assert [residue.name for residue in topology.residues] == RESIDUES
    assert list(topology.residues[2].atom_indices) == [3, 4, 5]
    assert topology.periodic
    assert sorted(topology.bonds) == [(0, 1), (0, 2), (3, 4), (3, 5)]
    assert topology.atoms[2].element == "Fe"


def test_molecules_follow_the_bonds_not_the_residue_order(tmp_path):
    """MCPB bonds join residues that are not contiguous."""

    topology = read_prmtop(write_prmtop(tmp_path / "s.prmtop"))

    assert topology.molecules() == [[0, 1, 2], [3, 4, 5], [6]]
    assert topology.metal_bonds() == [(2, 0)]
    assert topology.label(0) == "HD1 1:NE2"


def test_an_old_topology_without_atomic_numbers_falls_back_to_masses(tmp_path):
    topology = read_prmtop(write_prmtop(tmp_path / "s.prmtop", with_atomic_numbers=False))

    assert [atom.element for atom in topology.atoms] == ["N", "C", "Fe", "O", "H", "H", "Cl"]


def test_something_else_is_refused(tmp_path):
    path = tmp_path / "not.prmtop"
    path.write_text("hello\n", encoding="utf-8")

    with pytest.raises(PrmtopError, match="not an Amber topology"):
        read_prmtop(path)


# --------------------------------------------------------------------------
# Charges
# --------------------------------------------------------------------------


def test_the_metal_site_is_shown_as_a_whole(tmp_path):
    topology = read_prmtop(write_prmtop(tmp_path / "s.prmtop"))

    kinds = {residue.name: residue.kind for residue in charges.residue_charges(topology)}
    assert kinds == {"HD1": "metal_site", "FE1": "metal_site", "WAT": "water", "Cl-": "ion"}
    site = charges.metal_site(topology)
    assert site["total"] == pytest.approx(-0.52 + 0.30 + 1.41)
    assert site["bonds"] == [
        {
            "metal": "FE1 2:FE",
            "metal_charge": 1.41,
            "donor": "HD1 1:NE2",
            "donor_type": "Y1",
            "donor_charge": -0.52,
        }
    ]
    totals = charges.summary(topology)
    assert totals["ions"] == {"Cl-": 1}
    assert totals["by_kind"]["water"] == pytest.approx(0.0)
    assert {entry["residue"] for entry in totals["fractional_residues"]} == {"HD1 1", "FE1 2"}


# --------------------------------------------------------------------------
# DCD and the trajectory
# --------------------------------------------------------------------------


OCTAHEDRON = (40.0, 40.0, 40.0, 109.4712206, 109.4712206, 109.4712206)


def _frames(count=3):
    rng = np.random.default_rng(5)
    base = rng.normal(size=(len(NAMES), 3)) * 2.0
    base[3:6] = base[3] + np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    return np.array([base + 0.05 * index for index in range(count)])


def test_the_dcd_reader_reads_what_openmm_writes(tmp_path):
    frames = _frames()
    path = write_dcd(tmp_path / "t.dcd", frames, [OCTAHEDRON] * len(frames))

    data = read_dcd(path)

    assert data.positions.shape == (3, len(NAMES), 3)
    assert np.allclose(data.positions, frames, atol=1e-4)
    assert data.timestep_ps == pytest.approx(1.0, rel=1e-5)  # 0.002 ps x 500
    assert np.allclose(data.cells[0], cell_vectors(*OCTAHEDRON), atol=1e-6)


def test_cell_vectors_reproduce_the_lengths_and_angles():
    vectors = cell_vectors(*OCTAHEDRON)

    lengths = np.linalg.norm(vectors, axis=1)
    assert np.allclose(lengths, 40.0)
    cosine = vectors[0] @ vectors[1] / (lengths[0] * lengths[1])
    assert math.degrees(math.acos(cosine)) == pytest.approx(109.4712206)


def test_a_molecule_split_across_the_box_is_put_back_beside_the_solute(tmp_path):
    frames = _frames(1)
    cell = cell_vectors(*OCTAHEDRON)
    frames[0, 6] += cell[0]  # the chloride, one whole box away
    write_prmtop(tmp_path / "s.prmtop")
    write_dcd(tmp_path / "t.dcd", frames, [OCTAHEDRON])

    imaged = tr.image(tr.load_trajectory(tmp_path / "s.prmtop", tmp_path / "t.dcd"))

    solute_centre = imaged.positions[0][[0, 1, 2]].mean(axis=0)
    assert np.allclose(solute_centre, 0.0, atol=1e-6)
    assert np.linalg.norm(imaged.positions[0, 6]) < 20.0


def test_rmsd_and_rmsf_after_superposition_ignore_rigid_motion(tmp_path):
    frames = _frames(1).repeat(4, axis=0)
    angle = math.radians(30.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
    )
    frames[2] = frames[2] @ rotation.T + 5.0
    write_prmtop(tmp_path / "s.prmtop")
    write_dcd(tmp_path / "t.dcd", frames, [OCTAHEDRON] * 4)
    trajectory = tr.load_trajectory(tmp_path / "s.prmtop", tmp_path / "t.dcd")

    atoms = [0, 1, 2, 3]
    assert np.allclose(tr.rmsd(trajectory, atoms), 0.0, atol=1e-4)
    assert np.allclose(tr.rmsf(trajectory, atoms), 0.0, atol=1e-4)
    assert tr.distances(trajectory, [(2, 0)]).shape == (4, 1)


def test_the_viewer_pdb_has_its_columns_where_readers_expect_them(tmp_path):
    from mdprep.structure.pdb import read_pdb

    write_prmtop(tmp_path / "s.prmtop")
    write_dcd(tmp_path / "t.dcd", _frames(2), [OCTAHEDRON] * 2)
    trajectory = tr.load_trajectory(tmp_path / "s.prmtop", tmp_path / "t.dcd")

    text = tr.pdb_models(trajectory, [0, 2, 3], bfactors=[-0.52, 1.41, -0.834])

    assert text.count("MODEL") == 2 and text.count("ENDMDL") == 2
    first = text.splitlines()[1]
    assert first[17:20] == "HD1" and first[21] == "A" and first[22:26] == "   1"
    # Not protein (no backbone), so drawn as a ligand.
    assert first.startswith("HETATM")
    assert float(first[60:66]) == pytest.approx(-0.52)
    (tmp_path / "frames.pdb").write_text(text, encoding="utf-8")
    parsed = read_pdb(tmp_path / "frames.pdb")
    assert [atom.name for atom in parsed.atoms] == ["NE2", "FE", "O"]


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


def test_the_equilibration_log_splits_into_its_stages(tmp_path):
    (tmp_path / "equilibration.csv").write_text(
        "stage,time_ps,stage_time_ps,potential_energy_kj_mol,kinetic_energy_kj_mol,"
        "temperature_k,box_volume_nm3,density_g_ml\n"
        "step01,0.0,0.0,-100,0,0,300,1.0\n"
        "step02,0.5,0.5,-90,50,150,300,1.0\n"
        "step02,1.0,1.0,-80,90,300,300,1.0\n"
        "step03,1.0,0.0,-95,0,0,300,1.0\n",
        encoding="utf-8",
    )

    rows = equilibration.equilibration_table(tmp_path)
    spans = equilibration.stage_spans(rows)

    assert rows[2]["temperature_k"] == 300.0
    assert [(span["stage"], span["start_ps"], span["end_ps"]) for span in spans] == [
        ("step01", 0.0, 0.0), ("step02", 0.0, 1.0), ("step03", 1.0, 1.0),
    ]
    assert spans[1]["what"].startswith("Heat")


def test_an_older_run_falls_back_to_the_step10_density(tmp_path):
    (tmp_path / "density_step10.csv").write_text(
        "time_ps,density_g_ml,box_volume_nm3\n1.0,1.02,280\n2.0,1.03,279\n", encoding="utf-8"
    )

    rows = equilibration.equilibration_table(tmp_path)

    assert [row["stage"] for row in rows] == ["step10", "step10"]
    assert rows[1]["density_g_ml"] == 1.03


@pytest.mark.parametrize("separator", ["\t", ","])
def test_the_production_log_is_read_whatever_its_separator(tmp_path, separator):
    header = separator.join(['#"Progress (%)"', '"Step"', '"Time (ps)"', '"Temperature (K)"'])
    (tmp_path / "production.log").write_text(
        header + "\n" + separator.join(["50.0%", "2500", "5.0", "300.5"]) + "\n", encoding="utf-8"
    )

    rows = equilibration.production_table(tmp_path)

    assert rows == [{"Progress (%)": "50.0%", "Step": 2500.0, "Time (ps)": 5.0, "Temperature (K)": 300.5}]


def test_tleaps_coordinates_and_box_are_read(tmp_path):
    from mdprep.analysis.prmtop import read_inpcrd

    numbers = [float(index) + 0.5 for index in range(3 * len(NAMES))]
    body = "".join(
        "".join(f"{value:12.7f}" for value in numbers[start : start + 6]) + "\n"
        for start in range(0, len(numbers), 6)
    )
    box = "".join(f"{value:12.7f}" for value in (40.0, 40.0, 40.0, 109.4712190, 109.4712190, 109.4712190))
    (tmp_path / "s.inpcrd").write_text(f"default_name\n{len(NAMES):6d}\n{body}{box}\n", encoding="utf-8")
    write_prmtop(tmp_path / "s.prmtop")

    coordinates, cell = read_inpcrd(tmp_path / "s.inpcrd")
    structure = tr.load_structure(tmp_path / "s.prmtop", tmp_path / "s.inpcrd")

    assert coordinates[1] == (3.5, 4.5, 5.5)
    assert cell == pytest.approx((40.0, 40.0, 40.0, 109.471219, 109.471219, 109.471219))
    assert structure.positions.shape == (1, len(NAMES), 3)
    assert np.allclose(np.linalg.norm(structure.cells[0], axis=1), 40.0)


def test_mcpb_force_constants_are_named_by_the_atoms_they_hold(tmp_path):
    from mdprep.analysis.metal_site import read_frcmod_terms, site_bonds

    frcmod = tmp_path / "site_mcpbpy.frcmod"
    frcmod.write_text(
        "REMARK GOES HERE, THIS FILE IS GENERATED BY MCPB.PY\n"
        "MASS\nM1 55.85                              Fe ion\n\n"
        "BOND\n"
        "Y1-M1   42.3    2.1887      Created by Seminario method using MCPB.py\n"
        "c -Y1  590.60   1.2190                2021    23398   0.0101\n\n"
        "ANGL\n"
        "CR-Y1-M1    28.28     117.29    Created by Seminario method using MCPB.py\n\n"
        "DIHE\n",
        encoding="utf-8",
    )
    topology = read_prmtop(write_prmtop(tmp_path / "s.prmtop"))

    terms = read_frcmod_terms(frcmod)
    assert [term.types for term in terms["BOND"]] == [("Y1", "M1"), ("c", "Y1")]
    assert terms["ANGL"][0].equilibrium == 117.29
    assert site_bonds(frcmod, topology) == [
        {
            "metal": "FE1 2:FE",
            "donor": "HD1 1:NE2",
            "types": "M1-Y1",
            "k_kcal_mol_A2": 42.3,
            "r0_A": 2.1887,
        }
    ]


def test_only_what_is_worth_drawing_is_drawn(tmp_path):
    """The site whole, hydrogens and waters included; elsewhere no hydrogens,
    and bulk water only when asked for."""

    write_prmtop(tmp_path / "s.prmtop")
    write_dcd(tmp_path / "t.dcd", _frames(1), [OCTAHEDRON])
    trajectory = tr.load_trajectory(tmp_path / "s.prmtop", tmp_path / "t.dcd")
    positions = trajectory.positions[0].copy()
    positions[3:6] += 50.0  # the water, far from the iron
    positions[6] += 50.0    # the chloride too

    near = tr.display_atoms(trajectory.topology, positions, near_metal=8.0)
    with_bulk = tr.display_atoms(trajectory.topology, positions, near_metal=8.0, bulk_water=True)

    assert near == [0, 1, 2, 6]  # site whole; the far chloride is a heavy atom
    assert with_bulk == [0, 1, 2, 3, 6]  # plus the far water's oxygen only
    with pytest.raises(ValueError):
        tr.display_atoms(trajectory.topology, positions, protein="all")
