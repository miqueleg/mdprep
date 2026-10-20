"""A production trajectory, imaged and superposed, ready to plot and to watch.

The frames are imaged exactly as mdprep images its snapshots
(``mdprep.md.imaging``): solute centred, every molecule at its nearest image,
so the protein does not wander out of the box in the viewer and the truncated
octahedron looks like one. They are then superposed on the protein's C-alpha
atoms, which is what RMSD and RMSF are measured after.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from mdprep.analysis.dcd import read_dcd
from mdprep.analysis.prmtop import AmberTopology, read_prmtop
from mdprep.md.imaging import image_molecules_about_solute, solvent_residue_names
from mdprep.structure.classify import STANDARD_PROTEIN_RESIDUES, WATER_RESIDUES


@dataclass
class Trajectory:
    topology: AmberTopology
    positions: np.ndarray  # (frames, atoms, 3), angstrom
    cells: np.ndarray | None  # (frames, 3, 3), angstrom
    times_ps: np.ndarray  # (frames,)

    @property
    def frames(self) -> int:
        return len(self.positions)


def load_trajectory(prmtop: str | Path, dcd: str | Path, *, first_time_ps: float = 0.0) -> Trajectory:
    topology = read_prmtop(prmtop)
    data = read_dcd(dcd)
    if data.positions.shape[1] != len(topology.atoms):
        raise ValueError(
            f"{Path(dcd).name} has {data.positions.shape[1]} atoms and "
            f"{Path(prmtop).name} has {len(topology.atoms)}: not the same system"
        )
    # The step is stored as a float32 in AKMA units, so round off the noise.
    step = round(data.timestep_ps, 4) if data.timestep_ps else 1.0
    times = first_time_ps + step * np.arange(1, len(data.positions) + 1)
    return Trajectory(topology=topology, positions=data.positions, cells=data.cells, times_ps=times)


def load_structure(prmtop: str | Path, inpcrd: str | Path) -> Trajectory:
    """The built system as a one-frame trajectory, from tleap's coordinates."""

    from mdprep.analysis.dcd import cell_vectors
    from mdprep.analysis.prmtop import read_inpcrd

    topology = read_prmtop(prmtop)
    coordinates, box = read_inpcrd(inpcrd)
    if len(coordinates) != len(topology.atoms):
        raise ValueError(f"{Path(inpcrd).name} and {Path(prmtop).name} differ in atom count")
    cells = None if box is None else np.array([cell_vectors(*box)])
    return Trajectory(
        topology=topology,
        positions=np.array([coordinates], dtype=float),
        cells=cells,
        times_ps=np.zeros(1),
    )


# --------------------------------------------------------------------------
# Selections
# --------------------------------------------------------------------------


def protein_atoms(topology: AmberTopology, names: Iterable[str] | None = None) -> list[int]:
    """Protein atoms, by residue name; only ``names`` (e.g. {"CA"}) if given.

    tleap writes termini as NALA/CALA, and MCPB.py renames a metal-bound
    residue (HIS becomes HD1, HE1...), which is still protein: it has a CA.
    """

    wanted = set(names) if names is not None else None
    protein_like = set(STANDARD_PROTEIN_RESIDUES) | {
        prefix + name for prefix in ("N", "C") for name in STANDARD_PROTEIN_RESIDUES
    } | {"NME", "ACE"}
    selected = []
    for residue in topology.residues:
        atom_names = {topology.atoms[i].name for i in residue.atom_indices}
        if residue.name not in protein_like and not {"N", "CA", "C"} <= atom_names:
            continue
        for index in residue.atom_indices:
            if wanted is None or topology.atoms[index].name in wanted:
                selected.append(index)
    return selected


def solute_atoms(topology: AmberTopology) -> list[int]:
    solvent = solvent_residue_names()
    selected = [
        atom.index
        for atom in topology.atoms
        if topology.residues[atom.residue_index].name.upper() not in solvent
    ]
    return selected or [atom.index for atom in topology.atoms]


def water_oxygens(topology: AmberTopology) -> list[int]:
    return [
        atom.index
        for atom in topology.atoms
        if topology.residues[atom.residue_index].name in WATER_RESIDUES and atom.atomic_number == 8
    ]


def residues_near(
    topology: AmberTopology, positions: np.ndarray, centre_atoms: Sequence[int], cutoff: float
) -> list[int]:
    """Residues with any atom within ``cutoff`` of ``centre_atoms`` in one frame."""

    centre = positions[list(centre_atoms)]
    near: set[int] = set()
    for residue in topology.residues:
        index = list(residue.atom_indices)
        distances = np.linalg.norm(positions[index][:, None, :] - centre[None, :, :], axis=2)
        if distances.min() <= cutoff:
            near.add(residue.index)
    return sorted(near)


def residue_clusters(
    topology: AmberTopology,
    positions: np.ndarray,
    residues: Sequence[int],
    cutoff: float = 12.0,
) -> list[list[int]]:
    """``residues`` gathered into groups that sit together, nearest first.

    Two residues belong together when any of their atoms are within
    ``cutoff`` of each other, and groups are joined through any member -- so
    a metal, its cofactor and the substrate beside them make one group, and
    each copy of a dimer's active site makes its own. A view framed on all of
    them at once would put the copies so far apart that neither can be read.
    """

    residues = list(residues)
    parent = list(range(len(residues)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    coordinates = [positions[list(topology.residues[index].atom_indices)] for index in residues]
    for first in range(len(residues)):
        for second in range(first + 1, len(residues)):
            gap = np.min(
                np.linalg.norm(
                    coordinates[first][:, None, :] - coordinates[second][None, :, :], axis=-1
                )
            )
            if gap <= cutoff:
                a, b = root(first), root(second)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for index, residue in enumerate(residues):
        groups.setdefault(root(index), []).append(residue)
    return sorted(groups.values(), key=lambda group: group[0])


def display_atoms(
    topology: AmberTopology,
    positions: np.ndarray,
    *,
    near_metal: float = 8.0,
    protein: str = "heavy",
    bulk_water: bool = False,
) -> list[int]:
    """The atoms worth drawing, out of a solvated system of tens of thousands.

    Every atom of every residue within ``near_metal`` of a metal -- the site,
    its hydrogens, its waters. Elsewhere the protein's heavy atoms
    (``protein="heavy"``) or its backbone only (``"backbone"``, for animating
    many frames), every non-water heavy atom, and with ``bulk_water`` the
    oxygen of every other water, enough to see the shape of the box. Drawing
    all of it inflated one trajectory view in a notebook to 80 MB.
    """

    if protein not in {"heavy", "backbone"}:
        raise ValueError("protein must be 'heavy' or 'backbone'")
    metals = [atom.index for atom in topology.atoms if topology.is_metal(atom.index)]
    near = set(residues_near(topology, positions, metals, near_metal)) if metals else set()
    protein_residues = set(
        topology.atoms[index].residue_index for index in protein_atoms(topology, {"CA"})
    )
    selected: list[int] = []
    for residue in topology.residues:
        if residue.index in near:
            selected.extend(residue.atom_indices)
            continue
        water = residue.name in WATER_RESIDUES
        for index in residue.atom_indices:
            atom = topology.atoms[index]
            if water:
                if bulk_water and atom.atomic_number == 8:
                    selected.append(index)
                continue
            if atom.atomic_number == 1:
                continue
            if (
                protein == "backbone"
                and residue.index in protein_residues
                and atom.name not in {"N", "CA", "C", "O"}
            ):
                continue
            selected.append(index)
    return selected


# --------------------------------------------------------------------------
# Imaging and superposition
# --------------------------------------------------------------------------


def image(trajectory: Trajectory) -> Trajectory:
    """Every frame imaged about the solute, as mdprep's snapshots are."""

    if trajectory.cells is None:
        return trajectory
    molecules = trajectory.topology.molecules()
    solute = solute_atoms(trajectory.topology)
    imaged = np.array(
        [
            image_molecules_about_solute(frame, cell, molecules, solute)
            for frame, cell in zip(trajectory.positions, trajectory.cells)
        ]
    )
    return replace(trajectory, positions=imaged)


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_centre, target_centre = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mobile_centre).T @ (target - target_centre))
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, mobile_centre, target_centre


def superpose(trajectory: Trajectory, atoms: Sequence[int], reference: int = 0) -> Trajectory:
    """Every frame fitted onto frame ``reference`` over ``atoms``."""

    selection = list(atoms)
    if len(selection) < 3:
        raise ValueError("Superposition needs at least three atoms")
    target = trajectory.positions[reference][selection]
    fitted = []
    for frame in trajectory.positions:
        rotation, mobile_centre, target_centre = _kabsch(frame[selection], target)
        fitted.append((frame - mobile_centre) @ rotation.T + target_centre)
    return replace(trajectory, positions=np.array(fitted))


def rmsd(trajectory: Trajectory, atoms: Sequence[int], reference: int = 0) -> np.ndarray:
    """RMSD of ``atoms`` to frame ``reference`` after fitting on the same atoms."""

    fitted = superpose(trajectory, atoms, reference).positions[:, list(atoms)]
    return np.sqrt(((fitted - fitted[reference]) ** 2).sum(axis=2).mean(axis=1))


def rmsf(trajectory: Trajectory, atoms: Sequence[int]) -> np.ndarray:
    """Per-atom fluctuation about the mean structure, after fitting on ``atoms``."""

    fitted = superpose(trajectory, atoms).positions[:, list(atoms)]
    return np.sqrt(((fitted - fitted.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))


def distances(trajectory: Trajectory, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """(frames, pairs) distances. Frames should be imaged so pairs are whole."""

    first = trajectory.positions[:, [pair[0] for pair in pairs]]
    second = trajectory.positions[:, [pair[1] for pair in pairs]]
    return np.linalg.norm(first - second, axis=2)


# --------------------------------------------------------------------------
# For the viewer
# --------------------------------------------------------------------------


def _pdb_name(name: str, element: str) -> str:
    if len(name) < 4 and len(element) == 1:
        return f" {name:<3s}"
    return f"{name:<4s}"[:4]


def _chain_letters(topology: AmberTopology, protein: set[int]) -> dict[int, str]:
    """A chain letter for every residue of each molecule with protein in it.

    A prmtop numbers residues straight through, with no chains. Written all
    as one chain, a viewer runs the cartoon on from the last residue of one
    copy of a dimer into the first of the next, and across any loop the
    structure never resolved -- both are separate molecules to Amber, and
    each gets its own letter here. Ligands, waters and ions not bonded to a
    protein are left without one.
    """

    letters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    chains: dict[int, str] = {}
    count = 0
    for molecule in topology.molecules():
        residues = {topology.atoms[index].residue_index for index in molecule}
        if not residues & protein:
            continue
        letter = letters[count % len(letters)]
        count += 1
        for residue in residues:
            chains[residue] = letter
    return chains


def pdb_models(
    trajectory: Trajectory,
    atoms: Sequence[int],
    frames: Sequence[int] | None = None,
    *,
    bfactors: Sequence[float] | None = None,
) -> str:
    """Multi-model PDB text of ``atoms`` in ``frames``, for py3Dmol to animate.

    Residues keep the prmtop's sequential numbering. ``bfactors`` (one per
    selected atom) goes in the B-factor column, which a viewer can colour by:
    charges, or fluctuations.
    """

    topology = trajectory.topology
    selection = list(atoms)
    chosen = range(trajectory.frames) if frames is None else frames
    # ATOM for the protein, HETATM for everything else, which is how a viewer
    # tells a ligand to draw as sticks from a chain to draw as a cartoon.
    protein = {
        topology.atoms[index].residue_index for index in protein_atoms(topology, {"CA"})
    }
    chains = _chain_letters(topology, protein)
    lines: list[str] = []
    for model, frame in enumerate(chosen, start=1):
        lines.append(f"MODEL     {model:4d}\n")
        for serial, index in enumerate(selection, start=1):
            atom = topology.atoms[index]
            residue = topology.residues[atom.residue_index]
            x, y, z = trajectory.positions[frame][index]
            record = "ATOM  " if residue.index in protein else "HETATM"
            bfactor = 0.0 if bfactors is None else float(bfactors[serial - 1])
            element = atom.element
            # Columns: name 13-16, residue name 18-20, chain 22, number 23-26,
            # coordinates from 31, B-factor 61-66, element 77-78.
            lines.append(
                f"{record}{serial % 100000:5d} {_pdb_name(atom.name, element)} "
                f"{residue.name[:3]:>3s} {chains.get(residue.index, ' ')}"
                f"{(residue.index + 1) % 10000:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{bfactor:6.2f}          {element.upper():>2s}\n"
            )
        lines.append("ENDMDL\n")
    lines.append("END\n")
    return "".join(lines)
