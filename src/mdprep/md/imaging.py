"""Putting a periodic snapshot back into the box the user asked for.

A truncated octahedron is stored as a triclinic cell, and OpenMM's
``enforcePeriodicBox`` wraps every molecule into the parallelepiped spanned
by that cell's three vectors, with a corner at the origin. That is the same
lattice, but it is not the shape anyone asked for: the solvent fills a
sheared prism, the solute sits wherever it happened to be rather than in the
middle, and water that was beside the protein is wrapped to the far corner.
Measured on a real run, water reached 54 A from the solute in tleap's
octahedron and 70 A in the wrapped snapshot.

So a snapshot is imaged the way cpptraj's ``autoimage`` images one: put the
solute at the origin, and give every other molecule the periodic image
nearest to it. For a truncated octahedron that is the octahedron back, and
it matches the convention tleap already wrote the solvated system in, so the
system looks the same before and after the MD rather than only after
somebody re-images it.

The nearest image is searched over the 27 translations with coefficients in
{-1, 0, 1}. That is enough for any cell: the Wigner-Seitz cell of a lattice
touches only its nearest neighbours, and for the body-centred lattice a
truncated octahedron tiles, those 14 neighbours all lie in that set.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import numpy as np


_TRANSLATIONS = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)


def image_molecules_about_solute(
    positions: np.ndarray,
    box_vectors: np.ndarray,
    molecules: Sequence[Sequence[int]],
    solute_atoms: Iterable[int],
) -> np.ndarray:
    """Centre ``solute_atoms`` and bring every molecule to its nearest image.

    ``positions`` is (natoms, 3) and ``box_vectors`` is (3, 3) with one cell
    vector per row, both in the same length unit. Molecules must already be
    whole, which is what OpenMM's ``enforcePeriodicBox`` guarantees.

    Returns a new array; the input is not modified.
    """

    coordinates = np.array(positions, dtype=float, copy=True)
    cell = np.array(box_vectors, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(f"box_vectors must be 3x3, got {cell.shape}")

    solute = np.asarray(list(solute_atoms), dtype=int)
    if solute.size == 0:
        raise ValueError("No solute atoms to centre on")
    coordinates -= coordinates[solute].mean(axis=0)

    # Rows of `cell` are the lattice vectors, so a point's fractional
    # coordinates are r @ inv(cell) and a translation by integer n moves it
    # by n @ cell.
    inverse = np.linalg.inv(cell)
    shifts = _TRANSLATIONS @ cell
    for molecule in molecules:
        index = np.asarray(molecule, dtype=int)
        if index.size == 0:
            continue
        centre = coordinates[index].mean(axis=0)
        # Into the parallelepiped around the origin first, so the search
        # below only ever has to look at neighbouring cells. Positions from
        # OpenMM are already inside one cell, but nothing here depends on
        # that being true.
        coarse = -np.round(centre @ inverse) @ cell
        distances = np.linalg.norm(centre + coarse + shifts, axis=1)
        best = coarse + shifts[int(np.argmin(distances))]
        if best.any():
            coordinates[index] += best
    return coordinates


def solvent_residue_names() -> frozenset[str]:
    """Residue names tleap gives the solvent it adds.

    The solute is everything else, so this is the only list that has to be
    right: a name missing here puts a water in the group the box is centred
    on, which moves the centre by a little, and a solute residue wrongly
    named here moves it by a lot.
    """

    from mdprep.structure.classify import WATER_RESIDUES

    ions = {
        "NA", "NA+", "CL", "CL-", "K", "K+", "LI", "LI+", "RB", "RB+",
        "CS", "CS+", "MG", "MG2", "CA", "CA2", "BR", "BR-", "IOD", "I-",
        "F", "F-", "ZN", "ZN2",
    }
    return frozenset(WATER_RESIDUES | ions)


def solute_atom_indices(topology) -> list[int]:
    """Atoms of everything that is not added solvent, by residue name.

    Falls back to every atom when a system is nothing but solvent, which is
    not a system mdprep prepares but is a sane thing for this not to crash
    on.
    """

    solvent = solvent_residue_names()
    indices = [
        atom.index
        for atom in topology.atoms()
        if atom.residue.name.strip().upper() not in solvent
    ]
    return indices or [atom.index for atom in topology.atoms()]


def write_imaged_pdb(path, topology, state, molecules, solute_atoms):
    """Write one snapshot: imaged about the solute, in the box it is in now.

    Both halves have to come from ``state`` rather than from the topology the
    run was set up with. The box, because NPT changes it -- every snapshot
    from step 7 on used to declare the pre-NPT cell, 83.4 A against the
    79.9 A the system had reached, so anything rebuilding periodic images
    from the header got them wrong. And the positions, because OpenMM wraps
    them into the parallelepiped rather than the cell the manifest asked
    for.
    """

    from openmm import unit
    from openmm.app import PDBFile

    cell = state.getPeriodicBoxVectors()
    imaged = image_molecules_about_solute(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        np.array(cell.value_in_unit(unit.nanometer)),
        molecules,
        solute_atoms,
    )
    # PDBFile takes CRYST1 from the topology, so it has to be told the box
    # this state is in before every write.
    topology.setPeriodicBoxVectors(cell)
    with open(path, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, imaged * unit.nanometer, handle, keepIds=True)
    return path
