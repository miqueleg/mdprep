"""Evaluate the 12-6-4 C4 term only for the pairs it is nonzero for.

OpenMM builds a prmtop's LENNARD_JONES_CCOEF table into a CustomNonbondedForce,
``-c/r^4; c=ccoef(type1, type2)``, over every pair of atoms in the box. The
table is zero except where one atom is a 12-6-4 ion: C4 is the ion's
ion-induced-dipole attraction. So almost every pair the force visits
contributes nothing, and the visit is not free: it is a second pass over
every pair within the cutoff. With NVIDIA's cuPotential computing the
ordinary nonbonded force, it is also the one pass left that needs OpenMM's
own neighbour list over the whole box.

An interaction group of the ions' atoms against every atom covers each
nonzero pair and nothing else. The forces are those of the full sum; the
energy differs only in how OpenMM averages the long-range correction, by
about 1e-3 kJ/mol on a 27,000-atom box.
"""

from __future__ import annotations

from typing import Any


def _is_c4_force(force: Any, mm: Any) -> bool:
    return (
        isinstance(force, mm.CustomNonbondedForce)
        and force.getNumTabulatedFunctions() == 1
        and force.getTabulatedFunctionName(0) == "ccoef"
        and force.getNumPerParticleParameters() == 1
        and force.getNumInteractionGroups() == 0
    )


def _covering_types(values: Any, size: int) -> set[int]:
    """Atom types such that every nonzero C4 entry has one of them on a side.

    Greedy, the type in the most uncovered pairs first: that is the ion,
    whose row is nonzero against every type, and then nothing is left.
    """

    uncovered = {
        (row, column)
        for row in range(size)
        for column in range(row, size)
        if values[row + size * column] != 0.0 or values[column + size * row] != 0.0
    }
    cover: set[int] = set()
    while uncovered:
        counts: dict[int, int] = {}
        for row, column in uncovered:
            counts[row] = counts.get(row, 0) + 1
            if column != row:
                counts[column] = counts.get(column, 0) + 1
        chosen = max(counts, key=lambda kind: (counts[kind], -kind))
        cover.add(chosen)
        uncovered = {pair for pair in uncovered if chosen not in pair}
    return cover


def restrict_c4_to_ion_pairs(system: Any) -> list[int]:
    """Limit each 12-6-4 C4 force in ``system`` to the pairs with an ion in them.

    Returns the atoms the C4 forces were restricted to: the 12-6-4 ions,
    empty when the system has no C4 term.
    """

    import openmm as mm

    restricted: list[int] = []
    for force in system.getForces():
        if not _is_c4_force(force, mm):
            continue
        xsize, ysize, values = force.getTabulatedFunction(0).getFunctionParameters()
        if xsize != ysize:
            continue
        cover = _covering_types(values, xsize)
        everyone = range(force.getNumParticles())
        ions = [
            index
            for index in everyone
            if round(force.getParticleParameters(index)[0]) in cover
        ]
        if not ions:
            continue
        force.addInteractionGroup(ions, everyone)
        restricted.extend(ions)
    return sorted(set(restricted))
