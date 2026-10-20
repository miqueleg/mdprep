"""Which atoms sit in a metal's first coordination shell, judged by distance.

One definition, used in two places that have to agree. The decision engine
proposes a metal's shell from it, and tells the user that a water bound to a
metal stays whatever they answer about crystal waters; normalisation keeps
those same waters. Two copies of "bound" would let the question promise one
set of waters while the build kept another.
"""

from __future__ import annotations

from typing import Any, Sequence

from mdprep.structure.classify import is_water_residue
from mdprep.structure.models import ResidueRecord

# First-shell metal-donor distances sit near 2.0-2.4 A. 2.6 A admits a long
# halide or carboxylate contact while excluding the second-shell carbons that
# sit behind a coordinating oxygen, which a 3.0 A search picks up.
COORDINATION_CUTOFF_ANGSTROM = 2.6

# Carbon and hydrogen are not metal donors; admitting them turns the atom
# behind a coordinating oxygen into a spurious bond.
METAL_DONOR_ELEMENTS = {"N", "O", "S", "SE", "P", "F", "CL", "BR", "I"}


def metal_donor_atoms(
    residues: Sequence[ResidueRecord],
    metal: ResidueRecord,
    *,
    cutoff_angstrom: float = COORDINATION_CUTOFF_ANGSTROM,
) -> list[tuple[ResidueRecord, Any, float]]:
    """Candidate first-shell donors of a metal, nearest first.

    This proposes a coordination shell for a user to confirm; it does not
    decide one. mdprep requires coordinating bonds to be explicit in the
    manifest, and the questions built from this keep them so.
    """

    origin = metal.atoms[0]
    found: list[tuple[ResidueRecord, Any, float]] = []
    for residue in residues:
        if residue.id == metal.id:
            continue
        for atom in residue.atoms:
            element = (atom.element or "").strip().upper()
            if element not in METAL_DONOR_ELEMENTS:
                continue
            distance = (
                (atom.x - origin.x) ** 2
                + (atom.y - origin.y) ** 2
                + (atom.z - origin.z) ** 2
            ) ** 0.5
            if distance <= cutoff_angstrom:
                found.append((residue, atom, distance))
    found.sort(key=lambda item: item[2])
    return found


def metal_bound_waters(
    residues: Sequence[ResidueRecord], metals: Sequence[ResidueRecord]
) -> list[ResidueRecord]:
    """Waters in the first shell of any of ``metals``, each once, in order."""

    found: list[ResidueRecord] = []
    for metal in metals:
        for residue, _, _ in metal_donor_atoms(residues, metal):
            if is_water_residue(residue) and all(residue.id != seen.id for seen in found):
                found.append(residue)
    return found
