"""Disulfide candidate detection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import dist

from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord


@dataclass(frozen=True)
class DisulfideCandidate:
    a: ResidueId
    b: ResidueId
    distance_angstrom: float

    def to_dict(self) -> dict[str, object]:
        return {
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "distance_angstrom": self.distance_angstrom,
        }


def detect_possible_disulfides(
    residues: list[ResidueRecord],
    *,
    cutoff_angstrom: float = 2.2,
) -> list[DisulfideCandidate]:
    cys_sg_atoms: list[tuple[ResidueRecord, AtomRecord]] = []
    for residue in residues:
        if residue.id.resname not in {"CYS", "CYX"}:
            continue
        sg = next((atom for atom in residue.atoms if atom.name == "SG"), None)
        if sg is not None:
            cys_sg_atoms.append((residue, sg))

    candidates: list[DisulfideCandidate] = []
    for (residue_a, atom_a), (residue_b, atom_b) in combinations(cys_sg_atoms, 2):
        distance_angstrom = dist((atom_a.x, atom_a.y, atom_a.z), (atom_b.x, atom_b.y, atom_b.z))
        if distance_angstrom <= cutoff_angstrom:
            candidates.append(
                DisulfideCandidate(
                    a=residue_a.id,
                    b=residue_b.id,
                    distance_angstrom=distance_angstrom,
                )
            )
    return candidates

