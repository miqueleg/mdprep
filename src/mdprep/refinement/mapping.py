"""Coordinate-anchored residue mapping across tleap reordering."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord


class RefinementMappingError(ValueError):
    """Raised when two structure representations cannot be mapped uniquely."""


@dataclass(frozen=True)
class MappedResidue:
    input_index: int
    topology_index: int
    input_residue: ResidueRecord
    topology_residue: ResidueRecord
    anchor_rmsd_angstrom: float


_PROTONATION_FAMILIES = (
    frozenset({"HIS", "HID", "HIE", "HIP"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"CYS", "CYM", "CYX"}),
    frozenset({"LYS", "LYN"}),
    frozenset({"HOH", "WAT", "H2O", "TIP3", "OPC"}),
)


def map_residues_by_coordinates(
    input_structure: PdbStructure,
    topology_structure: PdbStructure,
    *,
    max_anchor_rmsd_angstrom: float = 0.25,
    ambiguity_tolerance_angstrom: float = 1.0e-5,
) -> list[MappedResidue]:
    """Map residues after tleap without relying on output residue order.

    tleap commonly moves loaded ligand templates ahead of waters. Heavy-atom
    coordinates are preserved in the provisional build, so compatible residue
    names plus same-named atom anchors provide a deterministic mapping. This
    also distinguishes hundreds of otherwise identical water residues.
    """

    if len(input_structure.residues) != len(topology_structure.residues):
        raise RefinementMappingError(
            "Provisional tleap changed the residue count, so its atom ordering cannot be "
            f"mapped safely ({len(input_structure.residues)} input residues, "
            f"{len(topology_structure.residues)} output residues)"
        )
    unused = set(range(len(topology_structure.residues)))
    result: list[MappedResidue] = []
    for input_index, input_residue in enumerate(input_structure.residues):
        candidates: list[tuple[float, int, ResidueRecord]] = []
        for topology_index in unused:
            topology_residue = topology_structure.residues[topology_index]
            if not compatible_resnames(
                input_residue.id.resname,
                topology_residue.id.resname,
            ):
                continue
            score = _anchor_rmsd(input_residue, topology_residue)
            if score is not None and score <= max_anchor_rmsd_angstrom:
                candidates.append((score, topology_index, topology_residue))
        if not candidates:
            raise RefinementMappingError(
                "No coordinate-preserving provisional residue matched "
                f"{input_residue.id.display()} within {max_anchor_rmsd_angstrom:.3f} A"
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        best_score, topology_index, topology_residue = candidates[0]
        if (
            len(candidates) > 1
            and abs(candidates[1][0] - best_score) <= ambiguity_tolerance_angstrom
        ):
            raise RefinementMappingError(
                "Coordinate mapping is ambiguous for "
                f"{input_residue.id.display()}: provisional residues "
                f"{topology_index + 1} and {candidates[1][1] + 1} have indistinguishable "
                "anchor coordinates"
            )
        unused.remove(topology_index)
        result.append(
            MappedResidue(
                input_index=input_index,
                topology_index=topology_index,
                input_residue=input_residue,
                topology_residue=topology_residue,
                anchor_rmsd_angstrom=best_score,
            )
        )
    if unused:
        raise RefinementMappingError(
            f"{len(unused)} provisional residues remained unmapped after coordinate matching"
        )
    return result


def compatible_resnames(left: str, right: str) -> bool:
    if left == right:
        return True
    return any(left in family and right in family for family in _PROTONATION_FAMILIES)


def is_virtual_site(atom: AtomRecord) -> bool:
    element = (atom.element or "").strip().upper()
    name = atom.name.strip().upper()
    return element in {"EP", "LP", "X"} or name.startswith("EP") or name.startswith("LP")


def _anchor_rmsd(
    input_residue: ResidueRecord,
    topology_residue: ResidueRecord,
) -> float | None:
    input_by_name = _unique_physical_atoms(input_residue)
    topology_by_name = _unique_physical_atoms(topology_residue)
    common_heavy = sorted(
        name
        for name in input_by_name.keys() & topology_by_name.keys()
        if not _is_hydrogen(input_by_name[name])
        and not _is_hydrogen(topology_by_name[name])
    )
    names = common_heavy or sorted(input_by_name.keys() & topology_by_name.keys())
    if not names:
        return None
    squared = 0.0
    for name in names:
        left = input_by_name[name]
        right = topology_by_name[name]
        if _element(left) != _element(right):
            return None
        squared += (
            (left.x - right.x) ** 2
            + (left.y - right.y) ** 2
            + (left.z - right.z) ** 2
        )
    return sqrt(squared / len(names))


def _unique_physical_atoms(residue: ResidueRecord) -> dict[str, AtomRecord]:
    result: dict[str, AtomRecord] = {}
    duplicates: set[str] = set()
    for atom in residue.atoms:
        if is_virtual_site(atom):
            continue
        if atom.name in result:
            duplicates.add(atom.name)
        result[atom.name] = atom
    for name in duplicates:
        result.pop(name, None)
    return result


def _element(atom: AtomRecord) -> str:
    return (atom.element or "").strip().upper()


def _is_hydrogen(atom: AtomRecord) -> bool:
    return _element(atom) == "H"
