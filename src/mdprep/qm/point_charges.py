"""MM point-charge extraction for QMMESP electrostatic embedding."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from mdprep.config.models import LigandConfig, ManifestConfig, QmmespConfig
from mdprep.residues.workflow import kept_residue_names
from mdprep.structure.classify import STANDARD_PROTEIN_RESIDUES, WATER_RESIDUES

# Closer than this to a QM atom, an MM atom is overlapping it, not around it.
MINIMUM_EMBEDDING_DISTANCE_ANGSTROM = 1.0


class PointChargeError(ValueError):
    """Raised when QMMESP point charges cannot be extracted safely."""


@dataclass(frozen=True)
class PointCharge:
    x: float
    y: float
    z: float
    charge: float
    residue_name: str
    residue_number: int | None
    atom_name: str
    category: str

    def to_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "charge": self.charge,
            "residue_name": self.residue_name,
            "residue_number": self.residue_number,
            "atom_name": self.atom_name,
            "category": self.category,
        }


@dataclass(frozen=True)
class PointChargeSelection:
    target_atom_indices: list[int]
    point_charges: list[PointCharge]
    total_before_cutoff: int
    total_after_cutoff: int
    net_embedding_charge: float
    min_distance: float | None
    max_distance: float | None
    categories: dict[str, int]
    # A residue of a chain is fitted between caps that stand in for the
    # residues either side of it; those are left out of its embedding whole.
    target_kind: str = "ligand"
    excluded_residues: tuple[str, ...] = ()
    boundary_atoms_excluded: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "target_atom_indices": self.target_atom_indices,
            "target_atom_count": len(self.target_atom_indices),
            f"target_{self.target_kind}_excluded_from_embedding": True,
            "point_charge_count_before_cutoff": self.total_before_cutoff,
            "point_charge_count_after_cutoff": self.total_after_cutoff,
            "net_embedding_charge": self.net_embedding_charge,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "categories": self.categories,
        }
        if self.target_kind == "residue":
            data["excluded_residues"] = list(self.excluded_residues)
            data["boundary_atoms_excluded"] = list(self.boundary_atoms_excluded)
            data["qmmesp_interpretation"] = (
                "The capped residue's SCF uses electrostatic QM/MM embedding through "
                "pyscf.qmmm.mm_charge. MM point charges polarize its QM density; they are "
                "not fitted and are not written to the residue's library. The residue, "
                "the two residues its caps stand in for, and the peptide bonds that join "
                "those to the chain beyond are not among them."
            )
        else:
            data["qmmesp_interpretation"] = (
                "The target-ligand SCF uses electrostatic QM/MM embedding through "
                "pyscf.qmmm.mm_charge. MM point charges polarize the target-ligand QM density; "
                "they are not fitted and are not written to the ligand mol2."
            )
        return data

    @property
    def charge_array(self) -> np.ndarray:
        return np.asarray([charge.charge for charge in self.point_charges], dtype=float)

    @property
    def coordinate_array(self) -> np.ndarray:
        return np.asarray([[charge.x, charge.y, charge.z] for charge in self.point_charges], dtype=float)


def parmed_available() -> bool:
    try:
        import parmed  # noqa: F401
    except Exception:
        return False
    return True


def extract_point_charges_from_prmtop(
    *,
    prmtop: str | Path,
    inpcrd: str | Path,
    ligand: LigandConfig,
    manifest: ManifestConfig,
    target_coordinates: np.ndarray,
    target_atom_names: list[str] | None = None,
) -> PointChargeSelection:
    parmed = _parmed()
    try:
        structure = parmed.load_file(str(prmtop), str(inpcrd))
    except Exception as exc:
        raise PointChargeError(f"Could not load provisional Amber topology with ParmEd: {exc}") from exc
    coordinates = np.asarray(structure.coordinates, dtype=float)
    target_indices = _find_target_ligand_atoms(
        structure,
        ligand,
        topology_coordinates=coordinates,
        target_coordinates=target_coordinates,
        target_atom_names=target_atom_names,
    )
    target_set = set(target_indices)
    target_coords = np.asarray(target_coordinates, dtype=float)
    charges: list[PointCharge] = []
    before_cutoff = 0
    distances_kept: list[float] = []
    categories: dict[str, int] = {}
    protein_names = kept_residue_names(manifest)
    for index, atom in enumerate(structure.atoms):
        if index in target_set:
            continue
        category = _category(atom, protein_names)
        env = ligand.qmmesp.environment if ligand.qmmesp is not None else None
        if env is not None:
            if category == "protein" and not env.include_protein:
                continue
            if category == "water" and not env.include_waters:
                continue
            if category == "ligand" and not env.include_other_ligands:
                continue
        before_cutoff += 1
        distance = float(np.min(np.linalg.norm(target_coords - coordinates[index], axis=1)))
        cutoff = ligand.qmmesp.embedding_cutoff_angstrom if ligand.qmmesp is not None else None
        if cutoff is not None and distance > cutoff:
            continue
        residue = atom.residue
        charge = PointCharge(
            x=float(coordinates[index][0]),
            y=float(coordinates[index][1]),
            z=float(coordinates[index][2]),
            charge=float(atom.charge),
            residue_name=str(residue.name),
            residue_number=getattr(residue, "number", None),
            atom_name=str(atom.name),
            category=category,
        )
        charges.append(charge)
        distances_kept.append(distance)
        categories[category] = categories.get(category, 0) + 1
    return PointChargeSelection(
        target_atom_indices=target_indices,
        point_charges=charges,
        total_before_cutoff=before_cutoff,
        total_after_cutoff=len(charges),
        net_embedding_charge=float(sum(charge.charge for charge in charges)),
        min_distance=min(distances_kept) if distances_kept else None,
        max_distance=max(distances_kept) if distances_kept else None,
        categories=categories,
    )


def write_point_charge_files(selection: PointChargeSelection, *, csv_path: str | Path, xyz_path: str | Path, summary_path: str | Path) -> None:
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y", "z", "charge", "residue_name", "residue_number", "atom_name", "category"])
        writer.writeheader()
        for charge in selection.point_charges:
            writer.writerow(charge.to_dict())
    xyz_output = Path(xyz_path)
    lines = [str(len(selection.point_charges)), "mdprep MM point charges"]
    for charge in selection.point_charges:
        lines.append(f"X {charge.x:.8f} {charge.y:.8f} {charge.z:.8f} {charge.charge:.8f}")
    xyz_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(summary_path).write_text(json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_target_ligand_atoms(
    structure: object,
    ligand: LigandConfig,
    *,
    topology_coordinates: np.ndarray,
    target_coordinates: np.ndarray,
    target_atom_names: list[str] | None,
    coordinate_tolerance_angstrom: float = 0.1,
) -> list[int]:
    target = np.asarray(target_coordinates, dtype=float)
    if target.ndim != 2 or target.shape[1] != 3 or len(target) == 0:
        raise PointChargeError("Target-ligand coordinates must have shape (n_atoms, 3) and be non-empty.")
    if target_atom_names is not None and len(target_atom_names) != len(target):
        raise PointChargeError("Target-ligand atom-name and coordinate counts do not match.")

    matches: list[tuple[list[int], float]] = []
    rejected: list[str] = []
    for residue in structure.residues:
        if residue.name != ligand.selector.resname:
            continue
        indices = [atom.idx for atom in residue.atoms]
        if len(indices) != len(target):
            rejected.append(f"residue {getattr(residue, 'number', '?')}: {len(indices)} atoms")
            continue
        names = [str(atom.name) for atom in residue.atoms]
        if target_atom_names is not None and names != target_atom_names:
            rejected.append(f"residue {getattr(residue, 'number', '?')}: atom names {names}")
            continue
        candidate = topology_coordinates[indices]
        max_deviation = float(np.max(np.linalg.norm(candidate - target, axis=1)))
        if max_deviation <= coordinate_tolerance_angstrom:
            matches.append((indices, max_deviation))
        else:
            rejected.append(
                f"residue {getattr(residue, 'number', '?')}: coordinate max deviation "
                f"{max_deviation:.3f} A"
            )
    if len(matches) != 1:
        details = "; ".join(rejected) if rejected else "no residue-name candidates"
        raise PointChargeError(
            f"Could not map ligand {ligand.id} uniquely in provisional topology; "
            "mapping requires residue name, atom count/order/names, and coordinates to agree. "
            f"Candidates: {details}."
        )
    return matches[0][0]


def extract_residue_point_charges(
    *,
    prmtop: str | Path,
    inpcrd: str | Path,
    resname: str,
    residue_coordinates: Mapping[str, Sequence[float]],
    qm_coordinates: np.ndarray,
    qmmesp: QmmespConfig,
    manifest: ManifestConfig,
    label: str,
    expected_neighbours: Sequence[tuple[str, str]] | None = None,
    minimum_distance_angstrom: float = MINIMUM_EMBEDDING_DISTANCE_ANGSTROM,
) -> PointChargeSelection:
    """MM point charges around a residue of a chain fitted between caps.

    The residue is found in the provisional topology by name and by where
    each of its atoms is -- tleap orders a library residue's atoms its own
    way, so not by order. The residues bonded to it are the ones whose atoms
    the caps are made of, and they are left out whole rather than atom by
    atom: a cap's hydrogens sit almost where those residues' atoms are. So
    are the peptide bonds that join them to the chain beyond, a bond's length
    from those hydrogens. Every other atom is a point charge, as the
    manifest's environment settings and cutoff say. ``qm_coordinates`` are the capped residue's, caps included:
    the cutoff and the overlap check are measured from all of them.

    ``expected_neighbours`` are the residue names and the structure's labels of the
    residue before and the residue after, the ones the caps were built from:
    the topology's must be those, and are reported by those labels rather
    than by tleap's numbering.
    """

    parmed = _parmed()
    try:
        structure = parmed.load_file(str(prmtop), str(inpcrd))
    except Exception as exc:
        raise PointChargeError(f"Could not load provisional Amber topology with ParmEd: {exc}") from exc
    coordinates = np.asarray(structure.coordinates, dtype=float)
    wanted = {name: np.asarray(xyz, dtype=float) for name, xyz in residue_coordinates.items()}
    matches = []
    rejected: list[str] = []
    for residue in structure.residues:
        if str(residue.name) != resname:
            continue
        names = {str(atom.name): atom.idx for atom in residue.atoms}
        number = getattr(residue, "number", "?")
        if set(names) != set(wanted):
            rejected.append(f"residue {number}: atoms {sorted(names)}")
            continue
        deviation = max(
            float(np.linalg.norm(coordinates[names[name]] - wanted[name])) for name in names
        )
        if deviation <= 0.1:
            matches.append(residue)
        else:
            rejected.append(f"residue {number}: atoms up to {deviation:.3f} A away")
    if len(matches) != 1:
        raise PointChargeError(
            f"Could not find {label} uniquely in the provisional topology by name, atom "
            "names and coordinates. Candidates: "
            + ("; ".join(rejected) if rejected else "none of that name")
            + "."
        )
    (target,) = matches
    target_indices = sorted(int(atom.idx) for atom in target.atoms)
    links: dict[int, list[tuple[str, str]]] = {}
    neighbours: dict[int, object] = {}
    for atom in target.atoms:
        for partner in atom.bond_partners:
            if partner.residue is target:
                continue
            neighbours[partner.residue.idx] = partner.residue
            links.setdefault(partner.residue.idx, []).append((str(atom.name), str(partner.name)))
    before = [index for index, pairs in links.items() if pairs == [("N", "C")]]
    after = [index for index, pairs in links.items() if pairs == [("C", "N")]]
    peptide = before + after
    if len(before) != 1 or len(after) != 1 or len(links) != 2:
        described = "; ".join(
            f"{neighbours[index].name}{getattr(neighbours[index], 'number', '?')} by "
            + ", ".join(f"{own}-{theirs}" for own, theirs in pairs)
            for index, pairs in sorted(links.items())
        )
        raise PointChargeError(
            f"{label} is bonded in the provisional topology to {described or 'nothing'}. "
            "Its QM model is capped at the two peptide bonds of the chain and nothing "
            "else, so it must be bonded to the residue before it by N and to the one "
            "after it by C, and to nothing more."
        )
    labels = [
        f"{neighbours[index].name}{getattr(neighbours[index], 'number', '?')}" for index in peptide
    ]
    if expected_neighbours is not None:
        found = [str(neighbours[index].name) for index in peptide]
        wanted_names = [name for name, _ in expected_neighbours]
        if [_family(name) for name in found] != [_family(name) for name in wanted_names]:
            raise PointChargeError(
                f"{label} is bonded in the provisional topology to {found[0]} before it and "
                f"{found[1]} after it, but its caps were built from {wanted_names[0]} and "
                f"{wanted_names[1]}."
            )
        labels = [display for _, display in expected_neighbours]
    excluded = set(target_indices)
    for index in peptide:
        excluded.update(int(atom.idx) for atom in neighbours[index].atoms)
    # The peptide bonds one residue further out: the carbonyl before the
    # residue before, the amide after the residue after. A cap's methyl
    # hydrogens stand where the neighbour's N and C were, so these atoms are
    # a bond's length from the QM model -- bonded to it, in effect -- and
    # are left out with the neighbours, as a QM/MM boundary leaves out the
    # MM atoms bonded across it.
    boundary: list[str] = []
    for index in peptide:
        for atom in neighbours[index].atoms:
            for partner in atom.bond_partners:
                if partner.residue is neighbours[index] or partner.residue is target:
                    continue
                group = [partner] + [
                    other
                    for other in partner.bond_partners
                    if other.residue is partner.residue and len(other.bond_partners) == 1
                ]
                for member in group:
                    if int(member.idx) not in excluded:
                        excluded.add(int(member.idx))
                        boundary.append(f"{member.residue.name}@{member.name}")
    qm = np.asarray(qm_coordinates, dtype=float)
    environment = qmmesp.environment
    protein_names = kept_residue_names(manifest)
    charges: list[PointCharge] = []
    distances: list[float] = []
    categories: dict[str, int] = {}
    before_cutoff = 0
    overlapping: list[str] = []
    for index, atom in enumerate(structure.atoms):
        if index in excluded:
            continue
        category = _category(atom, protein_names)
        if category == "protein" and not environment.include_protein:
            continue
        if category == "water" and not environment.include_waters:
            continue
        if category == "ligand" and not environment.include_other_ligands:
            continue
        before_cutoff += 1
        distance = float(np.min(np.linalg.norm(qm - coordinates[index], axis=1)))
        if distance < minimum_distance_angstrom:
            overlapping.append(
                f"{atom.residue.name}{getattr(atom.residue, 'number', '?')}@{atom.name} "
                f"({distance:.2f} A)"
            )
            continue
        if qmmesp.embedding_cutoff_angstrom is not None and distance > qmmesp.embedding_cutoff_angstrom:
            continue
        residue = atom.residue
        charges.append(
            PointCharge(
                x=float(coordinates[index][0]),
                y=float(coordinates[index][1]),
                z=float(coordinates[index][2]),
                charge=float(atom.charge),
                residue_name=str(residue.name),
                residue_number=getattr(residue, "number", None),
                atom_name=str(atom.name),
                category=category,
            )
        )
        distances.append(distance)
        categories[category] = categories.get(category, 0) + 1
    if overlapping:
        raise PointChargeError(
            f"MM atoms lie within {minimum_distance_angstrom:.1f} A of {label}'s QM model: "
            + ", ".join(overlapping[:8])
            + ". A point charge that close is not an environment but a clash; look at "
            "the structure there before fitting charges in it."
        )
    return PointChargeSelection(
        target_atom_indices=target_indices,
        point_charges=charges,
        total_before_cutoff=before_cutoff,
        total_after_cutoff=len(charges),
        net_embedding_charge=float(sum(charge.charge for charge in charges)),
        min_distance=min(distances) if distances else None,
        max_distance=max(distances) if distances else None,
        categories=categories,
        target_kind="residue",
        excluded_residues=tuple(labels),
        boundary_atoms_excluded=tuple(boundary),
    )


# One residue under the names its protonation and bonding give it.
_FAMILIES = {
    name: family
    for family, names in {
        "HIS": ("HIS", "HID", "HIE", "HIP"),
        "CYS": ("CYS", "CYX", "CYM"),
        "ASP": ("ASP", "ASH"),
        "GLU": ("GLU", "GLH"),
        "LYS": ("LYS", "LYN"),
    }.items()
    for name in names
}


def _family(name: str) -> str:
    return _FAMILIES.get(name.strip().upper(), name.strip().upper())


def _parmed():
    try:
        import parmed
    except Exception as exc:
        raise PointChargeError("ParmEd is required for qmmesp_pyscf point-charge extraction.") from exc
    return parmed


def _category(atom: object, protein_names: frozenset[str] = frozenset()) -> str:
    name = str(atom.residue.name).upper()
    if name in WATER_RESIDUES:
        return "water"
    if name in STANDARD_PROTEIN_RESIDUES or name in protein_names:
        return "protein"
    return "ligand"
