"""MM point-charge extraction for QMMESP-like embedding."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdprep.config.models import LigandConfig, ManifestConfig
from mdprep.structure.classify import STANDARD_PROTEIN_RESIDUES, WATER_RESIDUES


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

    def to_dict(self) -> dict[str, object]:
        return {
            "target_atom_indices": self.target_atom_indices,
            "target_atom_count": len(self.target_atom_indices),
            "target_ligand_excluded_from_embedding": True,
            "point_charge_count_before_cutoff": self.total_before_cutoff,
            "point_charge_count_after_cutoff": self.total_after_cutoff,
            "net_embedding_charge": self.net_embedding_charge,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "categories": self.categories,
            "qmmesp_interpretation": (
                "MM point charges are used only for electrostatic embedding/polarization of the "
                "target ligand QM density; they are not fitted and are not written to the ligand mol2."
            ),
        }

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
) -> PointChargeSelection:
    try:
        import parmed
    except Exception as exc:
        raise PointChargeError("ParmEd is required for qmmesp_pyscf point-charge extraction.") from exc
    try:
        structure = parmed.load_file(str(prmtop), str(inpcrd))
    except Exception as exc:
        raise PointChargeError(f"Could not load provisional Amber topology with ParmEd: {exc}") from exc
    coordinates = np.asarray(structure.coordinates, dtype=float)
    target_indices = _find_target_ligand_atoms(structure, ligand)
    target_set = set(target_indices)
    target_coords = np.asarray(target_coordinates, dtype=float)
    charges: list[PointCharge] = []
    before_cutoff = 0
    distances_kept: list[float] = []
    categories: dict[str, int] = {}
    for index, atom in enumerate(structure.atoms):
        if index in target_set:
            continue
        category = _category(atom)
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
        if ligand.qmmesp and distance > ligand.qmmesp.embedding_cutoff_angstrom:
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


def _find_target_ligand_atoms(structure: object, ligand: LigandConfig) -> list[int]:
    matches: list[list[int]] = []
    for residue in structure.residues:
        if residue.name != ligand.selector.resname:
            continue
        indices = [atom.idx for atom in residue.atoms]
        if len(indices) == 0:
            continue
        matches.append(indices)
    if len(matches) != 1:
        raise PointChargeError(
            f"Could not map ligand {ligand.id} uniquely in provisional topology; "
            "use unique ligand residue names or selectors."
        )
    return matches[0]


def _category(atom: object) -> str:
    name = str(atom.residue.name).upper()
    if name in WATER_RESIDUES:
        return "water"
    if name in STANDARD_PROTEIN_RESIDUES:
        return "protein"
    return "ligand"
