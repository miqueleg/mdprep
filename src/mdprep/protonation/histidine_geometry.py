"""Temporary histidine tautomer geometry for xTB ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, sqrt
from pathlib import Path

from mdprep.structure.classify import is_standard_protein_residue, is_water_residue
from mdprep.structure.models import AtomRecord, ResidueRecord


class HistidineGeometryError(ValueError):
    """Raised when temporary histidine tautomer geometry cannot be built."""


@dataclass(frozen=True)
class XyzAtom:
    element: str
    x: float
    y: float
    z: float
    name: str
    source: str = "input"


@dataclass(frozen=True)
class TautomerClusterModel:
    atoms: list[XyzAtom]
    fixed_atom_indices: list[int]
    cap_atom_indices: list[int]
    anchor_atom_indices: list[int]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_count": len(self.atoms),
            "fixed_atom_indices": self.fixed_atom_indices,
            "cap_atom_indices": self.cap_atom_indices,
            "anchor_atom_indices": self.anchor_atom_indices,
            "warnings": self.warnings,
        }


def place_histidine_tautomer_hydrogen(
    residue: ResidueRecord,
    *,
    tautomer: str,
    bond_length: float = 1.01,
) -> XyzAtom:
    ring = {atom.name: atom for atom in residue.atoms}
    if tautomer == "HID":
        required = ("ND1", "CG", "CE1")
        atom_name = "HD1"
        nitrogen_name = "ND1"
        neighbor_names = ("CG", "CE1")
    elif tautomer == "HIE":
        required = ("NE2", "CE1", "CD2")
        atom_name = "HE2"
        nitrogen_name = "NE2"
        neighbor_names = ("CE1", "CD2")
    else:
        raise HistidineGeometryError(f"Unsupported histidine tautomer {tautomer!r}")
    missing = [name for name in required if name not in ring]
    if missing:
        raise HistidineGeometryError(
            f"Histidine {residue.id.display()} is missing required atoms for {tautomer}: {', '.join(missing)}. "
            "Add a manual override if tautomer geometry cannot be generated."
        )

    nitrogen = ring[nitrogen_name]
    neighbor1 = ring[neighbor_names[0]]
    neighbor2 = ring[neighbor_names[1]]
    direction = (
        2.0 * nitrogen.x - neighbor1.x - neighbor2.x,
        2.0 * nitrogen.y - neighbor1.y - neighbor2.y,
        2.0 * nitrogen.z - neighbor1.z - neighbor2.z,
    )
    unit = _normalize(direction)
    return XyzAtom(
        element="H",
        x=nitrogen.x + unit[0] * bond_length,
        y=nitrogen.y + unit[1] * bond_length,
        z=nitrogen.z + unit[2] * bond_length,
        name=atom_name,
        source="histidine_tautomer",
    )


def build_tautomer_xyz_atoms(
    cluster_residues: list[ResidueRecord],
    histidine: ResidueRecord,
    *,
    tautomer: str,
) -> list[XyzAtom]:
    return build_tautomer_cluster_model(cluster_residues, histidine, tautomer=tautomer).atoms


def build_tautomer_cluster_model(
    cluster_residues: list[ResidueRecord],
    histidine: ResidueRecord,
    *,
    tautomer: str,
) -> TautomerClusterModel:
    atoms: list[XyzAtom] = []
    fixed_atom_indices: list[int] = []
    cap_atom_indices: list[int] = []
    anchor_atom_indices: list[int] = []
    warnings: list[str] = []
    for residue in cluster_residues:
        if is_standard_protein_residue(residue):
            before = len(atoms)
            fixed = _append_truncated_protein_residue(
                atoms,
                residue,
                target_histidine=residue is histidine,
            )
            fixed_atom_indices.extend(fixed["fixed"])
            cap_atom_indices.extend(fixed["caps"])
            anchor_atom_indices.extend(fixed["anchors"])
            _require_retained_hydrogens(residue, atoms[before:])
            continue
        if is_water_residue(residue):
            before = len(atoms)
            _append_full_residue(atoms, residue)
            _require_water_hydrogens(residue, atoms[before:])
            continue
        _append_full_residue(atoms, residue)

    atoms.append(place_histidine_tautomer_hydrogen(histidine, tautomer=tautomer))
    return TautomerClusterModel(
        atoms=atoms,
        fixed_atom_indices=sorted(set(fixed_atom_indices)),
        cap_atom_indices=sorted(set(cap_atom_indices)),
        anchor_atom_indices=sorted(set(anchor_atom_indices)),
        warnings=warnings,
    )


def write_xcontrol_fix_file(fixed_atom_indices: list[int], path: str | Path) -> None:
    path_obj = Path(path)
    if not fixed_atom_indices:
        path_obj.write_text("", encoding="utf-8")
        return
    atom_text = ",".join(str(index) for index in sorted(set(fixed_atom_indices)))
    path_obj.write_text(f"$fix\n atoms: {atom_text}\n$end\n", encoding="utf-8")


def write_xyz(atoms: list[XyzAtom], path: str | Path, *, comment: str = "mdprep histidine tautomer") -> None:
    path_obj = Path(path)
    path_obj.write_text(
        "\n".join(
            [
                str(len(atoms)),
                comment,
                *[f"{atom.element} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}" for atom in atoms],
                "",
            ]
        ),
        encoding="utf-8",
    )


def is_hydrogen_like(atom: AtomRecord) -> bool:
    element_field = atom.original_line[76:78].strip() if len(atom.original_line) >= 78 else ""
    if element_field:
        return element_field.upper() == "H"
    stripped = atom.name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    return stripped.upper().startswith("H")


def heavy_atom_distance(a: AtomRecord, b: AtomRecord) -> float:
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))


BACKBONE_ATOMS = {"N", "H", "H1", "H2", "H3", "C", "O", "OXT"}
TAUTOMER_HYDROGENS = {"HD1", "HE2", "1HD1", "2HD1", "1HE2", "2HE2"}


def _append_truncated_protein_residue(
    atoms: list[XyzAtom],
    residue: ResidueRecord,
    *,
    target_histidine: bool,
) -> dict[str, list[int]]:
    atom_by_name = {atom.name.strip(): atom for atom in residue.atoms}
    ca = atom_by_name.get("CA")
    if ca is None:
        raise HistidineGeometryError(
            f"Protein residue {residue.id.display()} is missing CA; cannot build capped xTB cluster."
        )
    fixed: list[int] = []
    caps: list[int] = []
    anchors: list[int] = []
    for atom in residue.atoms:
        name = atom.name.strip()
        if name in BACKBONE_ATOMS:
            continue
        if target_histidine and name in TAUTOMER_HYDROGENS:
            continue
        atoms.append(_xyz_from_atom(atom))
        if name == "CA":
            fixed.append(len(atoms))
            anchors.append(len(atoms))

    for boundary_name, cap_name in [("N", "HCA_NCAP"), ("C", "HCA_CCAP")]:
        boundary = atom_by_name.get(boundary_name)
        if boundary is None:
            continue
        cap = _cap_hydrogen_from_ca(ca, boundary, cap_name)
        atoms.append(cap)
        fixed.append(len(atoms))
        caps.append(len(atoms))
    return {"fixed": fixed, "caps": caps, "anchors": anchors}


def _append_full_residue(atoms: list[XyzAtom], residue: ResidueRecord) -> None:
    for atom in residue.atoms:
        atoms.append(_xyz_from_atom(atom))


def _require_retained_hydrogens(residue: ResidueRecord, fragment_atoms: list[XyzAtom]) -> None:
    non_cap_hydrogens = [
        atom for atom in fragment_atoms if atom.element.upper() == "H" and atom.source != "cap"
    ]
    if not non_cap_hydrogens:
        raise HistidineGeometryError(
            f"xTB histidine cluster requires a hydrogenated protein model; "
            f"residue {residue.id.display()} has no retained hydrogens after CA truncation. "
            "Provide an input structure with hydrogens or add a manual HIS override."
        )


def _require_water_hydrogens(residue: ResidueRecord, atoms: list[XyzAtom]) -> None:
    hydrogens = [atom for atom in atoms if atom.element.upper() == "H"]
    if len(hydrogens) < 2:
        raise HistidineGeometryError(
            f"Water residue {residue.id.display()} is in the xTB histidine cluster but lacks hydrogens. "
            "Use a hydrogenated input structure, remove nearby waters, or add a manual HIS override."
        )


def _xyz_from_atom(atom: AtomRecord) -> XyzAtom:
    return XyzAtom(element=_element(atom), x=atom.x, y=atom.y, z=atom.z, name=atom.name, source="input")


def _cap_hydrogen_from_ca(ca: AtomRecord, boundary: AtomRecord, name: str, bond_length: float = 1.09) -> XyzAtom:
    direction = (boundary.x - ca.x, boundary.y - ca.y, boundary.z - ca.z)
    unit = _normalize(direction)
    return XyzAtom(
        element="H",
        x=ca.x + unit[0] * bond_length,
        y=ca.y + unit[1] * bond_length,
        z=ca.z + unit[2] * bond_length,
        name=name,
        source="cap",
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm < 1.0e-8:
        raise HistidineGeometryError("Degenerate histidine geometry; cannot place tautomer hydrogen.")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _element(atom: AtomRecord) -> str:
    return atom.element or atom.name.strip()[0].upper()
