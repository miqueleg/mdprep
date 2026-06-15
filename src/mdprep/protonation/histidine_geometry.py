"""Temporary histidine tautomer geometry for xTB ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist, sqrt
from pathlib import Path

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
    )


def build_tautomer_xyz_atoms(
    cluster_residues: list[ResidueRecord],
    histidine: ResidueRecord,
    *,
    tautomer: str,
) -> list[XyzAtom]:
    atoms: list[XyzAtom] = []
    for residue in cluster_residues:
        for atom in residue.atoms:
            if is_hydrogen_like(atom):
                continue
            atoms.append(XyzAtom(element=_element(atom), x=atom.x, y=atom.y, z=atom.z, name=atom.name))
    atoms.append(place_histidine_tautomer_hydrogen(histidine, tautomer=tautomer))
    return atoms


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


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm < 1.0e-8:
        raise HistidineGeometryError("Degenerate histidine geometry; cannot place tautomer hydrogen.")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _element(atom: AtomRecord) -> str:
    return atom.element or atom.name.strip()[0].upper()

