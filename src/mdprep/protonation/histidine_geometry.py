"""Temporary histidine tautomer geometry for xTB ranking."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, dist, radians, sin, sqrt
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
class TemporaryWaterHydrogenRecord:
    chain: str
    resname: str
    resid: int
    icode: str | None
    hydrogens_added: int
    final_hydrogen_count: int
    orientation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "chain": self.chain,
            "resname": self.resname,
            "resid": self.resid,
            "icode": self.icode,
            "hydrogens_added": self.hydrogens_added,
            "final_hydrogen_count": self.final_hydrogen_count,
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class TautomerClusterModel:
    atoms: list[XyzAtom]
    fixed_atom_indices: list[int]
    cap_atom_indices: list[int]
    anchor_atom_indices: list[int]
    temporary_water_hydrogens: list[TemporaryWaterHydrogenRecord]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "atom_count": len(self.atoms),
            "fixed_atom_indices": self.fixed_atom_indices,
            "cap_atom_indices": self.cap_atom_indices,
            "anchor_atom_indices": self.anchor_atom_indices,
            "temporary_water_hydrogens_added": sum(
                item.hydrogens_added for item in self.temporary_water_hydrogens
            ),
            "waters_modified_for_xtb_only": [
                item.to_dict() for item in self.temporary_water_hydrogens
            ],
            "final_pdb_modified": False,
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
    residue_states: dict[int, str] | None = None,
    add_missing_water_hydrogens: bool = True,
    water_oh_distance_angstrom: float = 0.9572,
    water_hoh_angle_degrees: float = 104.52,
) -> list[XyzAtom]:
    return build_tautomer_cluster_model(
        cluster_residues,
        histidine,
        tautomer=tautomer,
        residue_states=residue_states,
        add_missing_water_hydrogens=add_missing_water_hydrogens,
        water_oh_distance_angstrom=water_oh_distance_angstrom,
        water_hoh_angle_degrees=water_hoh_angle_degrees,
    ).atoms


def build_tautomer_cluster_model(
    cluster_residues: list[ResidueRecord],
    histidine: ResidueRecord,
    *,
    tautomer: str,
    residue_states: dict[int, str] | None = None,
    add_missing_water_hydrogens: bool = True,
    water_oh_distance_angstrom: float = 0.9572,
    water_hoh_angle_degrees: float = 104.52,
) -> TautomerClusterModel:
    states = {} if residue_states is None else residue_states
    atoms: list[XyzAtom] = []
    fixed_atom_indices: list[int] = []
    cap_atom_indices: list[int] = []
    anchor_atom_indices: list[int] = []
    temporary_water_hydrogens: list[TemporaryWaterHydrogenRecord] = []
    warnings: list[str] = []
    for residue in cluster_residues:
        if is_standard_protein_residue(residue):
            before = len(atoms)
            fixed = _append_truncated_protein_residue(
                atoms,
                residue,
                target_histidine=residue is histidine,
                tautomer=tautomer if residue is histidine else None,
                residue_state=states.get(id(residue), residue.id.resname),
            )
            fixed_atom_indices.extend(fixed["fixed"])
            cap_atom_indices.extend(fixed["caps"])
            anchor_atom_indices.extend(fixed["anchors"])
            _require_retained_hydrogens(residue, atoms[before:])
            continue
        if is_water_residue(residue):
            record = _append_water_residue(
                atoms,
                residue,
                cluster_residues=cluster_residues,
                fixed_atom_indices=fixed_atom_indices,
                add_missing_hydrogens=add_missing_water_hydrogens,
                oh_distance=water_oh_distance_angstrom,
                hoh_angle_degrees=water_hoh_angle_degrees,
                warnings=warnings,
            )
            if record is not None:
                temporary_water_hydrogens.append(record)
            continue
        fixed_atom_indices.extend(_append_full_residue(atoms, residue))

    atoms.append(place_histidine_tautomer_hydrogen(histidine, tautomer=tautomer))
    return TautomerClusterModel(
        atoms=atoms,
        fixed_atom_indices=sorted(set(fixed_atom_indices)),
        cap_atom_indices=sorted(set(cap_atom_indices)),
        anchor_atom_indices=sorted(set(anchor_atom_indices)),
        temporary_water_hydrogens=temporary_water_hydrogens,
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


def has_water_oxygen(residue: ResidueRecord) -> bool:
    return any(_is_oxygen_atom(atom) for atom in residue.atoms)


def count_water_hydrogens(residue: ResidueRecord) -> int:
    return sum(1 for atom in residue.atoms if is_hydrogen_like(atom))


def build_water_hydrogen_coordinates(
    residue: ResidueRecord,
    cluster_residues: list[ResidueRecord],
    *,
    oh_distance: float = 0.9572,
    hoh_angle_degrees: float = 104.52,
) -> list[tuple[float, float, float]]:
    oxygen_atoms = [atom for atom in residue.atoms if _is_oxygen_atom(atom)]
    if not oxygen_atoms:
        raise HistidineGeometryError(
            f"Water residue {residue.id.display()} is in the xTB histidine cluster but has no oxygen atom."
        )
    if len(oxygen_atoms) > 1:
        raise HistidineGeometryError(
            f"Water residue {residue.id.display()} has multiple oxygen atoms; cannot add temporary xTB hydrogens."
        )
    existing_hydrogens = [atom for atom in residue.atoms if is_hydrogen_like(atom)]
    if len(existing_hydrogens) >= 2:
        return []
    nearby_heavy = _nearby_nonwater_heavy_atoms(cluster_residues, water=residue)
    if not existing_hydrogens:
        coordinates, _ = _build_two_water_hydrogens(
            oxygen_atoms[0],
            nearby_heavy=nearby_heavy,
            oh_distance=oh_distance,
            hoh_angle_degrees=hoh_angle_degrees,
        )
        return coordinates
    coordinates, _ = _build_second_water_hydrogen(
        oxygen_atoms[0],
        existing_hydrogens[0],
        nearby_heavy=nearby_heavy,
        oh_distance=oh_distance,
        hoh_angle_degrees=hoh_angle_degrees,
    )
    return coordinates


def heavy_atom_distance(a: AtomRecord, b: AtomRecord) -> float:
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))


BACKBONE_ATOMS = {"N", "H", "H1", "H2", "H3", "C", "O", "OXT"}
REMOVED_BACKBONE_HEAVY_ATOMS = ("N", "C", "O", "OXT")
TAUTOMER_HYDROGENS = {"HD1", "HE2", "1HD1", "2HD1", "1HE2", "2HE2"}


def _append_truncated_protein_residue(
    atoms: list[XyzAtom],
    residue: ResidueRecord,
    *,
    target_histidine: bool,
    tautomer: str | None,
    residue_state: str,
) -> dict[str, list[int]]:
    _validate_residue_hydrogen_state(
        residue,
        residue_state=residue_state,
        target_histidine=target_histidine,
        tautomer=tautomer,
    )
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
        if _is_removed_backbone_atom_or_hydrogen(atom, residue):
            continue
        if _skip_hydrogen_for_cluster_state(
            atom,
            residue,
            residue_state=residue_state,
            target_histidine=target_histidine,
        ):
            continue
        atoms.append(_xyz_from_atom(atom))
        if not is_hydrogen_like(atom):
            fixed.append(len(atoms))
        if name == "CA":
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


def _is_removed_backbone_atom_or_hydrogen(atom: AtomRecord, residue: ResidueRecord) -> bool:
    name = atom.name.strip()
    if name in BACKBONE_ATOMS:
        return True
    if not is_hydrogen_like(atom):
        return False
    return _hydrogen_is_near_any(atom, residue, REMOVED_BACKBONE_HEAVY_ATOMS)


def _append_full_residue(atoms: list[XyzAtom], residue: ResidueRecord) -> list[int]:
    fixed: list[int] = []
    for atom in residue.atoms:
        atoms.append(_xyz_from_atom(atom))
        if not is_hydrogen_like(atom):
            fixed.append(len(atoms))
    return fixed


def _append_water_residue(
    atoms: list[XyzAtom],
    residue: ResidueRecord,
    *,
    cluster_residues: list[ResidueRecord],
    fixed_atom_indices: list[int],
    add_missing_hydrogens: bool,
    oh_distance: float,
    hoh_angle_degrees: float,
    warnings: list[str],
) -> TemporaryWaterHydrogenRecord | None:
    oxygen_atoms = [atom for atom in residue.atoms if _is_oxygen_atom(atom)]
    if not oxygen_atoms:
        raise HistidineGeometryError(
            f"Water residue {residue.id.display()} is in the xTB histidine cluster but has no oxygen atom."
        )
    if len(oxygen_atoms) > 1:
        raise HistidineGeometryError(
            f"Water residue {residue.id.display()} has multiple oxygen atoms; cannot add temporary xTB hydrogens."
        )
    existing_hydrogens = [atom for atom in residue.atoms if is_hydrogen_like(atom)]
    fixed_atom_indices.extend(_append_full_residue(atoms, residue))
    if len(existing_hydrogens) >= 2:
        return None
    if not add_missing_hydrogens:
        _require_water_hydrogens(residue, [_xyz_from_atom(atom) for atom in residue.atoms])
        return None

    oxygen = oxygen_atoms[0]
    nearby_heavy = _nearby_nonwater_heavy_atoms(cluster_residues, water=residue)
    if not existing_hydrogens:
        coordinates, orientation = _build_two_water_hydrogens(
            oxygen,
            nearby_heavy=nearby_heavy,
            oh_distance=oh_distance,
            hoh_angle_degrees=hoh_angle_degrees,
        )
    elif len(existing_hydrogens) == 1:
        coordinates, orientation = _build_second_water_hydrogen(
            oxygen,
            existing_hydrogens[0],
            nearby_heavy=nearby_heavy,
            oh_distance=oh_distance,
            hoh_angle_degrees=hoh_angle_degrees,
        )
    else:
        coordinates = []
        orientation = "input"
    existing_names = {atom.name.strip().upper() for atom in residue.atoms}
    for coordinate in coordinates:
        atoms.append(
            XyzAtom(
                element="H",
                x=coordinate[0],
                y=coordinate[1],
                z=coordinate[2],
                name=_next_water_hydrogen_name(existing_names),
                source="temporary_water_hydrogen",
            )
        )
    if orientation == "deterministic_fallback":
        warnings.append(
            f"Water residue {residue.id.display()} received temporary xTB-only hydrogens "
            "with an arbitrary deterministic orientation because no non-water heavy atoms were available."
        )
    return TemporaryWaterHydrogenRecord(
        chain=residue.id.chain_id,
        resname=residue.id.resname,
        resid=residue.id.resid,
        icode=residue.id.icode,
        hydrogens_added=len(coordinates),
        final_hydrogen_count=len(existing_hydrogens) + len(coordinates),
        orientation=orientation,
    )


def _validate_residue_hydrogen_state(
    residue: ResidueRecord,
    *,
    residue_state: str,
    target_histidine: bool,
    tautomer: str | None,
) -> None:
    if residue.id.resname in {"HIS", "HID", "HIE", "HIP"}:
        _validate_histidine_fragment_hydrogens(
            residue,
            residue_state=residue_state,
            target_histidine=target_histidine,
            tautomer=tautomer,
        )
    elif residue_state in {"ASH", "GLH"}:
        anchors = ("OD1", "OD2") if residue_state == "ASH" else ("OE1", "OE2")
        if _hydrogen_count_near_any(residue, anchors) < 1:
            raise HistidineGeometryError(
                f"xTB cluster residue {residue.id.display()} is assigned {residue_state}, "
                f"but no carboxyl proton is present on {'/'.join(anchors)}. "
                "Hydrogenate the input consistently or use a manual override that matches the input geometry."
            )
    elif residue_state == "CYS":
        if _hydrogen_count_near_any(residue, ("SG",)) < 1:
            raise HistidineGeometryError(
                f"xTB cluster residue {residue.id.display()} is assigned CYS, "
                "but no thiol proton is present on SG."
            )
    elif residue_state == "LYS":
        count = _hydrogen_count_near_any(residue, ("NZ",))
        if count < 3:
            raise HistidineGeometryError(
                f"xTB cluster residue {residue.id.display()} is assigned LYS, "
                f"but NZ has only {count} nearby hydrogens; expected 3."
            )
    elif residue_state == "LYN":
        count = _hydrogen_count_near_any(residue, ("NZ",))
        if count != 2:
            raise HistidineGeometryError(
                f"xTB cluster residue {residue.id.display()} is assigned LYN, "
                f"but NZ has {count} nearby hydrogens; expected 2."
            )
    elif residue_state == "ARG":
        required = {"NE": 1, "NH1": 2, "NH2": 2}
        missing = [
            f"{atom_name} expected {expected}, found {_hydrogen_count_near_any(residue, (atom_name,))}"
            for atom_name, expected in required.items()
            if _hydrogen_count_near_any(residue, (atom_name,)) < expected
        ]
        if missing:
            raise HistidineGeometryError(
                f"xTB cluster residue {residue.id.display()} is assigned ARG, "
                f"but guanidinium hydrogens are incomplete: {', '.join(missing)}."
            )


def _validate_histidine_fragment_hydrogens(
    residue: ResidueRecord,
    *,
    residue_state: str,
    target_histidine: bool,
    tautomer: str | None,
) -> None:
    missing_atoms = [name for name in ("CG", "ND1", "CE1", "NE2", "CD2") if _atom_by_name(residue, name) is None]
    if missing_atoms:
        raise HistidineGeometryError(
            f"Histidine {residue.id.display()} is missing required atoms: {', '.join(missing_atoms)}."
        )
    carbon_missing = [
        name
        for name in ("CE1", "CD2")
        if _hydrogen_count_near_any(residue, (name,)) < 1
    ]
    if carbon_missing:
        raise HistidineGeometryError(
            f"Histidine {residue.id.display()} lacks ring carbon hydrogens on {', '.join(carbon_missing)}; "
            "the xTB tautomer cluster would have an inconsistent valence."
        )
    if target_histidine:
        if tautomer not in {"HID", "HIE"}:
            raise HistidineGeometryError("Target histidine tautomer must be HID or HIE.")
        return

    nd1_h = _hydrogen_count_near_any(residue, ("ND1",))
    ne2_h = _hydrogen_count_near_any(residue, ("NE2",))
    expected = {
        "HID": (1, 0),
        "HIE": (0, 1),
        "HIP": (1, 1),
    }.get(residue_state)
    if expected is None:
        total = nd1_h + ne2_h
        if total != 1:
            raise HistidineGeometryError(
                f"Neighbor histidine {residue.id.display()} is unresolved in the xTB cluster; "
                f"found {total} imidazole N-H hydrogens. Assign it manually to HID/HIE/HIP."
            )
        return
    if nd1_h != expected[0] or ne2_h != expected[1]:
        raise HistidineGeometryError(
            f"Histidine {residue.id.display()} is assigned {residue_state}, "
            f"but has ND1 hydrogens={nd1_h} and NE2 hydrogens={ne2_h}; "
            f"expected {expected[0]} and {expected[1]}."
        )


def _skip_hydrogen_for_cluster_state(
    atom: AtomRecord,
    residue: ResidueRecord,
    *,
    residue_state: str,
    target_histidine: bool,
) -> bool:
    if not is_hydrogen_like(atom):
        return False
    if target_histidine and _hydrogen_is_near_any(atom, residue, ("ND1", "NE2")):
        return True
    if residue_state in {"ASP", "GLU"}:
        anchors = ("OD1", "OD2") if residue_state == "ASP" else ("OE1", "OE2")
        return _hydrogen_is_near_any(atom, residue, anchors)
    if residue_state in {"CYM", "CYX"}:
        return _hydrogen_is_near_any(atom, residue, ("SG",))
    return False


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
            "Enable protonation.histidine.xtb.add_missing_water_hydrogens, remove nearby waters, "
            "or add a manual HIS override."
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


def _atom_by_name(residue: ResidueRecord, name: str) -> AtomRecord | None:
    return next((atom for atom in residue.atoms if atom.name.strip() == name), None)


def _hydrogen_count_near_any(
    residue: ResidueRecord,
    atom_names: tuple[str, ...],
    *,
    cutoff_angstrom: float = 1.25,
) -> int:
    return sum(
        1
        for atom in residue.atoms
        if is_hydrogen_like(atom) and _hydrogen_is_near_any(atom, residue, atom_names, cutoff_angstrom=cutoff_angstrom)
    )


def _hydrogen_is_near_any(
    hydrogen: AtomRecord,
    residue: ResidueRecord,
    atom_names: tuple[str, ...],
    *,
    cutoff_angstrom: float = 1.25,
) -> bool:
    for name in atom_names:
        anchor = _atom_by_name(residue, name)
        if anchor is None:
            continue
        if dist((hydrogen.x, hydrogen.y, hydrogen.z), (anchor.x, anchor.y, anchor.z)) <= cutoff_angstrom:
            return True
    return False


def _is_oxygen_atom(atom: AtomRecord) -> bool:
    element = _element(atom).upper()
    return element == "O"


def _nearby_nonwater_heavy_atoms(
    cluster_residues: list[ResidueRecord],
    *,
    water: ResidueRecord,
) -> list[AtomRecord]:
    nearby: list[AtomRecord] = []
    for residue in cluster_residues:
        if residue is water:
            continue
        for atom in residue.atoms:
            if not is_hydrogen_like(atom):
                nearby.append(atom)
    return nearby


def _build_two_water_hydrogens(
    oxygen: AtomRecord,
    *,
    nearby_heavy: list[AtomRecord],
    oh_distance: float,
    hoh_angle_degrees: float,
) -> tuple[list[tuple[float, float, float]], str]:
    half_angle = radians(hoh_angle_degrees / 2.0)
    candidates: list[tuple[list[tuple[float, float, float]], str]] = []
    for bisector in _candidate_bisectors(oxygen, nearby_heavy):
        basis_a, basis_b = _perpendicular_basis(bisector)
        for phi_degrees in (0.0, 45.0, 90.0, 135.0):
            phi = radians(phi_degrees)
            perpendicular = _add(
                _scale(basis_a, cos(phi)),
                _scale(basis_b, sin(phi)),
            )
            direction1 = _normalize_vec(
                _add(_scale(bisector, cos(half_angle)), _scale(perpendicular, sin(half_angle)))
            )
            direction2 = _normalize_vec(
                _add(_scale(bisector, cos(half_angle)), _scale(perpendicular, -sin(half_angle)))
            )
            candidates.append(
                (
                    [
                        _point_from_atom(oxygen, direction1, oh_distance),
                        _point_from_atom(oxygen, direction2, oh_distance),
                    ],
                    "clash_aware" if nearby_heavy else "deterministic_fallback",
                )
            )
    return _best_water_candidate(candidates, nearby_heavy)


def _build_second_water_hydrogen(
    oxygen: AtomRecord,
    existing_hydrogen: AtomRecord,
    *,
    nearby_heavy: list[AtomRecord],
    oh_distance: float,
    hoh_angle_degrees: float,
) -> tuple[list[tuple[float, float, float]], str]:
    existing_direction = (
        existing_hydrogen.x - oxygen.x,
        existing_hydrogen.y - oxygen.y,
        existing_hydrogen.z - oxygen.z,
    )
    try:
        existing_unit = _normalize_vec(existing_direction)
    except HistidineGeometryError as exc:
        raise HistidineGeometryError(
            f"Water residue {oxygen.chain_id}:{oxygen.resname}{oxygen.resid}{oxygen.icode or ''} "
            "has a degenerate existing O-H vector."
        ) from exc
    basis_a, basis_b = _perpendicular_basis(existing_unit)
    angle = radians(hoh_angle_degrees)
    candidates: list[tuple[list[tuple[float, float, float]], str]] = []
    for phi_degrees in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        phi = radians(phi_degrees)
        perpendicular = _add(_scale(basis_a, cos(phi)), _scale(basis_b, sin(phi)))
        direction = _normalize_vec(
            _add(_scale(existing_unit, cos(angle)), _scale(perpendicular, sin(angle)))
        )
        candidates.append(
            (
                [_point_from_atom(oxygen, direction, oh_distance)],
                "clash_aware" if nearby_heavy else "deterministic_fallback",
            )
        )
    return _best_water_candidate(candidates, nearby_heavy)


def _candidate_bisectors(
    oxygen: AtomRecord,
    nearby_heavy: list[AtomRecord],
) -> list[tuple[float, float, float]]:
    candidates: list[tuple[float, float, float]] = []
    if nearby_heavy:
        nearest = min(
            nearby_heavy,
            key=lambda atom: dist((oxygen.x, oxygen.y, oxygen.z), (atom.x, atom.y, atom.z)),
        )
        candidates.append(
            _normalize_vec((oxygen.x - nearest.x, oxygen.y - nearest.y, oxygen.z - nearest.z))
        )
        summed = (0.0, 0.0, 0.0)
        for atom in nearby_heavy:
            direction = _normalize_vec((oxygen.x - atom.x, oxygen.y - atom.y, oxygen.z - atom.z))
            summed = _add(summed, direction)
        try:
            candidates.append(_normalize_vec(summed))
        except HistidineGeometryError:
            pass
    candidates.extend(
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        ]
    )
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        key = tuple(round(value * 1000) for value in candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _best_water_candidate(
    candidates: list[tuple[list[tuple[float, float, float]], str]],
    nearby_heavy: list[AtomRecord],
) -> tuple[list[tuple[float, float, float]], str]:
    if not candidates:
        raise HistidineGeometryError("No candidate water hydrogen coordinates were generated.")
    best_coordinates, best_orientation = candidates[0]
    best_score = _score_water_candidate(best_coordinates, nearby_heavy)
    for coordinates, orientation in candidates[1:]:
        score = _score_water_candidate(coordinates, nearby_heavy)
        if score > best_score:
            best_coordinates = coordinates
            best_orientation = orientation
            best_score = score
    return best_coordinates, best_orientation


def _score_water_candidate(
    coordinates: list[tuple[float, float, float]],
    nearby_heavy: list[AtomRecord],
) -> float:
    if not nearby_heavy:
        return float("inf")
    return min(
        dist(coordinate, (atom.x, atom.y, atom.z))
        for coordinate in coordinates
        for atom in nearby_heavy
    )


def _next_water_hydrogen_name(existing_names: set[str]) -> str:
    for name in ("H1", "H2", "H3", "HW1", "HW2"):
        if name not in existing_names:
            existing_names.add(name)
            return name
    index = 1
    while True:
        name = f"HT{index}"
        if name not in existing_names:
            existing_names.add(name)
            return name
        index += 1


def _point_from_atom(
    atom: AtomRecord,
    direction: tuple[float, float, float],
    distance_angstrom: float,
) -> tuple[float, float, float]:
    return (
        atom.x + direction[0] * distance_angstrom,
        atom.y + direction[1] * distance_angstrom,
        atom.z + direction[2] * distance_angstrom,
    )


def _perpendicular_basis(
    vector: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    reference = (1.0, 0.0, 0.0)
    if abs(_dot(vector, reference)) > 0.8:
        reference = (0.0, 1.0, 0.0)
    basis_a = _normalize_vec(_cross(vector, reference))
    basis_b = _normalize_vec(_cross(vector, basis_a))
    return basis_a, basis_b


def _normalize_vec(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in vector))
    if norm < 1.0e-8:
        raise HistidineGeometryError("Degenerate geometry; cannot place temporary hydrogen.")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)


def _add(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )
