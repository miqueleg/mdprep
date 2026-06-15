"""Focused Tripos mol2 parser, writer, and ligand validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path

from mdprep.config.models import LigandConfig
from mdprep.structure.models import AtomRecord


class Mol2Error(ValueError):
    """Raised when a mol2 file cannot be parsed or validated safely."""


@dataclass(frozen=True)
class Mol2Atom:
    atom_id: int
    name: str
    x: float
    y: float
    z: float
    atom_type: str
    subst_id: int
    subst_name: str
    charge: float


@dataclass
class Mol2File:
    path: Path
    lines: list[str]
    molecule_name: str | None
    atoms: list[Mol2Atom]
    atom_line_indices: list[int]

    @property
    def total_charge(self) -> float:
        return sum(atom.charge for atom in self.atoms)


@dataclass(frozen=True)
class Mol2ValidationResult:
    final_mol2_path: Path
    charges_csv_path: Path
    validation_json_path: Path
    charge_sum_before_correction: float
    charge_correction_applied: float
    charge_sum_final: float
    coordinate_max_deviation: float
    atom_names_preserved: bool
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "final_mol2_path": str(self.final_mol2_path),
            "charges_csv_path": str(self.charges_csv_path),
            "validation_json_path": str(self.validation_json_path),
            "charge_sum_before_correction": self.charge_sum_before_correction,
            "charge_correction_applied": self.charge_correction_applied,
            "charge_sum_final": self.charge_sum_final,
            "coordinate_max_deviation": self.coordinate_max_deviation,
            "atom_names_preserved": self.atom_names_preserved,
            "warnings": self.warnings,
        }


def read_mol2(path: str | Path) -> Mol2File:
    mol2_path = Path(path)
    lines = mol2_path.read_text(encoding="utf-8").splitlines()
    atom_start = _marker_index(lines, "@<TRIPOS>ATOM")
    if atom_start is None:
        raise Mol2Error(f"mol2 file is missing @<TRIPOS>ATOM section: {mol2_path}")
    atom_end = next(
        (index for index in range(atom_start + 1, len(lines)) if lines[index].startswith("@<TRIPOS>")),
        len(lines),
    )
    atoms: list[Mol2Atom] = []
    atom_line_indices: list[int] = []
    for index in range(atom_start + 1, atom_end):
        line = lines[index]
        if not line.strip():
            continue
        atoms.append(_parse_atom_line(line, mol2_path))
        atom_line_indices.append(index)
    if not atoms:
        raise Mol2Error(f"mol2 file has no atom records: {mol2_path}")
    molecule_name = _molecule_name(lines)
    return Mol2File(
        path=mol2_path,
        lines=lines,
        molecule_name=molecule_name,
        atoms=atoms,
        atom_line_indices=atom_line_indices,
    )


def write_mol2(mol2: Mol2File, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = list(mol2.lines)
    for atom, line_index in zip(mol2.atoms, mol2.atom_line_indices, strict=True):
        lines[line_index] = _format_atom(atom)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mol2_with_charges(
    mol2_path: str | Path,
    charges: list[float],
    output_path: str | Path,
) -> None:
    mol2 = read_mol2(mol2_path)
    if len(charges) != len(mol2.atoms):
        raise Mol2Error(
            f"Charge count {len(charges)} does not match mol2 atom count {len(mol2.atoms)}."
        )
    mol2.atoms = [
        replace(atom, charge=float(charge))
        for atom, charge in zip(mol2.atoms, charges, strict=True)
    ]
    write_mol2(mol2, output_path)


def validate_and_write_final_mol2(
    *,
    mol2_path: str | Path,
    extracted_atoms: list[AtomRecord],
    ligand: LigandConfig,
    final_mol2_path: str | Path,
    charges_csv_path: str | Path,
    validation_json_path: str | Path,
) -> Mol2ValidationResult:
    mol2 = read_mol2(mol2_path)
    warnings: list[str] = []
    if len(mol2.atoms) != len(extracted_atoms):
        raise Mol2Error(
            f"Ligand {ligand.id} atom-count mismatch: extracted PDB has "
            f"{len(extracted_atoms)} atoms but mol2 has {len(mol2.atoms)} atoms."
        )
    _validate_element_order(mol2.atoms, extracted_atoms, ligand.id)
    atoms = list(mol2.atoms)
    if ligand.preserve_atom_names:
        atoms = [
            replace(mol2_atom, name=pdb_atom.name)
            for mol2_atom, pdb_atom in zip(atoms, extracted_atoms, strict=True)
        ]
    atom_names_preserved = [atom.name for atom in atoms] == [atom.name for atom in extracted_atoms]
    if ligand.preserve_atom_names and not ligand.allow_atom_renaming and not atom_names_preserved:
        raise Mol2Error(f"Ligand {ligand.id} atom names could not be preserved in final mol2.")

    coordinate_max_deviation = _max_coordinate_deviation(atoms, extracted_atoms)
    if (
        ligand.preserve_coordinates
        and not ligand.allow_coordinate_changes
        and coordinate_max_deviation > 0.05
    ):
        raise Mol2Error(
            f"Ligand {ligand.id} mol2 coordinates deviate from extracted PDB by "
            f"{coordinate_max_deviation:.3f} A; maximum allowed is 0.050 A."
        )

    charge_before = sum(atom.charge for atom in atoms)
    residual = ligand.net_charge - charge_before
    correction = 0.0
    if abs(residual) > 0.01 + 1.0e-9:
        raise Mol2Error(
            f"Ligand {ligand.id} mol2 charge sum {charge_before:.6f} differs from "
            f"target charge {ligand.net_charge} by {residual:.6f} e."
        )
    if abs(residual) > 1.0e-9:
        target_index = max(range(len(atoms)), key=lambda index: abs(atoms[index].charge))
        atoms[target_index] = replace(atoms[target_index], charge=atoms[target_index].charge + residual)
        correction = residual
        warnings.append(
            f"Applied charge residual {residual:.6f} e to atom {atoms[target_index].name}."
        )

    mol2.atoms = atoms
    final_path = Path(final_mol2_path)
    write_mol2(mol2, final_path)
    charges_path = Path(charges_csv_path)
    _write_charges_csv(atoms, charges_path)
    result = Mol2ValidationResult(
        final_mol2_path=final_path,
        charges_csv_path=charges_path,
        validation_json_path=Path(validation_json_path),
        charge_sum_before_correction=charge_before,
        charge_correction_applied=correction,
        charge_sum_final=sum(atom.charge for atom in atoms),
        coordinate_max_deviation=coordinate_max_deviation,
        atom_names_preserved=atom_names_preserved,
        warnings=warnings,
    )
    result.validation_json_path.parent.mkdir(parents=True, exist_ok=True)
    result.validation_json_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_atom_line(line: str, path: Path) -> Mol2Atom:
    fields = line.split()
    if len(fields) < 9:
        raise Mol2Error(f"Invalid mol2 atom record in {path}: {line!r}")
    try:
        return Mol2Atom(
            atom_id=int(fields[0]),
            name=fields[1],
            x=float(fields[2]),
            y=float(fields[3]),
            z=float(fields[4]),
            atom_type=fields[5],
            subst_id=int(fields[6]),
            subst_name=fields[7],
            charge=float(fields[8]),
        )
    except ValueError as exc:
        raise Mol2Error(f"Invalid mol2 atom record in {path}: {line!r}") from exc


def _format_atom(atom: Mol2Atom) -> str:
    return (
        f"{atom.atom_id:7d} {atom.name:<8} "
        f"{atom.x:10.4f} {atom.y:10.4f} {atom.z:10.4f} "
        f"{atom.atom_type:<8} {atom.subst_id:4d} {atom.subst_name:<8} {atom.charge:10.6f}"
    )


def _marker_index(lines: list[str], marker: str) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return index
    return None


def _molecule_name(lines: list[str]) -> str | None:
    index = _marker_index(lines, "@<TRIPOS>MOLECULE")
    if index is None:
        return None
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _validate_element_order(mol2_atoms: list[Mol2Atom], pdb_atoms: list[AtomRecord], ligand_id: str) -> None:
    mol2_elements = [_element_from_mol2(atom) for atom in mol2_atoms]
    pdb_elements = [(atom.element or _element_from_atom_name(atom.name) or "").upper() for atom in pdb_atoms]
    if mol2_elements != pdb_elements:
        raise Mol2Error(
            f"Ligand {ligand_id} element-order mismatch between extracted PDB and mol2: "
            f"{pdb_elements} != {mol2_elements}."
        )


def _element_from_mol2(atom: Mol2Atom) -> str:
    token = atom.atom_type.split(".", 1)[0].strip()
    if not token:
        token = atom.name
    return _element_from_atom_name(token).upper()


def _element_from_atom_name(name: str) -> str:
    stripped = name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    upper = stripped.upper()
    if upper[:2] in {"CL", "BR", "NA", "MG", "FE", "ZN", "CU", "MN", "CO", "NI", "CA"}:
        return upper[:2]
    if not upper:
        return ""
    return upper[0]


def _max_coordinate_deviation(mol2_atoms: list[Mol2Atom], pdb_atoms: list[AtomRecord]) -> float:
    if not mol2_atoms:
        return 0.0
    return max(
        dist((mol2.x, mol2.y, mol2.z), (pdb.x, pdb.y, pdb.z))
        for mol2, pdb in zip(mol2_atoms, pdb_atoms, strict=True)
    )


def _write_charges_csv(atoms: list[Mol2Atom], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["atom_id", "atom_name", "charge"])
        writer.writeheader()
        for atom in atoms:
            writer.writerow({"atom_id": atom.atom_id, "atom_name": atom.name, "charge": atom.charge})
