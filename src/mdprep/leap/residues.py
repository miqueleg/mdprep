"""Residue and PDB helpers for tleap input generation."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mdprep.ambertools.mol2 import Mol2Error, read_mol2
from mdprep.config.models import ManifestConfig
from mdprep.protonation.apply import ProtonationRecord, ProtonationResult
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector
from mdprep.structure.writer import write_pdb

if TYPE_CHECKING:
    from mdprep.ligands.workflow import LigandStageResult


class LeapResidueError(ValueError):
    """Raised when residues cannot be mapped safely for tleap."""


@dataclass(frozen=True)
class LeapInputResult:
    path: Path
    structure: PdbStructure
    water_renames: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "water_renames": self.water_renames,
        }


@dataclass(frozen=True)
class LigandParameterFiles:
    ligand_id: str
    variable_name: str
    residue_name: str
    final_mol2_path: Path
    final_frcmod_path: Path
    atom_names: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "ligand_id": self.ligand_id,
            "variable_name": self.variable_name,
            "residue_name": self.residue_name,
            "final_mol2_path": str(self.final_mol2_path),
            "final_frcmod_path": str(self.final_frcmod_path),
            "atom_names": self.atom_names,
        }


@dataclass(frozen=True)
class DisulfideBondCommand:
    residue_a: dict[str, object]
    residue_b: dict[str, object]
    index_a: int
    index_b: int
    command: str

    def to_dict(self) -> dict[str, object]:
        return {
            "residue_a": self.residue_a,
            "residue_b": self.residue_b,
            "index_a": self.index_a,
            "index_b": self.index_b,
            "command": self.command,
        }


def prepare_leap_input_pdb(
    structure: PdbStructure,
    output_path: str | Path,
) -> LeapInputResult:
    atoms: list[AtomRecord] = []
    water_renames: list[dict[str, object]] = []
    for atom in structure.atoms:
        new_atom = atom
        if atom.resname in {"HOH", "H2O", "TIP3", "OPC"}:
            new_name = atom.name
            if (atom.element or "").upper() == "O" and atom.name.strip() != "O":
                new_name = "O"
            new_atom = replace(atom, resname="WAT", name=new_name)
            if new_atom.resname != atom.resname or new_atom.name != atom.name:
                water_renames.append(
                    {
                        "chain_id": atom.chain_id,
                        "resid": atom.resid,
                        "icode": atom.icode,
                        "old_resname": atom.resname,
                        "new_resname": new_atom.resname,
                        "old_atom_name": atom.name,
                        "new_atom_name": new_atom.name,
                    }
                )
        atoms.append(new_atom)
    output = Path(output_path)
    leap_structure = PdbStructure(
        path=output,
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
    )
    write_pdb(leap_structure, output)
    return LeapInputResult(path=output, structure=leap_structure, water_renames=water_renames)


def residue_index_map(structure: PdbStructure) -> dict[tuple[str, int, str | None], int]:
    mapping: dict[tuple[str, int, str | None], int] = {}
    for index, residue in enumerate(structure.residues, start=1):
        key = _residue_key(residue.id)
        if key in mapping:
            raise LeapResidueError(
                f"Residue identity {residue.id.display()} is ambiguous for tleap residue-index mapping."
            )
        mapping[key] = index
    return mapping


def validate_ligand_parameter_files(
    *,
    manifest: ManifestConfig,
    structure: PdbStructure,
    ligand_result: LigandStageResult,
) -> list[LigandParameterFiles]:
    items_by_id = {item.ligand_id: item for item in ligand_result.ligands}
    variable_names: set[str] = set()
    residue_parameter_files: dict[str, tuple[bytes, bytes]] = {}
    validated: list[LigandParameterFiles] = []
    for ligand in manifest.ligands:
        item = items_by_id.get(ligand.id)
        if item is None:
            raise LeapResidueError(f"Ligand {ligand.id} was not parameterized before tleap.")
        if item.final_mol2_path is None or not item.final_mol2_path.exists():
            raise LeapResidueError(f"Ligand {ligand.id} final mol2 is missing.")
        if item.final_frcmod_path is None or not item.final_frcmod_path.exists():
            raise LeapResidueError(f"Ligand {ligand.id} final frcmod is missing.")
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError as exc:
            raise LeapResidueError(f"Ligand {ligand.id} selector failed during tleap validation: {exc}") from exc
        mol2 = read_mol2(item.final_mol2_path)
        mol2_resnames = {atom.subst_name for atom in mol2.atoms}
        if mol2_resnames != {residue.id.resname}:
            raise LeapResidueError(
                f"Ligand {ligand.id} residue name mismatch: PDB has {residue.id.resname}, "
                f"mol2 has {sorted(mol2_resnames)}."
            )
        pdb_atom_names = [atom.name for atom in residue.atoms]
        mol2_atom_names = [atom.name for atom in mol2.atoms]
        if pdb_atom_names != mol2_atom_names:
            raise LeapResidueError(
                f"Ligand {ligand.id} atom-name mismatch between PDB and mol2: "
                f"{pdb_atom_names} != {mol2_atom_names}."
            )
        try:
            key = (
                item.final_mol2_path.read_bytes(),
                item.final_frcmod_path.read_bytes(),
            )
        except OSError as exc:
            raise LeapResidueError(f"Could not read ligand parameter files for {ligand.id}: {exc}") from exc
        previous = residue_parameter_files.get(residue.id.resname)
        if previous is not None and previous != key:
            raise LeapResidueError(
                f"Multiple different parameter sets for residue name {residue.id.resname} are not supported yet; "
                "use unique residue names or shared parameter files."
            )
        residue_parameter_files[residue.id.resname] = key
        variable = _unique_variable_name(ligand.id, variable_names)
        validated.append(
            LigandParameterFiles(
                ligand_id=ligand.id,
                variable_name=variable,
                residue_name=residue.id.resname,
                final_mol2_path=item.final_mol2_path,
                final_frcmod_path=item.final_frcmod_path,
                atom_names=pdb_atom_names,
            )
        )
    return validated


def disulfide_bond_commands(
    *,
    structure: PdbStructure,
    protonation_result: ProtonationResult,
) -> list[DisulfideBondCommand]:
    assignments = protonation_result.disulfide_assignments_applied
    pair_keys: set[frozenset[tuple[str, int, str | None]]] = set()
    for record in assignments:
        partner = _partner_key(record)
        if partner is None:
            continue
        current = (record.chain, record.resid, record.icode)
        pair = frozenset({current, partner})
        if len(pair) != 2:
            continue
        pair_keys.add(pair)

    cyx_keys = [
        _residue_key(residue.id)
        for residue in structure.residues
        if residue.id.resname == "CYX"
    ]
    for key in cyx_keys:
        if sum(1 for pair in pair_keys if key in pair) != 1:
            raise LeapResidueError(
                "CYX residue has no unique disulfide pair; add the pair under disulfides.force "
                "or change the residue state to CYS/CYM if it is not disulfide-bonded."
            )

    mapping = residue_index_map(structure)
    residue_by_key = {_residue_key(residue.id): residue for residue in structure.residues}
    commands: list[DisulfideBondCommand] = []
    for pair in sorted(pair_keys, key=lambda item: sorted(item)):
        key_a, key_b = sorted(pair)
        residue_a = residue_by_key.get(key_a)
        residue_b = residue_by_key.get(key_b)
        if residue_a is None or residue_b is None:
            raise LeapResidueError("Disulfide residue pair is not present in the leap-input structure.")
        if "SG" not in residue_a.atom_names() or "SG" not in residue_b.atom_names():
            raise LeapResidueError("Disulfide pair is missing SG atoms required for tleap bond commands.")
        index_a = mapping[key_a]
        index_b = mapping[key_b]
        commands.append(
            DisulfideBondCommand(
                residue_a=residue_a.id.to_dict(),
                residue_b=residue_b.id.to_dict(),
                index_a=index_a,
                index_b=index_b,
                command=f"bond system.{index_a}.SG system.{index_b}.SG",
            )
        )
    return commands


def _partner_key(record: ProtonationRecord) -> tuple[str, int, str | None] | None:
    partner = record.metadata.get("partner")
    if not isinstance(partner, dict):
        return None
    return (
        str(partner.get("chain_id", "")),
        int(partner["resid"]),
        partner.get("icode"),  # type: ignore[arg-type]
    )


def _unique_variable_name(ligand_id: str, used: set[str]) -> str:
    base = re.sub(r"\W+", "_", ligand_id.upper()).strip("_")
    if not base:
        base = "LIG"
    if base[0].isdigit():
        base = f"L_{base}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _residue_key(residue_id: ResidueId) -> tuple[str, int, str | None]:
    return (residue_id.chain_id, residue_id.resid, residue_id.icode)


def _build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    residues: list[ResidueRecord] = []
    for index, ((chain_id, resname, resid, icode), residue_atoms) in enumerate(grouped.items()):
        residues.append(
            ResidueRecord(
                id=ResidueId(chain_id=chain_id, resname=resname, resid=resid, icode=icode),
                atoms=residue_atoms,
                record_names={atom.record_name for atom in residue_atoms},
                original_index=index,
            )
        )
    return residues
