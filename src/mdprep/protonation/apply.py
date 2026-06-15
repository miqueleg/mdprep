"""Apply safe manual protonation-stage residue-name changes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.protonation.disulfide_states import (
    DisulfideAssignmentError,
    DisulfideResidueAssignment,
    resolve_disulfide_assignments,
)
from mdprep.protonation.overrides import ManualOverrideError, resolve_manual_overrides
from mdprep.structure.classify import is_histidine, is_titratable_residue
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord


AUTOMATED_NOT_IMPLEMENTED_MESSAGE = (
    "Automated protonation is not implemented yet in mdprep; use method: manual_only for Task 4 workflows."
)


class ProtonationApplicationError(ValueError):
    """Raised when the protonation stage cannot be applied safely."""


@dataclass(frozen=True)
class ProtonationRecord:
    chain: str
    resid: int
    icode: str | None
    original_resname: str
    final_resname: str
    source: str
    reason: str
    selector: dict[str, object] | None = None

    @property
    def changed(self) -> bool:
        return self.original_resname != self.final_resname

    def to_dict(self) -> dict[str, object]:
        return {
            "chain": self.chain,
            "resid": self.resid,
            "icode": self.icode,
            "original_resname": self.original_resname,
            "final_resname": self.final_resname,
            "source": self.source,
            "reason": self.reason,
            "changed": self.changed,
            "selector": self.selector,
        }


@dataclass
class ProtonationResult:
    input_normalized_pdb_path: Path
    output_protonation_pdb_path: Path
    method: str
    ph: float
    structure: PdbStructure
    manual_overrides_applied: list[ProtonationRecord] = field(default_factory=list)
    disulfide_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    hydrogen_atoms_removed: int = 0
    unresolved_histidines: list[dict[str, object]] = field(default_factory=list)
    titratable_residues_not_explicitly_assigned: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def records(self) -> list[ProtonationRecord]:
        return self.manual_overrides_applied + self.disulfide_assignments_applied

    def to_report_dict(self) -> dict[str, object]:
        changed = [record.to_dict() for record in self.records if record.changed]
        unchanged = [record.to_dict() for record in self.records if not record.changed]
        return {
            "input_normalized_pdb_path": str(self.input_normalized_pdb_path),
            "output_protonation_pdb_path": str(self.output_protonation_pdb_path),
            "method": self.method,
            "ph": self.ph,
            "manual_overrides_applied": [record.to_dict() for record in self.manual_overrides_applied],
            "disulfide_assignments_applied": [
                record.to_dict() for record in self.disulfide_assignments_applied
            ],
            "hydrogen_atoms_removed": self.hydrogen_atoms_removed,
            "residues_changed": changed,
            "residues_unchanged_but_explicitly_assigned": unchanged,
            "unresolved_histidines_remaining_as_his": self.unresolved_histidines,
            "titratable_residues_not_explicitly_assigned": self.titratable_residues_not_explicitly_assigned,
            "warnings": self.warnings,
        }


def apply_protonation_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    input_normalized_pdb_path: str | Path,
    output_protonation_pdb_path: str | Path,
) -> ProtonationResult:
    if manifest.protonation.method != "manual_only":
        raise ProtonationApplicationError(AUTOMATED_NOT_IMPLEMENTED_MESSAGE)

    try:
        manual_assignments = resolve_manual_overrides(structure, manifest)
        disulfide_assignments = resolve_disulfide_assignments(structure, manifest)
    except (ManualOverrideError, DisulfideAssignmentError) as exc:
        raise ProtonationApplicationError(str(exc)) from exc

    final_by_residue: dict[int, str] = {}
    records: list[ProtonationRecord] = []
    manual_state_by_residue = {id(item.residue): item.requested_state for item in manual_assignments}

    for assignment in manual_assignments:
        final_by_residue[id(assignment.residue)] = assignment.requested_state
        records.append(
            _record(
                assignment.residue,
                final_resname=assignment.requested_state,
                source="manual_override",
                reason=assignment.reason,
                selector=assignment.selector,
            )
        )

    disulfide_records: list[ProtonationRecord] = []
    for assignment in disulfide_assignments:
        _validate_disulfide_manual_compatibility(assignment, manual_state_by_residue)
        final_by_residue[id(assignment.residue)] = "CYX"
        disulfide_records.append(
            _record(
                assignment.residue,
                final_resname="CYX",
                source=assignment.source,
                reason=assignment.reason,
                selector=None,
            )
        )

    renamed_atoms = _rename_atoms(structure, final_by_residue)
    hydrogen_atoms_removed = 0
    if manifest.structure.remove_input_hydrogens:
        before = len(renamed_atoms)
        renamed_atoms = [atom for atom in renamed_atoms if not is_hydrogen_atom(atom)]
        hydrogen_atoms_removed = before - len(renamed_atoms)

    protonated_structure = PdbStructure(
        path=Path(output_protonation_pdb_path),
        atoms=renamed_atoms,
        residues=_build_residues(renamed_atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
    )
    explicit_keys = {
        (record.chain, record.resid, record.icode)
        for record in (records + disulfide_records)
    }
    unresolved_his = [
        _residue_dict(residue)
        for residue in protonated_structure.residues
        if is_histidine(residue) and residue.id.resname == "HIS"
    ]
    unassigned_titratable = [
        _residue_dict(residue)
        for residue in protonated_structure.residues
        if is_titratable_residue(residue)
        and (residue.id.chain_id, residue.id.resid, residue.id.icode) not in explicit_keys
    ]
    return ProtonationResult(
        input_normalized_pdb_path=Path(input_normalized_pdb_path),
        output_protonation_pdb_path=Path(output_protonation_pdb_path),
        method=manifest.protonation.method,
        ph=manifest.protonation.ph,
        structure=protonated_structure,
        manual_overrides_applied=records,
        disulfide_assignments_applied=disulfide_records,
        hydrogen_atoms_removed=hydrogen_atoms_removed,
        unresolved_histidines=unresolved_his,
        titratable_residues_not_explicitly_assigned=unassigned_titratable,
        warnings=list(protonated_structure.warnings),
    )


def is_hydrogen_atom(atom: AtomRecord) -> bool:
    element_field = atom.original_line[76:78].strip() if len(atom.original_line) >= 78 else ""
    if element_field:
        return element_field.upper() == "H"
    stripped = atom.name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    return stripped.upper().startswith("H")


def _validate_disulfide_manual_compatibility(
    assignment: DisulfideResidueAssignment,
    manual_state_by_residue: dict[int, str],
) -> None:
    manual_state = manual_state_by_residue.get(id(assignment.residue))
    if manual_state is not None and manual_state != "CYX":
        residue = assignment.residue.id.display()
        raise ProtonationApplicationError(
            f"Manual protonation override for {residue} requests {manual_state}, "
            f"but {assignment.source} requires CYX."
        )


def _rename_atoms(structure: PdbStructure, final_by_residue: dict[int, str]) -> list[AtomRecord]:
    final_by_key: dict[tuple[str, str, int, str | None], str] = {}
    for residue in structure.residues:
        final = final_by_residue.get(id(residue))
        if final is not None:
            final_by_key[(residue.id.chain_id, residue.id.resname, residue.id.resid, residue.id.icode)] = final

    renamed: list[AtomRecord] = []
    for atom in structure.atoms:
        final_resname = final_by_key.get(atom.residue_key, atom.resname)
        renamed.append(replace(atom, resname=final_resname))
    return renamed


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


def _record(
    residue: ResidueRecord,
    *,
    final_resname: str,
    source: str,
    reason: str,
    selector: dict[str, object] | None,
) -> ProtonationRecord:
    return ProtonationRecord(
        chain=residue.id.chain_id,
        resid=residue.id.resid,
        icode=residue.id.icode,
        original_resname=residue.id.resname,
        final_resname=final_resname,
        source=source,
        reason=reason,
        selector=selector,
    )


def _residue_dict(residue: ResidueRecord) -> dict[str, object]:
    return {
        **residue.id.to_dict(),
        "atom_count": len(residue.atoms),
        "record_names": sorted(residue.record_names),
        "original_index": residue.original_index,
    }

