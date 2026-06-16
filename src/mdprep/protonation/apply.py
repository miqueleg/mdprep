"""Apply safe protonation-stage residue-name changes."""

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
from mdprep.protonation.histidine_xtb import (
    HistidineXtbError,
    HistidineXtbSelection,
    select_histidine_tautomer,
)
from mdprep.protonation.overrides import ManualOverrideError, resolve_manual_overrides
from mdprep.protonation.pka_rules import PkaDecision, PkaRuleError, decide_residue_state
from mdprep.protonation.propka import (
    PropkaExecutionError,
    PropkaWorkflowResult,
    run_propka_workflow,
)
from mdprep.protonation.propka_parser import PropkaParseError, PropkaRecord, map_propka_records
from mdprep.structure.classify import is_histidine, is_titratable_residue
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord


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
    pka: float | None = None
    ph: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

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
            "pka": self.pka,
            "ph": self.ph,
            "metadata": self.metadata,
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
    input_state_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    propka_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    xtb_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    propka_result: PropkaWorkflowResult | None = None
    xtb_selections: list[HistidineXtbSelection] = field(default_factory=list)
    hydrogen_atoms_removed: int = 0
    unresolved_histidines: list[dict[str, object]] = field(default_factory=list)
    titratable_residues_not_explicitly_assigned: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def records(self) -> list[ProtonationRecord]:
        return (
            self.manual_overrides_applied
            + self.disulfide_assignments_applied
            + self.input_state_assignments_applied
            + self.propka_assignments_applied
            + self.xtb_assignments_applied
        )

    def to_report_dict(self) -> dict[str, object]:
        changed = [record.to_dict() for record in self.records if record.changed]
        unchanged = [record.to_dict() for record in self.records if not record.changed]
        propka = self.propka_result.to_dict() if self.propka_result is not None else None
        parsed_pkas = (
            [record.to_dict() for record in self.propka_result.records]
            if self.propka_result is not None
            else []
        )
        return {
            "input_normalized_pdb_path": str(self.input_normalized_pdb_path),
            "output_protonation_pdb_path": str(self.output_protonation_pdb_path),
            "method": self.method,
            "ph": self.ph,
            "propka": propka,
            "parsed_pkas": parsed_pkas,
            "xtb_histidines": [selection.to_dict() for selection in self.xtb_selections],
            "temporary_water_hydrogens_for_xtb_clusters": [
                {
                    "histidine": _histidine_label(selection.residue),
                    "temporary_water_hydrogens_added": selection.temporary_water_hydrogens_added,
                    "waters_modified_for_xtb_only": selection.waters_modified_for_xtb_only,
                    "final_pdb_modified": selection.final_pdb_modified_by_temporary_water_hydrogens,
                }
                for selection in self.xtb_selections
                if selection.temporary_water_hydrogens_added
            ],
            "manual_overrides_applied": [record.to_dict() for record in self.manual_overrides_applied],
            "disulfide_assignments_applied": [
                record.to_dict() for record in self.disulfide_assignments_applied
            ],
            "input_state_assignments_applied": [
                record.to_dict() for record in self.input_state_assignments_applied
            ],
            "propka_assignments_applied": [
                record.to_dict() for record in self.propka_assignments_applied
            ],
            "xtb_assignments_applied": [
                record.to_dict() for record in self.xtb_assignments_applied
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
    try:
        manual_assignments = resolve_manual_overrides(structure, manifest)
        disulfide_assignments = resolve_disulfide_assignments(structure, manifest)
    except (ManualOverrideError, DisulfideAssignmentError) as exc:
        raise ProtonationApplicationError(str(exc)) from exc

    final_by_residue: dict[int, str] = {}
    manual_records: list[ProtonationRecord] = []
    manual_state_by_residue = {id(item.residue): item.requested_state for item in manual_assignments}

    for assignment in manual_assignments:
        final_by_residue[id(assignment.residue)] = assignment.requested_state
        manual_records.append(
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
                metadata={
                    "partner": assignment.partner.id.to_dict(),
                    "distance_angstrom": assignment.distance_angstrom,
                },
            )
        )

    warnings = list(structure.warnings)
    input_state_records: list[ProtonationRecord] = []
    propka_records: list[ProtonationRecord] = []
    xtb_records: list[ProtonationRecord] = []
    propka_result: PropkaWorkflowResult | None = None
    xtb_selections: list[HistidineXtbSelection] = []
    if manifest.protonation.method in {"propka", "propka_xtb_his"}:
        try:
            propka_result = run_propka_workflow(
                structure,
                manifest,
                work_dir=_protonation_work_dir(output_protonation_pdb_path) / "propka",
            )
            mapped_pkas = map_propka_records(structure, propka_result.records)
            pka_decisions = _decide_propka_states(
                structure,
                mapped_pkas=mapped_pkas,
                manifest=manifest,
                final_by_residue=final_by_residue,
            )
        except (PropkaExecutionError, PropkaParseError, PkaRuleError) as exc:
            raise ProtonationApplicationError(str(exc)) from exc

        xtb_needed: list[PkaDecision] = []
        for decision in pka_decisions:
            warnings.extend(decision.warnings)
            if decision.needs_xtb:
                xtb_needed.append(decision)
                continue
            if decision.final_state is None:
                continue
            final_by_residue[id(decision.residue)] = decision.final_state
            record = _record(
                decision.residue,
                final_resname=decision.final_state,
                source=decision.source,
                reason=decision.reason,
                selector=None,
                pka=decision.pka,
                ph=manifest.protonation.ph,
            )
            if decision.source == "input_state":
                input_state_records.append(record)
            else:
                propka_records.append(record)

        if xtb_needed and manifest.protonation.histidine.neutral_tautomer_method != "xtb":
            raise ProtonationApplicationError(
                "Neutral HIS residues require HID/HIE assignment; set "
                "protonation.histidine.neutral_tautomer_method: xtb or add manual overrides."
            )
        for decision in xtb_needed:
            try:
                selection = select_histidine_tautomer(
                    structure,
                    decision.residue,
                    manifest,
                    work_dir=_protonation_work_dir(output_protonation_pdb_path) / "histidine_xtb",
                    planned_states=final_by_residue,
                )
            except HistidineXtbError as exc:
                raise ProtonationApplicationError(str(exc)) from exc
            warnings.extend(selection.warnings)
            xtb_selections.append(selection)
            final_by_residue[id(decision.residue)] = selection.selected_state
            xtb_records.append(
                _record(
                    decision.residue,
                    final_resname=selection.selected_state,
                    source="propka_xtb_his",
                    reason=(
                        "Neutral HIS assigned by xTB HID/HIE comparison; "
                        f"delta(HID-HIE)={selection.delta_kcal_mol:.3f} kcal/mol"
                    ),
                    selector=None,
                    pka=decision.pka,
                    ph=manifest.protonation.ph,
                    metadata=selection.to_dict(),
                )
            )
    elif manifest.protonation.method != "manual_only":
        raise ProtonationApplicationError(
            f"Unsupported protonation method: {manifest.protonation.method}"
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
        for record in (
            manual_records
            + disulfide_records
            + input_state_records
            + propka_records
            + xtb_records
        )
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
        manual_overrides_applied=manual_records,
        disulfide_assignments_applied=disulfide_records,
        input_state_assignments_applied=input_state_records,
        propka_assignments_applied=propka_records,
        xtb_assignments_applied=xtb_records,
        propka_result=propka_result,
        xtb_selections=xtb_selections,
        hydrogen_atoms_removed=hydrogen_atoms_removed,
        unresolved_histidines=unresolved_his,
        titratable_residues_not_explicitly_assigned=unassigned_titratable,
        warnings=warnings,
    )


def _decide_propka_states(
    structure: PdbStructure,
    *,
    mapped_pkas: dict[int, PropkaRecord],
    manifest: ManifestConfig,
    final_by_residue: dict[int, str],
) -> list[PkaDecision]:
    decisions: list[PkaDecision] = []
    for residue in structure.residues:
        if id(residue) in final_by_residue:
            continue
        if not is_titratable_residue(residue):
            continue
        decision = decide_residue_state(
            residue,
            record=mapped_pkas.get(id(residue)),  # type: ignore[arg-type]
            ph=manifest.protonation.ph,
            method=manifest.protonation.method,  # type: ignore[arg-type]
        )
        if decision is not None:
            decisions.append(decision)
    return decisions


def _protonation_work_dir(output_protonation_pdb_path: str | Path) -> Path:
    output_path = Path(output_protonation_pdb_path)
    output_dir = output_path.parent.parent if output_path.parent.name == "intermediate" else output_path.parent
    return output_dir / "protonation"


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
    pka: float | None = None,
    ph: float | None = None,
    metadata: dict[str, object] | None = None,
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
        pka=pka,
        ph=ph,
        metadata={} if metadata is None else metadata,
    )


def _residue_dict(residue: ResidueRecord) -> dict[str, object]:
    return {
        **residue.id.to_dict(),
        "atom_count": len(residue.atoms),
        "record_names": sorted(residue.record_names),
        "original_index": residue.original_index,
    }


def _histidine_label(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    icode = residue.id.icode or ""
    return f"{chain}:{residue.id.resname}{residue.id.resid}{icode}"
