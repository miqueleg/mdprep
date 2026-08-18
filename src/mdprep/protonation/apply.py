"""Apply safe protonation-stage residue-name changes."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.metals.coordination import MetalCoordinationError, resolve_metal_sites
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
from mdprep.protonation.temporary_hydrogenation import (
    TemporaryHydrogenationError,
    TemporaryHydrogenationResult,
    add_temporary_protein_hydrogens,
)
from mdprep.structure.classify import (
    is_histidine,
    is_standard_protein_residue,
    is_titratable_residue,
    is_water_residue,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


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
    metal_coordination_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    disulfide_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    input_state_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    propka_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    xtb_assignments_applied: list[ProtonationRecord] = field(default_factory=list)
    propka_result: PropkaWorkflowResult | None = None
    xtb_selections: list[HistidineXtbSelection] = field(default_factory=list)
    xtb_temporary_hydrogenation: TemporaryHydrogenationResult | None = None
    hydrogen_atoms_removed: int = 0
    unresolved_histidines: list[dict[str, object]] = field(default_factory=list)
    titratable_residues_not_explicitly_assigned: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reused_initial_assignments_after_refinement: bool = False
    preserved_refined_hydrogen_count: int = 0

    @property
    def records(self) -> list[ProtonationRecord]:
        return (
            self.manual_overrides_applied
            + self.metal_coordination_assignments_applied
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
            "xtb_temporary_protein_hydrogenation": (
                self.xtb_temporary_hydrogenation.to_dict()
                if self.xtb_temporary_hydrogenation is not None
                else None
            ),
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
            "metal_coordination_assignments_applied": [
                record.to_dict() for record in self.metal_coordination_assignments_applied
            ],
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
            "reused_initial_assignments_after_refinement": (
                self.reused_initial_assignments_after_refinement
            ),
            "preserved_refined_hydrogen_count": self.preserved_refined_hydrogen_count,
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
        metal_sites = resolve_metal_sites(
            structure,
            manifest,
            validate_mcpb_cutoff=False,
        )
    except (ManualOverrideError, DisulfideAssignmentError, MetalCoordinationError) as exc:
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

    metal_records = _apply_metal_histidine_constraints(
        metal_sites,
        final_by_residue=final_by_residue,
        manual_state_by_residue=manual_state_by_residue,
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
    xtb_temporary_hydrogenation: TemporaryHydrogenationResult | None = None
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
        xtb_structure = structure
        xtb_config = manifest.protonation.histidine.xtb
        if (
            xtb_needed
            and xtb_config.add_missing_protein_hydrogens
            and _protein_hydrogenation_is_incomplete(structure)
        ):
            try:
                xtb_temporary_hydrogenation = add_temporary_protein_hydrogens(
                    structure,
                    ph=manifest.protonation.ph,
                    random_seed=xtb_config.temporary_hydrogen_random_seed,
                    work_dir=(
                        _protonation_work_dir(output_protonation_pdb_path)
                        / "histidine_xtb"
                        / "temporary_environment_hydrogenation"
                    ),
                )
            except TemporaryHydrogenationError as exc:
                raise ProtonationApplicationError(str(exc)) from exc
            xtb_structure = xtb_temporary_hydrogenation.structure
            warnings.append(
                "PDBFixer added temporary protein hydrogens for xTB HID/HIE "
                "ranking; these hydrogens are not written to the protonation or "
                "final prepared PDB."
            )
        pending_xtb_states = _initial_xtb_environment_states(xtb_needed, warnings)
        for decision in xtb_needed:
            try:
                selection_states = dict(pending_xtb_states)
                selection_states.update(final_by_residue)
                selection_residue = _matching_residue(
                    xtb_structure,
                    decision.residue,
                )
                selection = select_histidine_tautomer(
                    xtb_structure,
                    selection_residue,
                    manifest,
                    work_dir=_protonation_work_dir(output_protonation_pdb_path) / "histidine_xtb",
                    planned_states=_map_residue_states(
                        selection_states,
                        source_structure=structure,
                        target_structure=xtb_structure,
                    ),
                )
            except HistidineXtbError as exc:
                raise ProtonationApplicationError(str(exc)) from exc
            selection = replace(selection, residue=decision.residue)
            warnings.extend(selection.warnings)
            xtb_selections.append(selection)
            final_by_residue[id(decision.residue)] = selection.selected_state
            pending_xtb_states[id(decision.residue)] = selection.selected_state
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
        protected_ligand_keys = _configured_ligand_atom_keys(structure, manifest)
        renamed_atoms = [
            atom
            for atom in renamed_atoms
            if not (is_hydrogen_atom(atom) and _atom_residue_key(atom) not in protected_ligand_keys)
        ]
        hydrogen_atoms_removed = before - len(renamed_atoms)
        if protected_ligand_keys:
            warnings.append(
                "Configured ligand residues were excluded from structure.remove_input_hydrogens so ligand "
                "parameterization sees the submitted ligand hydrogenation."
            )

    protonated_structure = PdbStructure(
        path=Path(output_protonation_pdb_path),
        atoms=renamed_atoms,
        residues=_build_residues(renamed_atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
        conect_bonds={
            bond
            for bond in structure.conect_bonds
            if bond[0] in {atom.serial for atom in renamed_atoms}
            and bond[1] in {atom.serial for atom in renamed_atoms}
        },
        ter_after_serials={
            serial
            for serial in structure.ter_after_serials
            if serial in {atom.serial for atom in renamed_atoms}
        },
    )
    explicit_keys = {
        (record.chain, record.resid, record.icode)
        for record in (
            manual_records
            + metal_records
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
        metal_coordination_assignments_applied=metal_records,
        disulfide_assignments_applied=disulfide_records,
        input_state_assignments_applied=input_state_records,
        propka_assignments_applied=propka_records,
        xtb_assignments_applied=xtb_records,
        propka_result=propka_result,
        xtb_selections=xtb_selections,
        xtb_temporary_hydrogenation=xtb_temporary_hydrogenation,
        hydrogen_atoms_removed=hydrogen_atoms_removed,
        unresolved_histidines=unresolved_his,
        titratable_residues_not_explicitly_assigned=unassigned_titratable,
        warnings=warnings,
    )


def reuse_protonation_assignments_after_refinement(
    structure: PdbStructure,
    previous: ProtonationResult,
    *,
    input_refined_pdb_path: str | Path,
    output_protonation_pdb_path: str | Path,
) -> ProtonationResult:
    """Restore initial residue states while retaining every refined hydrogen.

    Hydrogen-only refinement is meaningful only if its optimized proton
    coordinates reach the MCPB/charge/final-tleap stages.  This function
    reapplies the already reviewed pre-refinement assignments to the
    coordinate-transfer structure and deliberately does not rerun PropKa/xTB.
    """

    final_by_key: dict[tuple[str, int, str | None], str] = {}
    for record in previous.records:
        key = (record.chain, record.resid, record.icode)
        existing = final_by_key.get(key)
        if existing is not None and existing != record.final_resname:
            raise ProtonationApplicationError(
                "Pre-refinement protonation records contain conflicting final "
                f"states for {key}: {existing} and {record.final_resname}."
            )
        final_by_key[key] = record.final_resname

    observed_keys = {
        (residue.id.chain_id, residue.id.resid, residue.id.icode)
        for residue in structure.residues
    }
    missing = sorted(key for key in final_by_key if key not in observed_keys)
    if missing:
        raise ProtonationApplicationError(
            "QM/MM coordinate transfer lost residues with reviewed protonation "
            f"assignments: {missing}"
        )
    atoms = [
        replace(
            atom,
            resname=final_by_key.get(
                (atom.chain_id, atom.resid, atom.icode),
                atom.resname,
            ),
        )
        for atom in structure.atoms
    ]
    output = Path(output_protonation_pdb_path)
    reused_structure = PdbStructure(
        path=output,
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
        conect_bonds=set(structure.conect_bonds),
        ter_after_serials=set(structure.ter_after_serials),
    )
    _validate_reused_refined_hydrogens(reused_structure)
    hydrogen_count = sum(is_hydrogen_atom(atom) for atom in reused_structure.atoms)
    if hydrogen_count == 0:
        raise ProtonationApplicationError(
            "post_refinement_protonation: reuse_initial selected a refined "
            "structure containing no hydrogens."
        )
    warning = (
        "Initial PropKa/xTB/manual protonation assignments were reused after "
        "QM/MM refinement so all refined protein, ligand, and water hydrogen "
        "coordinates remain available to MCPB.py, QMMESP, and final tleap."
    )
    return ProtonationResult(
        input_normalized_pdb_path=Path(input_refined_pdb_path),
        output_protonation_pdb_path=output,
        method=previous.method,
        ph=previous.ph,
        structure=reused_structure,
        manual_overrides_applied=list(previous.manual_overrides_applied),
        metal_coordination_assignments_applied=list(
            previous.metal_coordination_assignments_applied
        ),
        disulfide_assignments_applied=list(previous.disulfide_assignments_applied),
        input_state_assignments_applied=list(previous.input_state_assignments_applied),
        propka_assignments_applied=list(previous.propka_assignments_applied),
        xtb_assignments_applied=list(previous.xtb_assignments_applied),
        propka_result=previous.propka_result,
        xtb_selections=list(previous.xtb_selections),
        xtb_temporary_hydrogenation=previous.xtb_temporary_hydrogenation,
        hydrogen_atoms_removed=0,
        unresolved_histidines=list(previous.unresolved_histidines),
        titratable_residues_not_explicitly_assigned=list(
            previous.titratable_residues_not_explicitly_assigned
        ),
        warnings=[*previous.warnings, warning],
        reused_initial_assignments_after_refinement=True,
        preserved_refined_hydrogen_count=hydrogen_count,
    )


def _validate_reused_refined_hydrogens(structure: PdbStructure) -> None:
    for residue in structure.residues:
        state = residue.id.resname
        if state in {"HID", "HIE", "HIP"}:
            nd1 = _hydrogen_count_near_any(residue, ("ND1",))
            ne2 = _hydrogen_count_near_any(residue, ("NE2",))
            expected = {"HID": (1, 0), "HIE": (0, 1), "HIP": (1, 1)}[state]
            if (nd1, ne2) != expected:
                raise ProtonationApplicationError(
                    f"Refined {residue.id.display()} hydrogen pattern is {(nd1, ne2)} "
                    f"at ND1/NE2; expected {expected} for {state}."
                )
        elif state in {"ASP", "GLU"}:
            names = ("OD1", "OD2") if state == "ASP" else ("OE1", "OE2")
            if _hydrogen_count_near_any(residue, names) != 0:
                raise ProtonationApplicationError(
                    f"Refined {residue.id.display()} retains a carboxyl proton "
                    f"in deprotonated state {state}."
                )
        elif state in {"ASH", "GLH"}:
            names = ("OD1", "OD2") if state == "ASH" else ("OE1", "OE2")
            if _hydrogen_count_near_any(residue, names) != 1:
                raise ProtonationApplicationError(
                    f"Refined {residue.id.display()} does not contain exactly one "
                    f"carboxyl proton for {state}."
                )
        elif state == "LYN" and _hydrogen_count_near_any(residue, ("NZ",)) != 2:
            raise ProtonationApplicationError(
                f"Refined {residue.id.display()} does not contain two NZ "
                "hydrogens for LYN."
            )
        elif state == "LYS" and _hydrogen_count_near_any(residue, ("NZ",)) != 3:
            raise ProtonationApplicationError(
                f"Refined {residue.id.display()} does not contain three NZ "
                "hydrogens for LYS."
            )
        elif state in {"CYS", "CYM", "CYX"}:
            # A normal S-H bond is longer than N-H/O-H.  The general 1.25 A
            # proximity cutoff used above would therefore reject a chemically
            # intact CYS thiol after coordinate rounding or QM/MM relaxation.
            observed = _hydrogen_count_near_any(
                residue,
                ("SG",),
                cutoff_angstrom=1.50,
            )
            expected = 1 if state == "CYS" else 0
            if observed != expected:
                raise ProtonationApplicationError(
                    f"Refined {residue.id.display()} has {observed} SG hydrogens; "
                    f"expected {expected} for {state}."
                )
        elif is_water_residue(residue):
            water_hydrogens = sum(is_hydrogen_atom(atom) for atom in residue.atoms)
            if water_hydrogens != 2:
                raise ProtonationApplicationError(
                    f"Refined water {residue.id.display()} contains "
                    f"{water_hydrogens} hydrogens; TIP3P requires two."
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


def _initial_xtb_environment_states(
    decisions: list[PkaDecision],
    warnings: list[str],
) -> dict[int, str]:
    states: dict[int, str] = {}
    for decision in decisions:
        residue = decision.residue
        inferred = _input_histidine_state_from_n_hydrogens(residue)
        if inferred in {"HID", "HIE"}:
            states[id(residue)] = inferred
            warnings.append(
                f"Neutral histidine {residue.id.display()} is pending xTB assignment; "
                f"using input-like {inferred} as its temporary environment state until its own HID/HIE comparison runs."
            )
        else:
            states[id(residue)] = "HIE"
            warnings.append(
                f"Neutral histidine {residue.id.display()} is pending xTB assignment and has ambiguous or missing "
                "imidazole hydrogens in the input; using HIE as a deterministic temporary environment state until "
                "its own HID/HIE comparison runs."
            )
    return states


def _protein_hydrogenation_is_incomplete(structure: PdbStructure) -> bool:
    """Return whether any standard residue has no explicit hydrogen at all."""

    return any(
        not any(is_hydrogen_atom(atom) for atom in residue.atoms)
        for residue in structure.residues
        if is_standard_protein_residue(residue)
    )


def _matching_residue(
    structure: PdbStructure,
    reference: ResidueRecord,
) -> ResidueRecord:
    matches = [
        residue
        for residue in structure.residues
        if (
            residue.id.chain_id,
            residue.id.resid,
            residue.id.icode,
        )
        == (
            reference.id.chain_id,
            reference.id.resid,
            reference.id.icode,
        )
    ]
    if len(matches) != 1:
        raise ProtonationApplicationError(
            "Temporary histidine environment mapping did not resolve exactly one "
            f"copy of {reference.id.display()}."
        )
    if matches[0].id.resname != reference.id.resname:
        raise ProtonationApplicationError(
            "Temporary histidine environment changed residue identity: "
            f"{reference.id.display()} -> {matches[0].id.display()}."
        )
    return matches[0]


def _map_residue_states(
    states: dict[int, str],
    *,
    source_structure: PdbStructure,
    target_structure: PdbStructure,
) -> dict[int, str]:
    if source_structure is target_structure:
        return states
    state_by_key = {
        (residue.id.chain_id, residue.id.resid, residue.id.icode): states[id(residue)]
        for residue in source_structure.residues
        if id(residue) in states
    }
    mapped: dict[int, str] = {}
    for residue in target_structure.residues:
        key = (residue.id.chain_id, residue.id.resid, residue.id.icode)
        if key in state_by_key:
            mapped[id(residue)] = state_by_key[key]
    if len(mapped) != len(state_by_key):
        raise ProtonationApplicationError(
            "Temporary histidine environment did not preserve every planned "
            "protonation-state residue."
        )
    return mapped


def _input_histidine_state_from_n_hydrogens(residue: ResidueRecord) -> str | None:
    nd1_h = _hydrogen_count_near_any(residue, ("ND1",))
    ne2_h = _hydrogen_count_near_any(residue, ("NE2",))
    if nd1_h == 1 and ne2_h == 0:
        return "HID"
    if nd1_h == 0 and ne2_h == 1:
        return "HIE"
    if nd1_h == 1 and ne2_h == 1:
        return "HIP"
    return None


def _hydrogen_count_near_any(
    residue: ResidueRecord,
    atom_names: tuple[str, ...],
    *,
    cutoff_angstrom: float = 1.25,
) -> int:
    return sum(
        1
        for atom in residue.atoms
        if is_hydrogen_atom(atom) and _hydrogen_is_near_any(atom, residue, atom_names, cutoff_angstrom=cutoff_angstrom)
    )


def _hydrogen_is_near_any(
    hydrogen: AtomRecord,
    residue: ResidueRecord,
    atom_names: tuple[str, ...],
    *,
    cutoff_angstrom: float = 1.25,
) -> bool:
    for atom in residue.atoms:
        if atom.name.strip() not in atom_names:
            continue
        distance = (
            (hydrogen.x - atom.x) ** 2
            + (hydrogen.y - atom.y) ** 2
            + (hydrogen.z - atom.z) ** 2
        ) ** 0.5
        if distance <= cutoff_angstrom:
            return True
    return False


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


def _configured_ligand_atom_keys(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> set[tuple[str, str, int, str | None]]:
    keys: set[tuple[str, str, int, str | None]] = set()
    for ligand in manifest.ligands:
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError as exc:
            raise ProtonationApplicationError(
                f"Ligand {ligand.id} selector did not resolve exactly one residue before hydrogen removal: {exc}"
            ) from exc
        keys.add((residue.id.chain_id, residue.id.resname, residue.id.resid, residue.id.icode))
    return keys


def _atom_residue_key(atom: AtomRecord) -> tuple[str, str, int, str | None]:
    return (atom.chain_id, atom.resname, atom.resid, atom.icode)


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


def _apply_metal_histidine_constraints(
    metal_sites: list[object],
    *,
    final_by_residue: dict[int, str],
    manual_state_by_residue: dict[int, str],
) -> list[ProtonationRecord]:
    """Apply safe protonation constraints from explicit MCPB bonds.

    A bond through ND1 requires ND1 to be unprotonated (HIE), while a bond
    through NE2 requires NE2 to be unprotonated (HID).  Coordination does not
    uniquely determine the protonation of other titratable sidechains, so they
    require a user override instead of inheriting a metal-blind prediction.
    """

    required_by_residue: dict[int, tuple[ResidueRecord, str, list[dict[str, object]]]] = {}
    for site in metal_sites:
        if site.config.model != "bonded_mcpb":  # type: ignore[attr-defined]
            continue
        for bond in site.bonds:  # type: ignore[attr-defined]
            residue = bond.coordinator_residue
            if not is_histidine(residue):
                if is_titratable_residue(residue):
                    if id(residue) not in manual_state_by_residue:
                        raise ProtonationApplicationError(
                            f"Metal-coordinating titratable residue {residue.id.display()}@"
                            f"{bond.coordinator.name} requires an explicit protonation override. "
                            "Coordination alone does not safely distinguish neutral and ionized "
                            "ASP/GLU/CYS/LYS/ARG chemistry."
                        )
                elif residue.id.resname == "TYR":
                    raise ProtonationApplicationError(
                        f"Metal coordination through {residue.id.display()}@{bond.coordinator.name} "
                        "is unsupported because mdprep does not currently expose neutral/phenolate "
                        "TYR protonation states."
                    )
                continue
            donor = bond.coordinator.name
            if donor == "ND1":
                required = "HIE"
            elif donor == "NE2":
                required = "HID"
            else:
                raise ProtonationApplicationError(
                    f"Histidine metal coordinator {residue.id.display()}@{donor} is not ND1 or NE2; "
                    "add a manual protonation override and correct the declared coordination atom."
                )
            metadata = {
                "metal_site_id": site.config.id,  # type: ignore[attr-defined]
                "metal_atom": bond.ion.atom.atom_identity,
                "coordinator_atom": bond.coordinator.atom_identity,
                "distance_angstrom": bond.distance_angstrom,
            }
            existing = required_by_residue.get(id(residue))
            if existing is not None and existing[1] != required:
                raise ProtonationApplicationError(
                    f"Metal bonds require incompatible HID and HIE states for {residue.id.display()}. "
                    "A histidine cannot coordinate through both neutral imidazole nitrogens."
                )
            if existing is None:
                required_by_residue[id(residue)] = (residue, required, [metadata])
            else:
                existing[2].append(metadata)

    records: list[ProtonationRecord] = []
    for residue_id, (residue, required, bonds) in required_by_residue.items():
        manual = manual_state_by_residue.get(residue_id)
        if manual is not None:
            if manual != required:
                raise ProtonationApplicationError(
                    f"Manual protonation override for {residue.id.display()} requests {manual}, "
                    f"but its explicitly declared metal bond requires {required}. Correct either "
                    "the override or the coordinating atom; mdprep will not silently replace a manual choice."
                )
            continue
        final_by_residue[residue_id] = required
        donor = "ND1" if required == "HIE" else "NE2"
        records.append(
            _record(
                residue,
                final_resname=required,
                source="metal_coordination",
                reason=(
                    f"Explicit metal coordination through {donor} requires that donor to be "
                    f"unprotonated; assigned {required}."
                ),
                selector=None,
                metadata={"bonds": bonds},
            )
        )
    return records


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
