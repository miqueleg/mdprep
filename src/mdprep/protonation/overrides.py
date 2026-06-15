"""Manual protonation override resolution."""

from __future__ import annotations

from dataclasses import dataclass

from mdprep.config.models import ManifestConfig
from mdprep.protonation.states import validate_state_transition
from mdprep.structure.models import ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector
from mdprep.structure.models import PdbStructure


class ManualOverrideError(ValueError):
    """Raised when a manual protonation override cannot be applied."""


@dataclass(frozen=True)
class ManualOverrideAssignment:
    residue: ResidueRecord
    requested_state: str
    reason: str
    selector: dict[str, object]


def resolve_manual_overrides(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[ManualOverrideAssignment]:
    assignments: list[ManualOverrideAssignment] = []
    for override in manifest.protonation.overrides:
        selector_data = override.selector.model_dump()
        selector_display = _selector_display(selector_data)
        try:
            residue = resolve_residue_selector(structure, selector_data)
        except SelectorError as exc:
            raise ManualOverrideError(
                f"Manual protonation selector did not resolve exactly one residue: {selector_display}. {exc}"
            ) from exc
        try:
            validate_state_transition(
                current_resname=residue.id.resname,
                requested_state=override.state,
                selector=selector_display,
            )
        except ValueError as exc:
            raise ManualOverrideError(str(exc)) from exc
        assignments.append(
            ManualOverrideAssignment(
                residue=residue,
                requested_state=override.state,
                reason=override.reason,
                selector=selector_data,
            )
        )
    return assignments


def _selector_display(selector: dict[str, object]) -> str:
    chain = selector.get("chain", selector.get("chain_id", ""))
    chain_display = chain if chain != "" else "<blank>"
    icode = selector.get("icode") or ""
    return f"{chain_display}:{selector.get('resname', '')}{selector['resid']}{icode}"

