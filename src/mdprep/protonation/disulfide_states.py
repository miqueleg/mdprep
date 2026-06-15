"""CYX assignment from configured and detected disulfides."""

from __future__ import annotations

from dataclasses import dataclass

from mdprep.config.models import ManifestConfig
from mdprep.structure.disulfides import detect_possible_disulfides
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


class DisulfideAssignmentError(ValueError):
    """Raised when disulfide residue assignment is incompatible with manual states."""


@dataclass(frozen=True)
class DisulfideResidueAssignment:
    residue: ResidueRecord
    source: str
    reason: str
    partner: ResidueRecord
    distance_angstrom: float | None = None


def resolve_disulfide_assignments(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[DisulfideResidueAssignment]:
    assignments: list[DisulfideResidueAssignment] = []
    forced_pairs = _resolve_configured_pairs(structure, manifest.disulfides.force, "force")
    forbidden_pairs = {
        _pair_key(a, b)
        for a, b in _resolve_configured_pairs(structure, manifest.disulfides.forbid, "forbid")
    }
    forced_keys = {_pair_key(a, b) for a, b in forced_pairs}

    for residue_a, residue_b in forced_pairs:
        assignments.extend(
            _pair_assignments(
                residue_a,
                residue_b,
                source="forced_disulfide",
                reason="Configured disulfide force entry",
                distance_angstrom=None,
            )
        )

    if manifest.disulfides.auto_detect:
        candidates = detect_possible_disulfides(
            structure.residues,
            cutoff_angstrom=manifest.disulfides.detection_cutoff_angstrom,
        )
        residue_lookup = {_residue_identity(residue): residue for residue in structure.residues}
        for candidate in candidates:
            residue_a = residue_lookup[_residue_identity_from_dict(candidate.a.to_dict())]
            residue_b = residue_lookup[_residue_identity_from_dict(candidate.b.to_dict())]
            key = _pair_key(residue_a, residue_b)
            if key in forbidden_pairs or key in forced_keys:
                continue
            assignments.extend(
                _pair_assignments(
                    residue_a,
                    residue_b,
                    source="auto_disulfide",
                    reason="SG-SG distance within configured cutoff",
                    distance_angstrom=candidate.distance_angstrom,
                )
            )
    return _dedupe_assignments(assignments)


def _resolve_configured_pairs(
    structure: PdbStructure,
    pairs: list[object],
    label: str,
) -> list[tuple[ResidueRecord, ResidueRecord]]:
    resolved: list[tuple[ResidueRecord, ResidueRecord]] = []
    for index, pair in enumerate(pairs, start=1):
        try:
            residue_a = resolve_residue_selector(structure, pair.a.model_dump())  # type: ignore[attr-defined]
            residue_b = resolve_residue_selector(structure, pair.b.model_dump())  # type: ignore[attr-defined]
        except SelectorError as exc:
            raise DisulfideAssignmentError(
                f"Disulfide {label}[{index}] selector did not resolve exactly one residue: {exc}"
            ) from exc
        resolved.append((residue_a, residue_b))
    return resolved


def _pair_assignments(
    residue_a: ResidueRecord,
    residue_b: ResidueRecord,
    *,
    source: str,
    reason: str,
    distance_angstrom: float | None,
) -> list[DisulfideResidueAssignment]:
    return [
        DisulfideResidueAssignment(
            residue=residue_a,
            source=source,
            reason=reason,
            partner=residue_b,
            distance_angstrom=distance_angstrom,
        ),
        DisulfideResidueAssignment(
            residue=residue_b,
            source=source,
            reason=reason,
            partner=residue_a,
            distance_angstrom=distance_angstrom,
        ),
    ]


def _dedupe_assignments(
    assignments: list[DisulfideResidueAssignment],
) -> list[DisulfideResidueAssignment]:
    seen: set[tuple[tuple[str, int, str | None], str]] = set()
    deduped: list[DisulfideResidueAssignment] = []
    for assignment in assignments:
        key = (_residue_identity(assignment.residue), assignment.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(assignment)
    return deduped


def _pair_key(a: ResidueRecord, b: ResidueRecord) -> frozenset[tuple[str, int, str | None]]:
    return frozenset({_residue_identity(a), _residue_identity(b)})


def _residue_identity(residue: ResidueRecord) -> tuple[str, int, str | None]:
    return (residue.id.chain_id, residue.id.resid, residue.id.icode)


def _residue_identity_from_dict(data: dict[str, object]) -> tuple[str, int, str | None]:
    return (str(data["chain_id"]), int(data["resid"]), data["icode"])  # type: ignore[arg-type]

