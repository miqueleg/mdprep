"""Safe structure-only normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.structure.classify import (
    is_likely_ligand_or_cofactor,
    is_nonstandard_nonwater_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


class StructureNormalizationError(ValueError):
    """Raised when structure normalization would require an unsafe decision."""


@dataclass(frozen=True)
class NormalizedLigand:
    ligand_id: str
    residue: dict[str, object]


@dataclass
class StructureNormalizationResult:
    input_path: Path
    output_path: Path | None
    input_summary: InspectionSummary
    normalized_structure: PdbStructure
    waters_kept: list[dict[str, object]] = field(default_factory=list)
    waters_removed: list[dict[str, object]] = field(default_factory=list)
    configured_ligands_kept: list[NormalizedLigand] = field(default_factory=list)
    unknown_heterogens_removed: list[dict[str, object]] = field(default_factory=list)
    unknown_heterogens_causing_failure: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_report_dict(self) -> dict[str, object]:
        normalized_summary = inspect_pdb_structure_from_structure(self.normalized_structure)
        return {
            "input_path": str(self.input_path),
            "output_normalized_pdb_path": str(self.output_path) if self.output_path else None,
            "atom_count_before": len(self.input_summary.structure.atoms),
            "atom_count_after": len(self.normalized_structure.atoms),
            "residue_count_before": len(self.input_summary.structure.residues),
            "residue_count_after": len(self.normalized_structure.residues),
            "chains": [{"chain_id": chain, "display": chain or "<blank>"} for chain in normalized_summary.chain_ids],
            "waters_kept": self.waters_kept,
            "waters_removed": self.waters_removed,
            "configured_ligands_kept": [
                {"id": ligand.ligand_id, **ligand.residue} for ligand in self.configured_ligands_kept
            ],
            "unknown_heterogens_removed": self.unknown_heterogens_removed,
            "unknown_heterogens_causing_failure": self.unknown_heterogens_causing_failure,
            "histidines": _residue_dicts(self.input_summary.histidines),
            "titratable_residues": _residue_dicts(self.input_summary.titratable_residues),
            "possible_disulfides": [
                candidate.to_dict() for candidate in self.input_summary.possible_disulfides
            ],
            "warnings": self.warnings,
        }


def normalize_structure_stage(
    manifest: ManifestConfig,
    *,
    output_path: str | Path | None = None,
) -> StructureNormalizationResult:
    input_path = Path(manifest.project.input_structure)
    structure = read_pdb(input_path, altloc_policy=manifest.structure.altloc_policy)
    input_summary = inspect_pdb_structure(
        input_path,
        altloc_policy=manifest.structure.altloc_policy,
        disulfide_cutoff_angstrom=manifest.disulfides.detection_cutoff_angstrom,
    )

    configured_ligands = _resolve_configured_ligands(structure, manifest)
    _validate_disulfide_selectors(structure, manifest)
    configured_ligand_keys = {id(residue) for _, residue in configured_ligands}
    likely_heterogens = [residue for residue in structure.residues if is_likely_ligand_or_cofactor(residue)]
    unknown_heterogens = [
        residue for residue in likely_heterogens if id(residue) not in configured_ligand_keys
    ]

    unknown_heterogen_dicts = _residue_dicts(unknown_heterogens)
    if unknown_heterogens and not manifest.structure.remove_unknown_heterogens:
        formatted = ", ".join(_format_residue_for_error(residue) for residue in unknown_heterogens)
        raise StructureNormalizationError(
            "Unknown heterogens are present but structure.remove_unknown_heterogens is false: "
            f"{formatted}. Add them to ligands: or set structure.remove_unknown_heterogens: true."
        )

    keep_residue_ids: set[int] = set()
    waters_kept: list[ResidueRecord] = []
    waters_removed: list[ResidueRecord] = []
    unknown_removed: list[ResidueRecord] = []

    for residue in structure.residues:
        if is_standard_protein_residue(residue):
            keep_residue_ids.add(id(residue))
        elif is_water_residue(residue):
            if manifest.structure.keep_crystal_waters:
                keep_residue_ids.add(id(residue))
                waters_kept.append(residue)
            else:
                waters_removed.append(residue)
        elif id(residue) in configured_ligand_keys:
            keep_residue_ids.add(id(residue))
        elif residue in unknown_heterogens:
            unknown_removed.append(residue)
        else:
            keep_residue_ids.add(id(residue))

    filtered_atoms: list[AtomRecord] = []
    for residue in structure.residues:
        if id(residue) in keep_residue_ids:
            filtered_atoms.extend(residue.atoms)

    normalized_structure = PdbStructure(
        path=Path(output_path) if output_path is not None else structure.path,
        atoms=filtered_atoms,
        residues=_filter_residues_by_ids(structure.residues, keep_residue_ids),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
    )

    warnings = list(structure.warnings)
    if manifest.structure.remove_input_hydrogens:
        warnings.append(
            "structure.remove_input_hydrogens is not applied during the structure-only stage."
        )

    return StructureNormalizationResult(
        input_path=input_path,
        output_path=Path(output_path) if output_path is not None else None,
        input_summary=input_summary,
        normalized_structure=normalized_structure,
        waters_kept=_residue_dicts(waters_kept),
        waters_removed=_residue_dicts(waters_removed),
        configured_ligands_kept=[
            NormalizedLigand(ligand_id=ligand_id, residue=_residue_dict(residue))
            for ligand_id, residue in configured_ligands
        ],
        unknown_heterogens_removed=_residue_dicts(unknown_removed),
        unknown_heterogens_causing_failure=unknown_heterogen_dicts
        if unknown_heterogens and not manifest.structure.remove_unknown_heterogens
        else [],
        warnings=warnings,
    )


def inspect_pdb_structure_from_structure(structure: PdbStructure) -> InspectionSummary:
    from mdprep.structure.classify import is_histidine, is_titratable_residue
    from mdprep.structure.disulfides import detect_possible_disulfides

    residues = structure.residues
    protein_residues = [residue for residue in residues if is_standard_protein_residue(residue)]
    water_residues = [residue for residue in residues if is_water_residue(residue)]
    heterogen_residues = [
        residue for residue in residues if is_nonstandard_nonwater_residue(residue)
    ]
    likely_ligands = [residue for residue in residues if is_likely_ligand_or_cofactor(residue)]
    histidines = [residue for residue in residues if is_histidine(residue)]
    titratable = [residue for residue in residues if is_titratable_residue(residue)]
    disulfides = detect_possible_disulfides(residues)
    return InspectionSummary(
        structure=structure,
        protein_residues=protein_residues,
        water_residues=water_residues,
        heterogen_residues=heterogen_residues,
        likely_ligands=likely_ligands,
        histidines=histidines,
        titratable_residues=titratable,
        possible_disulfides=disulfides,
    )


def _resolve_configured_ligands(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[tuple[str, ResidueRecord]]:
    resolved: list[tuple[str, ResidueRecord]] = []
    for ligand in manifest.ligands:
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError as exc:
            raise StructureNormalizationError(
                f"Ligand {ligand.id!r} selector did not resolve exactly one residue: {exc}"
            ) from exc
        resolved.append((ligand.id, residue))
    return resolved


def _validate_disulfide_selectors(structure: PdbStructure, manifest: ManifestConfig) -> None:
    for group_name, pairs in (
        ("force", manifest.disulfides.force),
        ("forbid", manifest.disulfides.forbid),
    ):
        for index, pair in enumerate(pairs, start=1):
            for side_name, selector in (("a", pair.a), ("b", pair.b)):
                try:
                    resolve_residue_selector(structure, selector.model_dump())
                except SelectorError as exc:
                    raise StructureNormalizationError(
                        f"Disulfide {group_name}[{index}].{side_name} selector did not resolve exactly one residue: {exc}"
                    ) from exc


def _filter_residues_by_ids(
    residues: list[ResidueRecord],
    keep_residue_ids: set[int],
) -> list[ResidueRecord]:
    return [residue for residue in residues if id(residue) in keep_residue_ids]


def _residue_dicts(residues: list[ResidueRecord]) -> list[dict[str, object]]:
    return [_residue_dict(residue) for residue in residues]


def _residue_dict(residue: ResidueRecord) -> dict[str, object]:
    return {
        **residue.id.to_dict(),
        "atom_count": len(residue.atoms),
        "record_names": sorted(residue.record_names),
        "original_index": residue.original_index,
    }


def _format_residue_for_error(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    icode = residue.id.icode or ""
    return f"{chain}:{residue.id.resname}{residue.id.resid}{icode}"
