"""Structure inspection summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdprep.structure.classify import (
    is_histidine,
    is_likely_ligand_or_cofactor,
    is_standard_protein_residue,
    is_titratable_residue,
    is_water_residue,
)
from mdprep.structure.disulfides import DisulfideCandidate, detect_possible_disulfides
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.pdb import AltlocPolicy, read_pdb


@dataclass(frozen=True)
class InspectionSummary:
    structure: PdbStructure
    protein_residues: list[ResidueRecord]
    water_residues: list[ResidueRecord]
    heterogen_residues: list[ResidueRecord]
    likely_ligands: list[ResidueRecord]
    histidines: list[ResidueRecord]
    titratable_residues: list[ResidueRecord]
    possible_disulfides: list[DisulfideCandidate]

    @property
    def chain_ids(self) -> list[str]:
        seen: list[str] = []
        for residue in self.structure.residues:
            if residue.id.chain_id not in seen:
                seen.append(residue.id.chain_id)
        return seen

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.structure.path),
            "total_atoms": len(self.structure.atoms),
            "total_residues": len(self.structure.residues),
            "model_count": self.structure.model_count,
            "used_model": self.structure.used_model,
            "warnings": list(self.structure.warnings),
            "chains": [{"chain_id": chain_id, "display": chain_id or "<blank>"} for chain_id in self.chain_ids],
            "counts": {
                "protein_residues": len(self.protein_residues),
                "water_residues": len(self.water_residues),
                "heterogen_residues": len(self.heterogen_residues),
                "likely_ligands": len(self.likely_ligands),
                "histidines": len(self.histidines),
                "titratable_residues": len(self.titratable_residues),
                "possible_disulfides": len(self.possible_disulfides),
            },
            "likely_ligands": [_residue_to_dict(residue) for residue in self.likely_ligands],
            "histidines": [_residue_to_dict(residue) for residue in self.histidines],
            "titratable_residues": [_residue_to_dict(residue) for residue in self.titratable_residues],
            "possible_disulfides": [candidate.to_dict() for candidate in self.possible_disulfides],
        }


def inspect_pdb_structure(
    path: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
    disulfide_cutoff_angstrom: float = 2.2,
) -> InspectionSummary:
    structure = read_pdb(path, altloc_policy=altloc_policy)
    residues = structure.residues
    protein_residues = [residue for residue in residues if is_standard_protein_residue(residue)]
    water_residues = [residue for residue in residues if is_water_residue(residue)]
    heterogen_residues = [
        residue
        for residue in residues
        if "HETATM" in residue.record_names and not is_water_residue(residue)
    ]
    likely_ligands = [residue for residue in residues if is_likely_ligand_or_cofactor(residue)]
    histidines = [residue for residue in residues if is_histidine(residue)]
    titratable_residues = [residue for residue in residues if is_titratable_residue(residue)]
    possible_disulfides = detect_possible_disulfides(
        residues,
        cutoff_angstrom=disulfide_cutoff_angstrom,
    )
    return InspectionSummary(
        structure=structure,
        protein_residues=protein_residues,
        water_residues=water_residues,
        heterogen_residues=heterogen_residues,
        likely_ligands=likely_ligands,
        histidines=histidines,
        titratable_residues=titratable_residues,
        possible_disulfides=possible_disulfides,
    )


def _residue_to_dict(residue: ResidueRecord) -> dict[str, object]:
    return {
        **residue.id.to_dict(),
        "atom_count": len(residue.atoms),
        "record_names": sorted(residue.record_names),
        "original_index": residue.original_index,
    }

