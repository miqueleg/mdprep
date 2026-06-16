"""Ligand residue extraction from prepared structures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mdprep.config.models import LigandConfig, ManifestConfig
from mdprep.ligands.identity import ligand_identity_dict
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.pdb import infer_element
from mdprep.structure.selectors import SelectorError, resolve_residue_selector
from mdprep.structure.writer import write_pdb


class LigandExtractionError(ValueError):
    """Raised when configured ligands cannot be extracted safely."""


@dataclass(frozen=True)
class ExtractedLigand:
    config: LigandConfig
    residue: ResidueRecord
    pdb_path: Path
    identity_path: Path
    warnings: list[str]

    @property
    def atoms(self):
        return self.residue.atoms

    def identity_dict(self) -> dict[str, object]:
        return ligand_identity_dict(
            ligand_id=self.config.id,
            selector=self.config.selector.model_dump(mode="json"),
            residue=self.residue,
            net_charge=self.config.net_charge,
            multiplicity=self.config.multiplicity,
            charge_method=self.config.charge_method,
            atom_types=self.config.atom_types,
        )


def extract_configured_ligands(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
) -> list[ExtractedLigand]:
    extracted: list[ExtractedLigand] = []
    for ligand in manifest.ligands:
        extracted.append(extract_ligand(structure, ligand, output_dir=output_dir))
    return extracted


def extract_ligand(
    structure: PdbStructure,
    ligand: LigandConfig,
    *,
    output_dir: str | Path,
) -> ExtractedLigand:
    try:
        residue = resolve_residue_selector(structure, ligand.selector.model_dump())
    except SelectorError as exc:
        raise LigandExtractionError(
            f"Ligand {ligand.id} selector did not resolve exactly one residue: {exc}"
        ) from exc
    if not residue.atoms:
        raise LigandExtractionError(f"Ligand {ligand.id} selector resolved a residue with zero atoms.")

    warnings: list[str] = []
    fixed_atoms = []
    for atom in residue.atoms:
        if atom.element is None:
            atom_field = atom.original_line[12:16] if len(atom.original_line) >= 16 else None
            inferred = infer_element(
                atom.name,
                resname=atom.resname,
                record_name=atom.record_name,
                atom_field=atom_field,
            )
            warnings.append(f"Inferred missing element for ligand {ligand.id} atom {atom.name}: {inferred}")
            fixed_atoms.append(atom.__class__(**{**atom.__dict__, "element": inferred}))
        else:
            fixed_atoms.append(atom)
    if fixed_atoms != residue.atoms:
        residue = ResidueRecord(
            id=residue.id,
            atoms=fixed_atoms,
            record_names=residue.record_names,
            original_index=residue.original_index,
        )

    ligand_dir = Path(output_dir) / "ligands" / ligand.id / "input"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = ligand_dir / f"{ligand.id}.pdb"
    identity_path = ligand_dir / "identity.json"
    ligand_structure = PdbStructure(
        path=pdb_path,
        atoms=list(residue.atoms),
        residues=[residue],
        model_count=1,
    )
    write_pdb(ligand_structure, pdb_path)
    extracted = ExtractedLigand(
        config=ligand,
        residue=residue,
        pdb_path=pdb_path,
        identity_path=identity_path,
        warnings=warnings,
    )
    identity_path.write_text(
        json.dumps(extracted.identity_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return extracted
