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
    conect_lines = _ligand_conect_lines(structure.path, residue.atoms)
    if conect_lines:
        _insert_conect_before_end(pdb_path, conect_lines)
        warnings.append(
            f"Preserved {len(conect_lines)} ligand CONECT record(s) for {ligand.id} in the extracted PDB."
        )
    elif _requires_pdb_chemistry_perception(ligand):
        warnings.append(
            "No ligand CONECT records were found in the input PDB. AmberTools will infer ligand "
            "connectivity from coordinates; for chemically sensitive or complex substrates, provide "
            "a curated user_mol2/user_frcmod so atom types and bonded terms are not inferred from PDB geometry."
        )
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


def _requires_pdb_chemistry_perception(ligand: LigandConfig) -> bool:
    if ligand.charge_method == "am1bcc":
        return True
    if ligand.charge_method in {"gas_resp_pyscf", "qmmesp_pyscf"}:
        return ligand.user_mol2 is None
    return False


def _ligand_conect_lines(input_path: Path, atoms: list) -> list[str]:
    serials = {atom.serial for atom in atoms if atom.serial is not None}
    if not serials or not input_path.exists():
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("CONECT"):
            continue
        numbers = _parse_conect_numbers(line)
        if not numbers:
            continue
        source, *targets = numbers
        if source not in serials:
            continue
        selected_targets = [target for target in targets if target in serials]
        for offset in range(0, len(selected_targets), 4):
            conect = _format_conect(source, selected_targets[offset : offset + 4])
            if conect not in seen:
                seen.add(conect)
                lines.append(conect)
    return lines


def _parse_conect_numbers(line: str) -> list[int]:
    numbers: list[int] = []
    for field in line[6:].split():
        try:
            numbers.append(int(field))
        except ValueError:
            continue
    return numbers


def _format_conect(source: int, targets: list[int]) -> str:
    return "CONECT" + f"{source:5d}" + "".join(f"{target:5d}" for target in targets) + "\n"


def _insert_conect_before_end(path: Path, conect_lines: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    insert_at = len(lines)
    if lines and lines[-1].strip() == "END":
        insert_at = len(lines) - 1
    lines[insert_at:insert_at] = conect_lines
    path.write_text("".join(lines), encoding="utf-8")
