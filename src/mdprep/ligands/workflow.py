"""Ligand extraction and parameterization stage."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from mdprep.ambertools.antechamber import run_antechamber
from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.ambertools.mol2 import Mol2Error, Mol2ValidationResult, validate_and_write_final_mol2
from mdprep.ambertools.parmchk2 import run_parmchk2
from mdprep.config.models import ManifestConfig
from mdprep.ligands.extract import ExtractedLigand, LigandExtractionError, extract_configured_ligands
from mdprep.structure.models import PdbStructure


UNIMPLEMENTED_QM_CHARGES = (
    "PySCF RESP/QMMESP ligand charges are not implemented yet; use am1bcc or user_mol2 for Task 6 workflows."
)


class LigandWorkflowError(ValueError):
    """Raised when ligand-stage preparation cannot complete safely."""


@dataclass(frozen=True)
class LigandWorkflowItem:
    ligand_id: str
    selector: dict[str, object]
    residue_identity: dict[str, object]
    atom_count: int
    charge_method: str
    atom_types: str
    net_charge: int
    multiplicity: int
    extracted_pdb_path: Path
    identity_path: Path
    final_mol2_path: Path | None = None
    final_frcmod_path: Path | None = None
    validation: Mol2ValidationResult | None = None
    antechamber: AmberToolRun | None = None
    parmchk2: AmberToolRun | None = None
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "ligand_id": self.ligand_id,
            "selector": self.selector,
            "residue_identity": self.residue_identity,
            "atom_count": self.atom_count,
            "charge_method": self.charge_method,
            "atom_types": self.atom_types,
            "net_charge": self.net_charge,
            "multiplicity": self.multiplicity,
            "extracted_pdb_path": str(self.extracted_pdb_path),
            "identity_path": str(self.identity_path),
            "antechamber": self.antechamber.to_dict() if self.antechamber else None,
            "parmchk2": self.parmchk2.to_dict() if self.parmchk2 else None,
            "final_mol2_path": str(self.final_mol2_path) if self.final_mol2_path else None,
            "final_frcmod_path": str(self.final_frcmod_path) if self.final_frcmod_path else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "warnings": self.warnings,
            "errors": [],
            "status": self.status,
        }


@dataclass(frozen=True)
class LigandStageResult:
    ligands: list[LigandWorkflowItem]

    def to_report_dict(self) -> dict[str, object]:
        return {"ligands": [item.to_dict() for item in self.ligands]}


def run_ligand_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
) -> LigandStageResult:
    try:
        extracted = extract_configured_ligands(structure, manifest, output_dir=output_dir)
        items = [_process_ligand(item, output_dir=output_dir) for item in extracted]
    except (LigandExtractionError, AmberToolsError, Mol2Error, FileNotFoundError) as exc:
        raise LigandWorkflowError(str(exc)) from exc
    return LigandStageResult(ligands=items)


def _process_ligand(extracted: ExtractedLigand, *, output_dir: str | Path) -> LigandWorkflowItem:
    ligand = extracted.config
    if ligand.charge_method in {"gas_resp_pyscf", "qmmesp_pyscf"}:
        raise LigandWorkflowError(UNIMPLEMENTED_QM_CHARGES)

    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    antechamber_run: AmberToolRun | None = None
    parmchk2_run: AmberToolRun | None = None
    warnings = list(extracted.warnings)

    if ligand.charge_method == "am1bcc":
        working_mol2 = parameters_dir / f"{ligand.id}.antechamber.mol2"
        antechamber_run = run_antechamber(
            ligand=ligand,
            input_pdb=extracted.pdb_path,
            output_mol2=working_mol2,
            residue_name=extracted.residue.id.resname,
            work_dir=parameters_dir,
        )
    elif ligand.charge_method == "user_mol2":
        assert ligand.user_mol2 is not None
        source = Path(ligand.user_mol2)
        if not source.exists():
            raise FileNotFoundError(f"Ligand {ligand.id} user_mol2 does not exist: {source}")
        working_mol2 = parameters_dir / f"{ligand.id}.user.mol2"
        shutil.copyfile(source, working_mol2)
    else:
        raise LigandWorkflowError(f"Unsupported ligand charge method: {ligand.charge_method}")

    final_mol2 = parameters_dir / f"{ligand.id}.final.mol2"
    validation = validate_and_write_final_mol2(
        mol2_path=working_mol2,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=final_mol2,
        charges_csv_path=parameters_dir / "charges.csv",
        validation_json_path=parameters_dir / "validation.json",
    )
    warnings.extend(validation.warnings)

    final_frcmod = parameters_dir / f"{ligand.id}.frcmod"
    if ligand.user_frcmod:
        source_frcmod = Path(ligand.user_frcmod)
        if not source_frcmod.exists():
            raise FileNotFoundError(f"Ligand {ligand.id} user_frcmod does not exist: {source_frcmod}")
        shutil.copyfile(source_frcmod, final_frcmod)
    else:
        parmchk2_run = run_parmchk2(
            ligand=ligand,
            input_mol2=final_mol2,
            output_frcmod=final_frcmod,
            work_dir=parameters_dir,
        )

    return LigandWorkflowItem(
        ligand_id=ligand.id,
        selector=ligand.selector.model_dump(mode="json"),
        residue_identity=extracted.residue.id.to_dict(),
        atom_count=len(extracted.atoms),
        charge_method=ligand.charge_method,
        atom_types=ligand.atom_types,
        net_charge=ligand.net_charge,
        multiplicity=ligand.multiplicity,
        extracted_pdb_path=extracted.pdb_path,
        identity_path=extracted.identity_path,
        final_mol2_path=final_mol2,
        final_frcmod_path=final_frcmod,
        validation=validation,
        antechamber=antechamber_run,
        parmchk2=parmchk2_run,
        warnings=warnings,
    )
