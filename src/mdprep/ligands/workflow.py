"""Ligand extraction and parameterization stage."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from mdprep.ambertools.antechamber import run_antechamber
from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.ambertools.mol2 import Mol2Error, Mol2ValidationResult, validate_and_write_final_mol2
from mdprep.ambertools.parmchk2 import run_parmchk2
from mdprep.config.models import ManifestConfig
from mdprep.leap.forcefields import forcefield_sources
from mdprep.leap.log_parser import LeapLogError, assert_tleap_success
from mdprep.leap.residues import (
    LeapResidueError,
    disulfide_bond_commands,
    ligand_coordinate_commands,
    prepare_leap_input_pdb,
    validate_ligand_parameter_files,
)
from mdprep.leap.runner import TLeapRunError, run_tleap
from mdprep.ligands.extract import ExtractedLigand, LigandExtractionError, extract_configured_ligands
from mdprep.ligands.pyscf_charges import LigandPySCFChargeError, LigandPySCFChargeResult, derive_pyscf_charges
from mdprep.protonation.apply import ProtonationResult
from mdprep.qm.point_charges import PointChargeError, extract_point_charges_from_prmtop
from mdprep.structure.models import PdbStructure

if TYPE_CHECKING:
    from mdprep.leap.builder import TLeapOutputs


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
    qm: LigandPySCFChargeResult | None = None
    provisional_mol2_path: Path | None = None
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
            "qm": self.qm.to_dict() if self.qm else None,
            "provisional_mol2_path": str(self.provisional_mol2_path) if self.provisional_mol2_path else None,
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
    protonation_result: ProtonationResult | None = None,
) -> LigandStageResult:
    try:
        extracted = extract_configured_ligands(structure, manifest, output_dir=output_dir)
        items: list[LigandWorkflowItem | None] = [None] * len(extracted)
        qmmesp_indices: list[int] = []
        for index, item in enumerate(extracted):
            if item.config.charge_method == "qmmesp_pyscf":
                qmmesp_indices.append(index)
                items[index] = _prepare_provisional_qm_ligand(item, output_dir=output_dir)
            else:
                items[index] = _process_ligand(item, output_dir=output_dir)
        if qmmesp_indices:
            if protonation_result is None:
                raise LigandWorkflowError("qmmesp_pyscf requires a protonation result to build the provisional Amber system.")
            provisional_result = LigandStageResult(ligands=[item for item in items if item is not None])
            provisional_outputs = _build_qmmesp_provisional_system(
                structure,
                manifest,
                provisional_result,
                output_dir=output_dir,
                protonation_result=protonation_result,
            )
            for index in qmmesp_indices:
                extracted_ligand = extracted[index]
                provisional_item = items[index]
                assert provisional_item is not None
                point_charges = extract_point_charges_from_prmtop(
                    prmtop=provisional_outputs.prmtop,
                    inpcrd=provisional_outputs.inpcrd,
                    ligand=extracted_ligand.config,
                    manifest=manifest,
                    target_coordinates=_coordinates(extracted_ligand),
                )
                items[index] = _finalize_qm_ligand(
                    extracted_ligand,
                    output_dir=output_dir,
                    method_name="qmmesp_pyscf",
                    provisional_item=provisional_item,
                    point_charges=point_charges,
                )
    except (
        LigandExtractionError,
        AmberToolsError,
        Mol2Error,
        FileNotFoundError,
        LigandPySCFChargeError,
        PointChargeError,
        LeapLogError,
        LeapResidueError,
        TLeapRunError,
    ) as exc:
        raise LigandWorkflowError(str(exc)) from exc
    return LigandStageResult(ligands=[item for item in items if item is not None])


def _process_ligand(extracted: ExtractedLigand, *, output_dir: str | Path) -> LigandWorkflowItem:
    ligand = extracted.config
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    antechamber_run: AmberToolRun | None = None
    parmchk2_run: AmberToolRun | None = None
    qm_result: LigandPySCFChargeResult | None = None
    provisional_mol2_path: Path | None = None
    warnings = list(extracted.warnings)

    if ligand.charge_method in {"am1bcc", "gas_resp_pyscf"}:
        if ligand.charge_method == "gas_resp_pyscf" and ligand.user_mol2:
            working_mol2 = _copy_user_mol2(ligand, parameters_dir, suffix="provisional_user")
            warnings.append(
                "User mol2 supplied provisional atom types and bonded topology for gas_resp_pyscf; "
                "its charges will be replaced by PySCF-fitted gas-phase charges."
            )
        else:
            working_mol2 = parameters_dir / f"{ligand.id}.antechamber.mol2"
            antechamber_run = run_antechamber(
                ligand=ligand,
                input_pdb=extracted.pdb_path,
                output_mol2=working_mol2,
                residue_name=extracted.residue.id.resname,
                work_dir=parameters_dir,
            )
        if ligand.charge_method == "gas_resp_pyscf":
            provisional_mol2_path = working_mol2
            pyscf_mol2 = parameters_dir / f"{ligand.id}.pyscf_charges.mol2"
            qm_result = derive_pyscf_charges(
                extracted=extracted,
                provisional_mol2_path=working_mol2,
                output_mol2_path=pyscf_mol2,
                output_dir=output_dir,
                method_name="gas_resp_pyscf",
                point_charges=None,
            )
            working_mol2 = pyscf_mol2
            if antechamber_run is None:
                warnings.append("User mol2 charges were provisional and replaced by PySCF-fitted gas-phase charges.")
            else:
                warnings.append("AM1-BCC charges were provisional and replaced by PySCF-fitted gas-phase charges.")
    elif ligand.charge_method == "user_mol2":
        working_mol2 = _copy_user_mol2(ligand, parameters_dir, suffix="user")
    elif ligand.charge_method == "qmmesp_pyscf":
        raise LigandWorkflowError("qmmesp_pyscf is processed in the QMMESP two-pass ligand workflow.")
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
        qm=qm_result,
        provisional_mol2_path=provisional_mol2_path,
        warnings=warnings,
    )


def _prepare_provisional_qm_ligand(
    extracted: ExtractedLigand,
    *,
    output_dir: str | Path,
) -> LigandWorkflowItem:
    ligand = extracted.config
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(extracted.warnings)
    antechamber_run: AmberToolRun | None = None
    parmchk2_run: AmberToolRun | None = None
    if ligand.user_mol2:
        source_mol2 = _copy_user_mol2(ligand, parameters_dir, suffix="provisional_user")
        warnings.append(
            "User mol2 supplied provisional atom types and bonded topology for QMMESP; "
            "its charges will be replaced by PySCF-fitted QMMESP charges."
        )
    else:
        source_mol2 = parameters_dir / f"{ligand.id}.provisional_antechamber.mol2"
        antechamber_run = run_antechamber(
            ligand=ligand,
            input_pdb=extracted.pdb_path,
            output_mol2=source_mol2,
            residue_name=extracted.residue.id.resname,
            work_dir=parameters_dir,
        )
        warnings.append("AM1-BCC charges are provisional for QMMESP and will be replaced by PySCF-fitted charges.")
    provisional_mol2 = parameters_dir / f"{ligand.id}.provisional.mol2"
    validation = validate_and_write_final_mol2(
        mol2_path=source_mol2,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=provisional_mol2,
        charges_csv_path=parameters_dir / "provisional_charges.csv",
        validation_json_path=parameters_dir / "provisional_validation.json",
    )
    warnings.extend(validation.warnings)
    provisional_frcmod = parameters_dir / f"{ligand.id}.frcmod"
    if ligand.user_frcmod:
        source_frcmod = Path(ligand.user_frcmod)
        if not source_frcmod.exists():
            raise FileNotFoundError(f"Ligand {ligand.id} user_frcmod does not exist: {source_frcmod}")
        shutil.copyfile(source_frcmod, provisional_frcmod)
        warnings.append("User frcmod supplied provisional bonded parameters for QMMESP.")
    else:
        parmchk2_run = run_parmchk2(
            ligand=ligand,
            input_mol2=provisional_mol2,
            output_frcmod=provisional_frcmod,
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
        final_mol2_path=provisional_mol2,
        final_frcmod_path=provisional_frcmod,
        validation=validation,
        antechamber=antechamber_run,
        parmchk2=parmchk2_run,
        provisional_mol2_path=source_mol2,
        warnings=warnings,
    )


def _finalize_qm_ligand(
    extracted: ExtractedLigand,
    *,
    output_dir: str | Path,
    method_name: str,
    provisional_item: LigandWorkflowItem,
    point_charges: object | None,
) -> LigandWorkflowItem:
    ligand = extracted.config
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    assert provisional_item.final_mol2_path is not None
    pyscf_mol2 = parameters_dir / f"{ligand.id}.pyscf_charges.mol2"
    qm_result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path=provisional_item.final_mol2_path,
        output_mol2_path=pyscf_mol2,
        output_dir=output_dir,
        method_name=method_name,
        point_charges=point_charges,  # type: ignore[arg-type]
    )
    final_mol2 = parameters_dir / f"{ligand.id}.final.mol2"
    validation = validate_and_write_final_mol2(
        mol2_path=pyscf_mol2,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=final_mol2,
        charges_csv_path=parameters_dir / "charges.csv",
        validation_json_path=parameters_dir / "validation.json",
    )
    warnings = list(provisional_item.warnings)
    warnings.extend(qm_result.warnings)
    warnings.append("Final mol2 charges are PySCF QMMESP-fitted charges; provisional charges were replaced.")
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
        final_frcmod_path=provisional_item.final_frcmod_path,
        validation=validation,
        antechamber=provisional_item.antechamber,
        parmchk2=provisional_item.parmchk2,
        qm=qm_result,
        provisional_mol2_path=provisional_item.final_mol2_path,
        warnings=warnings,
    )


def _build_qmmesp_provisional_system(
    structure: PdbStructure,
    manifest: ManifestConfig,
    ligand_result: LigandStageResult,
    *,
    output_dir: str | Path,
    protonation_result: ProtonationResult | None,
) -> "TLeapOutputs":
    from mdprep.leap.builder import TLeapOutputs, build_tleap_script

    output = Path(output_dir)
    work_dir = output / "qmmesp" / "provisional_leap"
    input_dir = work_dir / "input"
    work_dir.mkdir(parents=True, exist_ok=True)
    leap_input = prepare_leap_input_pdb(
        structure,
        input_dir / "provisional_input.pdb",
        manifest=manifest,
        ligand_result=ligand_result,
    )
    sources = forcefield_sources(
        protein_forcefield=manifest.protein.forcefield,
        water_model=manifest.protein.water_model,
        ligands=manifest.ligands,
    )
    ligand_files = validate_ligand_parameter_files(
        manifest=manifest,
        structure=leap_input.structure,
        ligand_result=ligand_result,
    )
    disulfides = (
        disulfide_bond_commands(structure=leap_input.structure, protonation_result=protonation_result)
        if protonation_result is not None
        else []
    )
    coordinate_commands = ligand_coordinate_commands(
        manifest=manifest,
        structure=leap_input.structure,
    )
    outputs = TLeapOutputs(
        prmtop=work_dir / "provisional.prmtop",
        inpcrd=work_dir / "provisional.inpcrd",
        pdb=work_dir / "provisional.pdb",
    )
    script = build_tleap_script(
        sources=sources,
        ligands=ligand_files,
        input_pdb=leap_input.path,
        disulfide_bonds=disulfides,
        ligand_coordinate_commands=coordinate_commands,
        outputs=outputs,
        work_dir=work_dir,
    )
    script_path = work_dir / "tleap.in"
    script_path.write_text(script, encoding="utf-8")
    run = run_tleap(script_path, work_dir=work_dir)
    assert_tleap_success(run.summary, fail_on_warnings=manifest.validation.fail_on_warnings, context="QMMESP provisional")
    for path in [outputs.prmtop, outputs.inpcrd, outputs.pdb]:
        if not path.exists() or path.stat().st_size == 0:
            raise LigandWorkflowError(f"QMMESP provisional tleap did not produce {path}")
    return outputs


def _coordinates(extracted: ExtractedLigand):
    import numpy as np

    return np.asarray([[atom.x, atom.y, atom.z] for atom in extracted.atoms], dtype=float)


def _copy_user_mol2(ligand, parameters_dir: Path, *, suffix: str) -> Path:
    assert ligand.user_mol2 is not None
    source = Path(ligand.user_mol2)
    if not source.exists():
        raise FileNotFoundError(f"Ligand {ligand.id} user_mol2 does not exist: {source}")
    target = parameters_dir / f"{ligand.id}.{suffix}.mol2"
    shutil.copyfile(source, target)
    return target
