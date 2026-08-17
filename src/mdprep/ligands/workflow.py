"""Ligand extraction and parameterization stage."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from mdprep.ambertools.antechamber import run_antechamber
from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.ambertools.mol2 import (
    Mol2Error,
    Mol2ValidationResult,
    read_mol2,
    validate_and_write_final_mol2,
    write_mol2_with_charges,
)
from mdprep.ambertools.parmchk2 import run_parmchk2
from mdprep.config.models import ManifestConfig
from mdprep.leap.forcefields import forcefield_sources
from mdprep.leap.log_parser import LeapLogError, assert_tleap_success
from mdprep.leap.residues import (
    LeapResidueError,
    append_disulfide_conect_records,
    disulfide_bond_commands,
    prepare_leap_input_pdb,
    validate_ligand_parameter_files,
)
from mdprep.leap.runner import TLeapRun, TLeapRunError, run_tleap
from mdprep.ligands.extract import ExtractedLigand, LigandExtractionError, extract_configured_ligands
from mdprep.ligands.pyscf_charges import LigandPySCFChargeError, LigandPySCFChargeResult, derive_pyscf_charges
from mdprep.protonation.apply import ProtonationResult
from mdprep.qm.point_charges import (
    PointChargeError,
    PointChargeSelection,
    extract_point_charges_from_prmtop,
)
from mdprep.structure.models import PdbStructure


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
    provisional_ligands: list[LigandWorkflowItem] = field(default_factory=list)
    qmmesp_provisional_system: "QmmespProvisionalSystem | None" = None

    def to_report_dict(self) -> dict[str, object]:
        return {
            "ligands": [item.to_dict() for item in self.ligands],
            "qmmesp_provisional_ligands": [item.to_dict() for item in self.provisional_ligands],
            "qmmesp_provisional_system": (
                self.qmmesp_provisional_system.to_dict() if self.qmmesp_provisional_system else None
            ),
        }


@dataclass(frozen=True)
class QmmespProvisionalSystem:
    leap_input_path: Path
    script_path: Path
    tleap_run: TLeapRun
    prmtop_path: Path
    inpcrd_path: Path
    pdb_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow": "all_configured_ligands_provisional_am1bcc",
            "leap_input_path": str(self.leap_input_path),
            "script_path": str(self.script_path),
            "tleap": self.tleap_run.to_dict(),
            "prmtop_path": str(self.prmtop_path),
            "inpcrd_path": str(self.inpcrd_path),
            "pdb_path": str(self.pdb_path),
        }


def run_ligand_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
    protonation_result: ProtonationResult | None = None,
) -> LigandStageResult:
    try:
        extracted = extract_configured_ligands(structure, manifest, output_dir=output_dir)
        qmmesp_indices = [
            index for index, item in enumerate(extracted) if item.config.charge_method == "qmmesp_pyscf"
        ]
        if not qmmesp_indices:
            return LigandStageResult(
                ligands=[_process_ligand(item, output_dir=output_dir) for item in extracted]
            )

        if qmmesp_indices:
            if protonation_result is None:
                raise LigandWorkflowError("qmmesp_pyscf requires a protonation result to build the provisional Amber system.")
            provisional_items = [
                _prepare_provisional_am1bcc_ligand(item, output_dir=output_dir) for item in extracted
            ]
            provisional_result = LigandStageResult(ligands=provisional_items)
            provisional_system = _build_qmmesp_provisional_system(
                structure,
                manifest,
                provisional_result,
                output_dir=output_dir,
                protonation_result=protonation_result,
                context="QMMESP provisional",
            )
            final_items: list[LigandWorkflowItem] = []
            for index, extracted_ligand in enumerate(extracted):
                provisional_item = provisional_items[index]
                if extracted_ligand.config.charge_method != "qmmesp_pyscf":
                    if extracted_ligand.config.charge_method == "am1bcc":
                        final_items.append(
                            _promote_provisional_am1bcc_ligand(
                                extracted_ligand,
                                provisional_item=provisional_item,
                                output_dir=output_dir,
                            )
                        )
                    else:
                        final_items.append(
                            _process_ligand(
                                extracted_ligand,
                                output_dir=output_dir,
                                qmmesp_provisional_item=provisional_item,
                            )
                        )
                    continue
                extracted_ligand = extracted[index]
                point_charges = extract_point_charges_from_prmtop(
                    prmtop=provisional_system.prmtop_path,
                    inpcrd=provisional_system.inpcrd_path,
                    ligand=extracted_ligand.config,
                    manifest=manifest,
                    target_coordinates=_coordinates(extracted_ligand),
                    target_atom_names=[atom.name for atom in extracted_ligand.atoms],
                )
                final_items.append(
                    _finalize_qm_ligand(
                        extracted_ligand,
                        output_dir=output_dir,
                        method_name="qmmesp_pyscf",
                        provisional_item=provisional_item,
                        point_charges=point_charges,
                    )
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
        _write_ligand_failure_report(output_dir, exc)
        raise LigandWorkflowError(str(exc)) from exc
    return LigandStageResult(
        ligands=final_items,
        provisional_ligands=provisional_items,
        qmmesp_provisional_system=provisional_system,
    )


def build_provisional_am1bcc_system(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
    protonation_result: ProtonationResult,
) -> tuple[LigandStageResult, QmmespProvisionalSystem]:
    """Build an unsolvated Amber system with AM1-BCC for every ligand.

    This is shared by QMMESP charge fitting and the optional QM/MM refinement.
    It deliberately ignores each ligand's requested *final* charge method: the
    returned parameters are provisional and may never be promoted implicitly.
    """

    try:
        extracted = extract_configured_ligands(
            structure,
            manifest,
            output_dir=output_dir,
        )
        provisional_items = [
            _prepare_provisional_am1bcc_ligand(item, output_dir=output_dir)
            for item in extracted
        ]
        provisional_result = LigandStageResult(ligands=provisional_items)
        provisional_system = _build_qmmesp_provisional_system(
            structure,
            manifest,
            provisional_result,
            output_dir=output_dir,
            protonation_result=protonation_result,
            context="QM/MM refinement provisional",
        )
    except (
        LigandExtractionError,
        AmberToolsError,
        Mol2Error,
        FileNotFoundError,
        LeapLogError,
        LeapResidueError,
        TLeapRunError,
    ) as exc:
        _write_ligand_failure_report(output_dir, exc)
        raise LigandWorkflowError(str(exc)) from exc
    return provisional_result, provisional_system


def _write_ligand_failure_report(output_dir: str | Path, exc: Exception) -> None:
    """Persist structured diagnostics even when ligand processing stops early."""

    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    warnings = [line for line in lines if line.startswith("WARNING")]
    errors = [line for line in lines if line.startswith("ERROR")]
    if not errors:
        errors = [str(exc)]
    report_path = Path(output_dir) / "reports" / "ligand_failure.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "warnings": warnings,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _process_ligand(
    extracted: ExtractedLigand,
    *,
    output_dir: str | Path,
    qmmesp_provisional_item: LigandWorkflowItem | None = None,
) -> LigandWorkflowItem:
    ligand = extracted.config
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    antechamber_run: AmberToolRun | None = None
    parmchk2_run: AmberToolRun | None = None
    qm_result: LigandPySCFChargeResult | None = None
    provisional_mol2_path: Path | None = None
    warnings = list(extracted.warnings)

    if ligand.charge_method in {"am1bcc", "gas_resp_pyscf", "mcpb_resp_pyscf"}:
        if ligand.charge_method == "gas_resp_pyscf" and ligand.user_mol2:
            working_mol2 = _copy_user_mol2(ligand, parameters_dir, suffix="provisional_user")
            warnings.append(
                "User mol2 supplied provisional atom types and bonded topology for gas_resp_pyscf; "
                "its charges will be replaced by PySCF-fitted gas-phase charges."
            )
        elif qmmesp_provisional_item is not None:
            if qmmesp_provisional_item.final_mol2_path is None:
                raise LigandWorkflowError(f"Ligand {ligand.id} has no provisional AM1-BCC mol2.")
            working_mol2 = qmmesp_provisional_item.final_mol2_path
            antechamber_run = qmmesp_provisional_item.antechamber
            warnings.append(
                "Reused the all-ligand provisional AM1-BCC scaffold for final gas-phase RESP fitting."
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
        if ligand.charge_method == "mcpb_resp_pyscf":
            provisional_mol2_path = working_mol2
            warnings.append(
                "AM1-BCC supplies only the provisional GAFF topology and charges for this "
                "metal-bound ligand. MCPB.py will replace the charge distribution with the "
                "embedded PySCF large-model RESP fit before the final Amber build."
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
    if qmmesp_provisional_item is not None and provisional_mol2_path is None:
        provisional_mol2_path = qmmesp_provisional_item.final_mol2_path

    final_frcmod = parameters_dir / f"{ligand.id}.frcmod"
    if ligand.user_frcmod:
        source_frcmod = Path(ligand.user_frcmod)
        if not source_frcmod.exists():
            raise FileNotFoundError(f"Ligand {ligand.id} user_frcmod does not exist: {source_frcmod}")
        shutil.copyfile(source_frcmod, final_frcmod)
    elif qmmesp_provisional_item is not None and qmmesp_provisional_item.final_frcmod_path is not None:
        shutil.copyfile(qmmesp_provisional_item.final_frcmod_path, final_frcmod)
        parmchk2_run = qmmesp_provisional_item.parmchk2
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
        status=(
            "provisional_for_mcpb"
            if ligand.charge_method == "mcpb_resp_pyscf"
            else "ok"
        ),
    )


def _prepare_provisional_am1bcc_ligand(
    extracted: ExtractedLigand,
    *,
    output_dir: str | Path,
) -> LigandWorkflowItem:
    """Generate an AM1-BCC provisional model for one configured ligand.

    This always runs antechamber, including for ligands whose final method is
    user_mol2 or QMMESP.  If a user mol2 is present, its atom typing and bonded
    topology are retained while the generated AM1-BCC charges are copied onto
    it for the provisional QM/MM system.
    """

    ligand = extracted.config
    provisional_dir = Path(output_dir) / "ligands" / ligand.id / "provisional_am1bcc"
    provisional_dir.mkdir(parents=True, exist_ok=True)
    warnings = list(extracted.warnings)

    generated_mol2 = provisional_dir / f"{ligand.id}.antechamber_am1bcc.mol2"
    antechamber_run = run_antechamber(
        ligand=ligand,
        input_pdb=extracted.pdb_path,
        output_mol2=generated_mol2,
        residue_name=extracted.residue.id.resname,
        work_dir=provisional_dir,
    )
    validated_am1bcc = provisional_dir / f"{ligand.id}.am1bcc.validated.mol2"
    generated_validation = validate_and_write_final_mol2(
        mol2_path=generated_mol2,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=validated_am1bcc,
        charges_csv_path=provisional_dir / "generated_am1bcc_charges.csv",
        validation_json_path=provisional_dir / "generated_am1bcc_validation.json",
    )
    warnings.extend(generated_validation.warnings)
    warnings.append(
        "Generated AM1-BCC charges for this ligand for the all-ligand provisional Amber system."
    )

    source_mol2 = validated_am1bcc
    if ligand.user_mol2:
        user_scaffold = _copy_user_mol2(ligand, provisional_dir, suffix="user_scaffold")
        am1bcc_charges = [atom.charge for atom in read_mol2(validated_am1bcc).atoms]
        source_mol2 = provisional_dir / f"{ligand.id}.user_scaffold_am1bcc.mol2"
        write_mol2_with_charges(user_scaffold, am1bcc_charges, source_mol2)
        warnings.append(
            "Preserved the user mol2 atom types and bonded topology in the provisional system, "
            "but replaced its provisional charges with generated AM1-BCC charges."
        )

    provisional_mol2 = provisional_dir / f"{ligand.id}.provisional.mol2"
    validation = validate_and_write_final_mol2(
        mol2_path=source_mol2,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=provisional_mol2,
        charges_csv_path=provisional_dir / "provisional_charges.csv",
        validation_json_path=provisional_dir / "provisional_validation.json",
    )
    warnings.extend(validation.warnings)

    provisional_frcmod = provisional_dir / f"{ligand.id}.provisional.frcmod"
    parmchk2_run: AmberToolRun | None = None
    if ligand.user_frcmod:
        source_frcmod = Path(ligand.user_frcmod)
        if not source_frcmod.exists():
            raise FileNotFoundError(f"Ligand {ligand.id} user_frcmod does not exist: {source_frcmod}")
        shutil.copyfile(source_frcmod, provisional_frcmod)
        warnings.append("User frcmod supplied bonded parameters for the provisional Amber system.")
    else:
        parmchk2_run = run_parmchk2(
            ligand=ligand,
            input_mol2=provisional_mol2,
            output_frcmod=provisional_frcmod,
            work_dir=provisional_dir,
        )
    return LigandWorkflowItem(
        ligand_id=ligand.id,
        selector=ligand.selector.model_dump(mode="json"),
        residue_identity=extracted.residue.id.to_dict(),
        atom_count=len(extracted.atoms),
        charge_method="am1bcc",
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
        provisional_mol2_path=generated_mol2,
        warnings=warnings,
    )


def _promote_provisional_am1bcc_ligand(
    extracted: ExtractedLigand,
    *,
    provisional_item: LigandWorkflowItem,
    output_dir: str | Path,
) -> LigandWorkflowItem:
    ligand = extracted.config
    if provisional_item.final_mol2_path is None or provisional_item.final_frcmod_path is None:
        raise LigandWorkflowError(f"Ligand {ligand.id} has incomplete provisional AM1-BCC parameters.")
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    final_mol2 = parameters_dir / f"{ligand.id}.final.mol2"
    validation = validate_and_write_final_mol2(
        mol2_path=provisional_item.final_mol2_path,
        extracted_atoms=extracted.atoms,
        ligand=ligand,
        final_mol2_path=final_mol2,
        charges_csv_path=parameters_dir / "charges.csv",
        validation_json_path=parameters_dir / "validation.json",
    )
    final_frcmod = parameters_dir / f"{ligand.id}.frcmod"
    shutil.copyfile(provisional_item.final_frcmod_path, final_frcmod)
    warnings = list(provisional_item.warnings)
    warnings.extend(validation.warnings)
    warnings.append("Promoted the all-ligand provisional AM1-BCC model to the final AM1-BCC model.")
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
        antechamber=provisional_item.antechamber,
        parmchk2=provisional_item.parmchk2,
        provisional_mol2_path=provisional_item.final_mol2_path,
        warnings=warnings,
    )


def _finalize_qm_ligand(
    extracted: ExtractedLigand,
    *,
    output_dir: str | Path,
    method_name: str,
    provisional_item: LigandWorkflowItem,
    point_charges: PointChargeSelection | None,
) -> LigandWorkflowItem:
    ligand = extracted.config
    parameters_dir = Path(output_dir) / "ligands" / ligand.id / "parameters"
    parameters_dir.mkdir(parents=True, exist_ok=True)
    assert provisional_item.final_mol2_path is not None
    pyscf_mol2 = parameters_dir / f"{ligand.id}.pyscf_charges.mol2"
    qm_result = derive_pyscf_charges(
        extracted=extracted,
        provisional_mol2_path=provisional_item.final_mol2_path,
        output_mol2_path=pyscf_mol2,
        output_dir=output_dir,
        method_name=method_name,
        point_charges=point_charges,
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
    if provisional_item.final_frcmod_path is None:
        raise LigandWorkflowError(f"Ligand {ligand.id} has no provisional bonded parameter file.")
    final_frcmod = parameters_dir / f"{ligand.id}.frcmod"
    shutil.copyfile(provisional_item.final_frcmod_path, final_frcmod)
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
    context: str = "QMMESP provisional",
) -> QmmespProvisionalSystem:
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
    append_disulfide_conect_records(leap_input.path, disulfides)
    metal_setup_commands = _qmmesp_provisional_metal_setup(
        structure,
        manifest,
        output_dir=output / "qmmesp" / "provisional_metals",
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
        outputs=outputs,
        work_dir=work_dir,
        setup_commands=metal_setup_commands,
    )
    script_path = work_dir / "tleap.in"
    script_path.write_text(script, encoding="utf-8")
    run = run_tleap(script_path, work_dir=work_dir)
    assert_tleap_success(
        run.summary,
        fail_on_warnings=manifest.validation.fail_on_warnings,
        context=context,
    )
    for path in [outputs.prmtop, outputs.inpcrd, outputs.pdb]:
        if not path.exists() or path.stat().st_size == 0:
            raise LigandWorkflowError(f"{context} tleap did not produce {path}")
    return QmmespProvisionalSystem(
        leap_input_path=leap_input.path,
        script_path=script_path,
        tleap_run=run,
        prmtop_path=outputs.prmtop,
        inpcrd_path=outputs.inpcrd,
        pdb_path=outputs.pdb,
    )


def _qmmesp_provisional_metal_setup(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: Path,
) -> list[str]:
    """Parameterize configured ions in a provisional Amber environment."""

    if not manifest.metals:
        return []
    from mdprep.config.models import NonbondedMetalConfig
    from mdprep.metals.coordination import resolve_metal_sites
    from mdprep.metals.nonbonded import prepare_nonbonded_site
    from mdprep.structure.selectors import resolve_residue_selector

    sites = resolve_metal_sites(structure, manifest, validate_mcpb_cutoff=True)
    qmmesp_residue_keys: set[tuple[str, int, str | None]] = set()
    for ligand in manifest.ligands:
        if ligand.charge_method != "qmmesp_pyscf":
            continue
        residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        qmmesp_residue_keys.add((residue.id.chain_id, residue.id.resid, residue.id.icode))
    commands: list[str] = []
    parameter_by_type: dict[str, tuple[str, str]] = {}
    template_by_resname: dict[str, tuple[str, int, str]] = {}
    for site in sites:
        if site.config.model == "bonded_mcpb":
            for bond in site.bonds:
                key = (
                    bond.coordinator_residue.id.chain_id,
                    bond.coordinator_residue.id.resid,
                    bond.coordinator_residue.id.icode,
                )
                if key in qmmesp_residue_keys:
                    raise LigandWorkflowError(
                        f"QMMESP target ligand is part of bonded MCPB.py site {site.config.id!r}. "
                        "MCPB.py step 3 would replace that ligand's QMMESP charges, so these two "
                        "charge models cannot be combined for the same coordinating residue."
                    )
            for residue in site.additional_residues:
                key = (residue.id.chain_id, residue.id.resid, residue.id.icode)
                if key in qmmesp_residue_keys:
                    raise LigandWorkflowError(
                        f"QMMESP target ligand is an additional residue in bonded MCPB.py site "
                        f"{site.config.id!r}. MCPB.py step 3 would replace that ligand's QMMESP "
                        "charges, so these two charge models cannot be combined for that residue."
                    )
            assert site.config.mcpb is not None
            family = site.config.mcpb.provisional_nonbonded_parameter_set
            if family is None:  # normally rejected during manifest validation
                raise LigandWorkflowError(
                    f"Bonded MCPB.py site {site.config.id!r} needs "
                    "mcpb.provisional_nonbonded_parameter_set for the provisional Amber system."
                )
            temporary_config = site.config.model_copy(
                update={
                    "model": "nonbonded",
                    "nonbonded": NonbondedMetalConfig(parameter_set=family),
                    "mcpb": None,
                }
            )
            provisional_site = replace(site, config=temporary_config)
        else:
            provisional_site = site
        result = prepare_nonbonded_site(
            provisional_site,
            water_model=manifest.protein.water_model,
            output_dir=output_dir / provisional_site.config.id,
        )
        for artifact in result.ions:
            parameter = artifact.parameter
            signature = (parameter.parameter_set, parameter.frcmod_name)
            previous = parameter_by_type.get(parameter.atom_type)
            if previous is not None and previous != signature:
                raise LigandWorkflowError(
                    f"Provisional Amber topology requests atom type "
                    f"{parameter.atom_type} with conflicting ion models {previous} and "
                    f"{signature}."
                )
            parameter_by_type[parameter.atom_type] = signature
            residue_name = artifact.ion.residue.id.resname
            template = (
                artifact.ion.element,
                artifact.ion.charge,
                artifact.ion.atom.name,
            )
            prior_template = template_by_resname.get(residue_name)
            if prior_template is not None and prior_template != template:
                raise LigandWorkflowError(
                    f"Provisional Amber topology maps metal residue name {residue_name!r} "
                    f"to conflicting templates {prior_template} and {template}."
                )
            template_by_resname[residue_name] = template
        commands.extend(result.leap_commands)
    return _deduplicate_tleap_setup(commands)


def _deduplicate_tleap_setup(commands: list[str]) -> list[str]:
    result: list[str] = []
    seen_commands: set[str] = set()
    seen_blocks: set[tuple[str, ...]] = set()
    index = 0
    while index < len(commands):
        if commands[index] == "addAtomTypes {":
            end = index + 1
            while end < len(commands) and commands[end] != "}":
                end += 1
            if end >= len(commands):
                raise LigandWorkflowError(
                    "Unterminated addAtomTypes block in provisional metal setup."
                )
            block = tuple(commands[index : end + 1])
            if block not in seen_blocks:
                seen_blocks.add(block)
                result.extend(block)
            index = end + 1
            continue
        command = commands[index]
        if command not in seen_commands:
            seen_commands.add(command)
            result.append(command)
        index += 1
    return result


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
