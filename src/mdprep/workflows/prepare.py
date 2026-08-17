"""Preparation workflow."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from mdprep import __version__
from mdprep.config.loader import load_manifest
from mdprep.external.runner import run_command
from mdprep.leap.builder import TLeapBuildError, TLeapStageResult, run_tleap_stage
from mdprep.leap.report import write_tleap_reports
from mdprep.ligands.report import write_ligand_reports
from mdprep.ligands.workflow import LigandStageResult, LigandWorkflowError, run_ligand_stage
from mdprep.metals.report import write_metal_reports
from mdprep.metals.workflow import (
    MetalStageResult,
    MetalWorkflowError,
    promote_mcpb_resp_ligands,
    run_metal_stage,
)
from mdprep.md.roe_brooks import RoeBrooksError, RoeBrooksResult, run_roe_brooks
from mdprep.protonation.apply import (
    ProtonationApplicationError,
    ProtonationResult,
    apply_protonation_stage,
    reuse_protonation_assignments_after_refinement,
)
from mdprep.protonation.report import write_protonation_reports
from mdprep.refinement.ash import AshRefinementError, capture_ash_refinement_cache
from mdprep.refinement.report import write_refinement_reports
from mdprep.refinement.workflow import (
    RefinementResult,
    RefinementWorkflowError,
    run_refinement_stage,
)
from mdprep.reports.structure_report import write_structure_reports
from mdprep.structure.normalize import StructureNormalizationResult, normalize_structure_stage
from mdprep.structure.writer import write_pdb
from mdprep.validation.openmm_check import openmm_version
from mdprep.validation.parmed_check import parmed_version
from mdprep.qm.pyscf_runner import pyscf_version
from mdprep.validation.topology import FinalValidationError, validate_final_outputs, write_validation_reports


class PrepareWorkflowError(ValueError):
    """Raised when the requested preparation workflow is not available."""


@dataclass
class PrepareResult:
    stage: str
    structure_result: StructureNormalizationResult
    protonation_result: ProtonationResult | None = None
    refinement_result: RefinementResult | None = None
    ligand_result: LigandStageResult | None = None
    metal_result: MetalStageResult | None = None
    tleap_result: TLeapStageResult | None = None
    md_result: RoeBrooksResult | None = None

    @property
    def output_path(self) -> Path | None:
        if self.md_result is not None:
            return self.md_result.final_pdb
        if self.tleap_result is not None:
            return self.tleap_result.final_outputs.pdb
        if self.metal_result is not None:
            if self.metal_result.mcpb_site is not None:
                site = self.metal_result.mcpb_site
                return site.restored_final_pdb_path or site.mcpb_input_pdb_path
            if self.metal_result.structure is not None:
                return self.metal_result.structure.path
        if self.protonation_result is not None:
            return self.protonation_result.output_protonation_pdb_path
        return self.structure_result.output_path


def prepare_system(
    manifest_path: str | Path,
    *,
    stop_after: str | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> PrepareResult:
    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file, resolve_paths=True)
    if stop_after is None:
        stop_after = "md" if manifest.molecular_dynamics.enabled else "tleap"
    if stop_after not in {
        "structure",
        "protonation",
        "refinement",
        "ligands",
        "metals",
        "tleap",
        "md",
    }:
        raise PrepareWorkflowError(
            "Unsupported stop stage; use --stop-after structure, --stop-after protonation, "
            "--stop-after refinement, --stop-after ligands, --stop-after metals, or "
            "--stop-after tleap, or --stop-after md."
        )
    if stop_after == "md" and not manifest.molecular_dynamics.enabled:
        raise PrepareWorkflowError(
            "--stop-after md requires molecular_dynamics.enabled: true in the manifest"
        )
    if stop_after == "refinement" and not manifest.refinement.enabled:
        raise PrepareWorkflowError(
            "--stop-after refinement requires refinement.enabled: true in the manifest"
        )
    output_dir = Path(manifest.project.output_dir)
    if overwrite and resume:
        raise PrepareWorkflowError("--overwrite and --resume are mutually exclusive")
    if resume and not output_dir.exists():
        raise FileNotFoundError(
            f"Cannot resume because the output directory does not exist: {output_dir}"
        )
    if output_dir.exists() and not overwrite and not resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace mdprep outputs.")
    cached_ash_run = None
    if resume and manifest.refinement.enabled and stop_after in {
        "refinement",
        "ligands",
        "metals",
        "tleap",
        "md",
    }:
        try:
            cached_ash_run = capture_ash_refinement_cache(
                output_dir / "refinement" / "ash"
            )
        except AshRefinementError as exc:
            raise PrepareWorkflowError(str(exc)) from exc

    intermediate_dir = output_dir / "intermediate"
    reports_dir = output_dir / "reports"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_input_path = output_dir / "manifest.input.yaml"
    shutil.copyfile(manifest_file, manifest_input_path)

    normalized_pdb = intermediate_dir / "00_input_normalized.pdb"
    result = normalize_structure_stage(manifest, output_path=normalized_pdb)
    write_pdb(result.normalized_structure, normalized_pdb)
    result.output_path = normalized_pdb

    report = write_structure_reports(
        result,
        json_path=reports_dir / "structure_report.json",
        markdown_path=reports_dir / "structure_report.md",
    )
    _write_manifest_lock(
        manifest=manifest,
        structure_report=report,
        path=output_dir / "manifest.lock.yaml",
        input_manifest_path=manifest_file,
    )
    _write_versions(output_dir / "versions.json")

    protonation_report = None
    protonation_result = None
    refinement_report = None
    refinement_result = None
    ligand_report = None
    ligand_result = None
    metal_report = None
    metal_result = None
    tleap_report = None
    validation_report = None
    tleap_result = None
    if stop_after in {"protonation", "refinement", "ligands", "metals", "tleap", "md"}:
        protonation_pdb = intermediate_dir / "01_protonation_assigned.pdb"
        protonation_result = apply_protonation_stage(
            result.normalized_structure,
            manifest,
            input_normalized_pdb_path=normalized_pdb,
            output_protonation_pdb_path=protonation_pdb,
        )
        write_pdb(protonation_result.structure, protonation_pdb)
        preliminary_report_name = (
            "protonation_pre_refinement_report"
            if manifest.refinement.enabled
            else "protonation_report"
        )
        protonation_report = write_protonation_reports(
            protonation_result,
            json_path=reports_dir / f"{preliminary_report_name}.json",
            csv_path=reports_dir / f"{preliminary_report_name}.csv",
            markdown_path=reports_dir / f"{preliminary_report_name}.md",
        )
        _write_manifest_lock(
            manifest=manifest,
            structure_report=report,
            path=output_dir / "manifest.lock.yaml",
            input_manifest_path=manifest_file,
            protonation_report=protonation_report,
        )
        _write_versions(
            output_dir / "versions.json",
            external_executables=_external_executables_from_protonation_report(protonation_report),
        )

    if manifest.refinement.enabled and stop_after in {
        "refinement",
        "ligands",
        "metals",
        "tleap",
        "md",
    }:
        assert protonation_result is not None
        refinement_kwargs = {}
        if cached_ash_run is not None:
            refinement_kwargs["cached_ash_run"] = cached_ash_run
        refinement_result = run_refinement_stage(
            protonated_structure=protonation_result.structure,
            reference_structure=result.normalized_structure,
            manifest=manifest,
            output_dir=output_dir,
            protonation_result=protonation_result,
            **refinement_kwargs,
        )
        refinement_report = write_refinement_reports(
            refinement_result,
            json_path=reports_dir / "refinement_report.json",
            markdown_path=reports_dir / "refinement_report.md",
        )
        final_protonation_pdb = intermediate_dir / "03_final_protonation_assigned.pdb"
        if manifest.refinement.post_refinement_protonation == "reuse_initial":
            protonation_result = reuse_protonation_assignments_after_refinement(
                refinement_result.structure,
                protonation_result,
                input_refined_pdb_path=(
                    refinement_result.refined_hydrogenated_pdb_path
                ),
                output_protonation_pdb_path=final_protonation_pdb,
            )
        else:
            protonation_result = apply_protonation_stage(
                refinement_result.structure,
                manifest,
                input_normalized_pdb_path=(
                    refinement_result.refined_hydrogenated_pdb_path
                ),
                output_protonation_pdb_path=final_protonation_pdb,
            )
        write_pdb(protonation_result.structure, final_protonation_pdb)
        protonation_report = write_protonation_reports(
            protonation_result,
            json_path=reports_dir / "protonation_report.json",
            csv_path=reports_dir / "protonation_report.csv",
            markdown_path=reports_dir / "protonation_report.md",
        )
        _write_manifest_lock(
            manifest=manifest,
            structure_report=report,
            path=output_dir / "manifest.lock.yaml",
            input_manifest_path=manifest_file,
            protonation_report=protonation_report,
            refinement_report=refinement_report,
        )
        external_versions = _external_executables_from_protonation_report(
            protonation_report
        )
        external_versions.update(
            _external_executables_from_refinement_report(refinement_report)
        )
        _write_versions(
            output_dir / "versions.json",
            external_executables=external_versions,
            include_optional_python_packages=True,
        )

    if stop_after in {"ligands", "metals", "tleap", "md"}:
        assert protonation_result is not None
        ligand_result = run_ligand_stage(
            protonation_result.structure,
            manifest,
            output_dir=output_dir,
            protonation_result=protonation_result,
        )
        ligand_report = write_ligand_reports(
            ligand_result,
            json_path=reports_dir / "ligand_report.json",
            csv_path=reports_dir / "ligand_report.csv",
            markdown_path=reports_dir / "ligand_report.md",
        )
        _write_manifest_lock(
            manifest=manifest,
            structure_report=report,
            path=output_dir / "manifest.lock.yaml",
            input_manifest_path=manifest_file,
            protonation_report=protonation_report,
            refinement_report=refinement_report,
            ligand_report=ligand_report,
        )
        external_versions = {}
        if protonation_report is not None:
            external_versions.update(_external_executables_from_protonation_report(protonation_report))
        if refinement_report is not None:
            external_versions.update(
                _external_executables_from_refinement_report(refinement_report)
            )
        external_versions.update(_external_executables_from_ligand_report(ligand_report))
        _write_versions(
            output_dir / "versions.json",
            external_executables=external_versions,
            include_optional_python_packages=True,
        )

    if stop_after in {"metals", "tleap", "md"}:
        assert protonation_result is not None
        assert ligand_result is not None
        metal_result = run_metal_stage(
            protonation_result.structure,
            manifest,
            output_dir=output_dir,
            ligand_result=ligand_result,
            protonation_result=protonation_result,
            refinement_result=refinement_result,
        )
        ligand_result = promote_mcpb_resp_ligands(ligand_result, metal_result)
        ligand_report = write_ligand_reports(
            ligand_result,
            json_path=reports_dir / "ligand_report.json",
            csv_path=reports_dir / "ligand_report.csv",
            markdown_path=reports_dir / "ligand_report.md",
        )
        metal_report = write_metal_reports(
            metal_result,
            json_path=reports_dir / "metal_report.json",
            markdown_path=reports_dir / "metal_report.md",
        )
        _write_manifest_lock(
            manifest=manifest,
            structure_report=report,
            path=output_dir / "manifest.lock.yaml",
            input_manifest_path=manifest_file,
            protonation_report=protonation_report,
            refinement_report=refinement_report,
            ligand_report=ligand_report,
            metal_report=metal_report,
        )
        external_versions = {}
        if protonation_report is not None:
            external_versions.update(_external_executables_from_protonation_report(protonation_report))
        if refinement_report is not None:
            external_versions.update(
                _external_executables_from_refinement_report(refinement_report)
            )
        if ligand_report is not None:
            external_versions.update(_external_executables_from_ligand_report(ligand_report))
        external_versions.update(_external_executables_from_metal_report(metal_report))
        _write_versions(
            output_dir / "versions.json",
            external_executables=external_versions,
            include_optional_python_packages=True,
        )
        if stop_after in {"tleap", "md"} and not metal_result.complete:
            assert metal_result.mcpb_site is not None
            required = ", ".join(metal_result.mcpb_site.expected_qm_artifacts)
            raise PrepareWorkflowError(
                "Bonded MCPB.py workflow prepared QM inputs but is not complete. Run the generated "
                f"QM jobs, configure metals[].mcpb.workflow: complete and artifacts ({required}), "
                "then rerun before tleap."
            )

    if stop_after in {"tleap", "md"}:
        assert protonation_result is not None
        assert ligand_result is not None
        assert metal_result is not None
        tleap_result = run_tleap_stage(
            structure=protonation_result.structure,
            manifest=manifest,
            output_dir=output_dir,
            protonation_result=protonation_result,
            ligand_result=ligand_result,
            metal_result=metal_result,
        )
        tleap_report = write_tleap_reports(
            tleap_result,
            json_path=reports_dir / "tleap_report.json",
            markdown_path=reports_dir / "tleap_report.md",
        )
        # Keep the legacy no-metal call shape so downstream integrations that
        # wrap the validation hook do not need to understand metal metadata.
        if manifest.metals:
            validation_report = validate_final_outputs(
                manifest=manifest,
                prmtop=tleap_result.final_outputs.prmtop,
                inpcrd=tleap_result.final_outputs.inpcrd,
                pdb=tleap_result.final_outputs.pdb,
                metal_result=metal_result,
            )
        else:
            validation_report = validate_final_outputs(
                manifest=manifest,
                prmtop=tleap_result.final_outputs.prmtop,
                inpcrd=tleap_result.final_outputs.inpcrd,
                pdb=tleap_result.final_outputs.pdb,
            )
        write_validation_reports(
            validation_report,
            json_path=reports_dir / "validation_report.json",
            markdown_path=reports_dir / "validation_report.md",
        )
        _write_manifest_lock(
            manifest=manifest,
            structure_report=report,
            path=output_dir / "manifest.lock.yaml",
            input_manifest_path=manifest_file,
            protonation_report=protonation_report,
            refinement_report=refinement_report,
            ligand_report=ligand_report,
            metal_report=metal_report,
            tleap_report=tleap_report,
            validation_report=validation_report,
        )
        external_versions = {}
        if protonation_report is not None:
            external_versions.update(_external_executables_from_protonation_report(protonation_report))
        if refinement_report is not None:
            external_versions.update(
                _external_executables_from_refinement_report(refinement_report)
            )
        if ligand_report is not None:
            external_versions.update(_external_executables_from_ligand_report(ligand_report))
        if metal_report is not None:
            external_versions.update(_external_executables_from_metal_report(metal_report))
        external_versions.update(_external_executables_from_tleap_report(tleap_report))
        _write_versions(
            output_dir / "versions.json",
            external_executables=external_versions,
            include_optional_python_packages=True,
        )

    md_result = None
    if stop_after == "md":
        assert tleap_result is not None
        try:
            md_result = run_roe_brooks(
                prmtop_path=tleap_result.final_outputs.prmtop,
                inpcrd_path=tleap_result.final_outputs.inpcrd,
                output_dir=output_dir / "md" / "roe_brooks_2020",
                config=manifest.molecular_dynamics,
            )
        except RoeBrooksError as exc:
            raise PrepareWorkflowError(str(exc)) from exc

    return PrepareResult(
        stage=stop_after,
        structure_result=result,
        protonation_result=protonation_result,
        refinement_result=refinement_result,
        ligand_result=ligand_result,
        metal_result=metal_result,
        tleap_result=tleap_result,
        md_result=md_result,
    )


def _write_manifest_lock(
    *,
    manifest: object,
    structure_report: dict[str, object],
    path: Path,
    input_manifest_path: Path,
    protonation_report: dict[str, object] | None = None,
    refinement_report: dict[str, object] | None = None,
    ligand_report: dict[str, object] | None = None,
    metal_report: dict[str, object] | None = None,
    tleap_report: dict[str, object] | None = None,
    validation_report: dict[str, object] | None = None,
) -> None:
    manifest_data = manifest.model_dump(mode="json")  # type: ignore[attr-defined]
    lock_data = {
        "mdprep_version": __version__,
        "manifest": manifest_data,
        "resolved": {
            "input_manifest_path": str(input_manifest_path),
            "input_structure": manifest_data["project"]["input_structure"],
            "selected_altloc_policy": manifest_data["structure"]["altloc_policy"],
            "keep_crystal_waters": manifest_data["structure"]["keep_crystal_waters"],
            "remove_unknown_heterogens": manifest_data["structure"]["remove_unknown_heterogens"],
            "configured_ligand_selectors": [
                {"id": ligand["id"], "selector": ligand["selector"]}
                for ligand in manifest_data.get("ligands", [])
            ],
            "unknown_heterogens_removed": structure_report["unknown_heterogens_removed"],
            "unknown_heterogens_causing_failure": structure_report["unknown_heterogens_causing_failure"],
            "possible_disulfides": structure_report["possible_disulfides"],
            "structure_repair": structure_report.get("structure_repair"),
        },
    }
    if protonation_report is not None:
        lock_data["resolved"]["protonation"] = {
            "method": protonation_report["method"],
            "ph": protonation_report["ph"],
            "applied_manual_overrides": protonation_report["manual_overrides_applied"],
            "metal_coordination_assignments": protonation_report.get(
                "metal_coordination_assignments_applied", []
            ),
            "applied_disulfide_assignments": protonation_report["disulfide_assignments_applied"],
            "input_state_assignments": protonation_report["input_state_assignments_applied"],
            "propka_assignments": protonation_report["propka_assignments_applied"],
            "xtb_assignments": protonation_report["xtb_assignments_applied"],
            "hydrogens_removed": protonation_report["hydrogen_atoms_removed"],
            "final_protonation_pdb_path": protonation_report["output_protonation_pdb_path"],
        }
        if protonation_report.get("propka") is not None:
            propka = protonation_report["propka"]
            lock_data["resolved"]["protonation"]["propka"] = {
                "executable": propka["executable"],
                "command": propka["command"],
                "output_pka_path": propka["output_pka_path"],
                "parsed_pkas_path": propka["parsed_pkas_path"],
            }
        if protonation_report.get("xtb_histidines"):
            manifest_data = manifest.model_dump(mode="json")  # type: ignore[attr-defined]
            lock_data["resolved"]["protonation"]["xtb"] = {
                "executable": protonation_report["xtb_histidines"][0]["executable"],
                "model": manifest_data["protonation"]["histidine"]["xtb"]["model"],
                "mode": manifest_data["protonation"]["histidine"]["xtb"]["mode"],
                "solvent": manifest_data["protonation"]["histidine"]["xtb"]["solvent"],
                "histidine_selections": protonation_report["xtb_histidines"],
            }
    if refinement_report is not None:
        lock_data["resolved"]["refinement"] = refinement_report
    if ligand_report is not None:
        lock_data["resolved"]["ligands"] = [
            {
                "id": ligand["ligand_id"],
                "selector": ligand["selector"],
                "charge_method": ligand["charge_method"],
                "atom_types": ligand["atom_types"],
                "final_mol2_path": ligand["final_mol2_path"],
                "final_frcmod_path": ligand["final_frcmod_path"],
                "antechamber": ligand["antechamber"],
                "parmchk2": ligand["parmchk2"],
                "provisional_mol2_path": ligand.get("provisional_mol2_path"),
                "qm": ligand.get("qm"),
            }
            for ligand in ligand_report["ligands"]
        ]
        lock_data["resolved"]["qmmesp_provisional_ligands"] = ligand_report.get(
            "qmmesp_provisional_ligands", []
        )
        lock_data["resolved"]["qmmesp_provisional_system"] = ligand_report.get(
            "qmmesp_provisional_system"
        )
    if metal_report is not None:
        lock_data["resolved"]["metals"] = metal_report
    if tleap_report is not None:
        lock_data["resolved"]["tleap"] = {
            "force_fields_sourced": tleap_report["force_fields_sourced"],
            "water_box": tleap_report["water_box"],
            "ligands": tleap_report["ligands"],
            "mcpb_resp_ligands": tleap_report.get("mcpb_resp_ligands", []),
            "disulfide_bond_commands": tleap_report["disulfide_bond_commands"],
            "solvation_enabled": tleap_report["solvation_enabled"],
            "solvation_box_type": tleap_report["solvation_box_type"],
            "buffer_angstrom": tleap_report["buffer_angstrom"],
            "neutralization_requested": tleap_report["neutralization_requested"],
            "neutralizing_ions_added": tleap_report["neutralizing_ions_added"],
            "salt_concentration_molar": tleap_report["salt_concentration_molar"],
            "salt_ion_pairs_requested": tleap_report["salt_ion_pairs_requested"],
            "c4_postprocessing": tleap_report.get("c4_postprocessing", []),
            "final_outputs": tleap_report["final_outputs"],
        }
    if validation_report is not None:
        lock_data["resolved"]["validation"] = {
            "final_prmtop_path": validation_report["final_prmtop_path"],
            "final_inpcrd_path": validation_report["final_inpcrd_path"],
            "final_pdb_path": validation_report["final_pdb_path"],
            "warnings": validation_report["warnings"],
            "errors": validation_report["errors"],
        }
    path.write_text(yaml.safe_dump(lock_data, sort_keys=False), encoding="utf-8")


def _write_versions(
    path: Path,
    *,
    external_executables: dict[str, str] | None = None,
    include_optional_python_packages: bool = False,
) -> None:
    versions = {
        "mdprep": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "external_tools_required": [],
        "external_versions": {
            name: _external_version(executable)
            for name, executable in (external_executables or {}).items()
        },
    }
    if include_optional_python_packages:
        versions["python_packages"] = {
            "parmed": parmed_version(),
            "openmm": openmm_version(),
            "pyscf": pyscf_version(),
        }
    path.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _external_executables_from_protonation_report(
    protonation_report: dict[str, object],
) -> dict[str, str]:
    executables: dict[str, str] = {}
    propka = protonation_report.get("propka")
    if isinstance(propka, dict) and isinstance(propka.get("executable"), str):
        executables["propka"] = propka["executable"]
    xtb_histidines = protonation_report.get("xtb_histidines")
    if isinstance(xtb_histidines, list) and xtb_histidines:
        first = xtb_histidines[0]
        if isinstance(first, dict) and isinstance(first.get("executable"), str):
            executables["xtb"] = first["executable"]
    return executables


def _external_executables_from_refinement_report(
    refinement_report: dict[str, object],
) -> dict[str, str]:
    executables: dict[str, str] = {}
    ash = refinement_report.get("ash")
    if isinstance(ash, dict):
        python_executable = ash.get("python_executable")
        xtb_executable = ash.get("xtb_executable")
        if isinstance(python_executable, str):
            executables["ash_python"] = python_executable
        if isinstance(xtb_executable, str):
            executables["xtb"] = xtb_executable
    provisional_ligands = refinement_report.get("provisional_ligands")
    if isinstance(provisional_ligands, list):
        ligand_shape = {
            "ligands": provisional_ligands,
            "qmmesp_provisional_ligands": [],
            "qmmesp_provisional_system": refinement_report.get("provisional_system"),
        }
        executables.update(_external_executables_from_ligand_report(ligand_shape))
    return executables


def _external_executables_from_ligand_report(
    ligand_report: dict[str, object],
) -> dict[str, str]:
    executables: dict[str, str] = {}
    ligand_groups = [ligand_report.get("ligands"), ligand_report.get("qmmesp_provisional_ligands")]
    for ligands in ligand_groups:
        if not isinstance(ligands, list):
            continue
        for ligand in ligands:
            if not isinstance(ligand, dict):
                continue
            antechamber = ligand.get("antechamber")
            if isinstance(antechamber, dict) and isinstance(antechamber.get("command"), list):
                command = antechamber["command"]
                if command:
                    executables["antechamber"] = str(command[0])
            parmchk2 = ligand.get("parmchk2")
            if isinstance(parmchk2, dict) and isinstance(parmchk2.get("command"), list):
                command = parmchk2["command"]
                if command:
                    executables["parmchk2"] = str(command[0])
            qm = ligand.get("qm")
            fit_result = qm.get("fit_result") if isinstance(qm, dict) else None
            commands = fit_result.get("commands") if isinstance(fit_result, dict) else None
            if isinstance(commands, list):
                for record in commands:
                    if not isinstance(record, dict) or not isinstance(record.get("command"), list):
                        continue
                    command = record["command"]
                    if command:
                        executables[Path(str(command[0])).name] = str(command[0])
    provisional_system = ligand_report.get("qmmesp_provisional_system")
    if isinstance(provisional_system, dict):
        tleap = provisional_system.get("tleap")
        if isinstance(tleap, dict) and isinstance(tleap.get("command"), list):
            command = tleap["command"]
            if command:
                executables["tleap"] = str(command[0])
    return executables


def _external_executables_from_tleap_report(
    tleap_report: dict[str, object],
) -> dict[str, str]:
    executables: dict[str, str] = {}
    dry = tleap_report.get("dry_tleap")
    if isinstance(dry, dict) and isinstance(dry.get("command"), list):
        command = dry["command"]
        if command:
            executables["tleap"] = str(command[0])
    c4_runs = tleap_report.get("c4_postprocessing")
    if isinstance(c4_runs, list):
        for run in c4_runs:
            if not isinstance(run, dict) or not isinstance(run.get("command"), list):
                continue
            command = run["command"]
            if command:
                executables["parmed"] = str(command[0])
    return executables


def _external_executables_from_metal_report(
    metal_report: dict[str, object],
) -> dict[str, str]:
    executables: dict[str, str] = {}
    bonded = metal_report.get("bonded_mcpb_site")
    if isinstance(bonded, dict):
        runs = bonded.get("runs")
        if isinstance(runs, list):
            for run in runs:
                if not isinstance(run, dict) or not isinstance(run.get("command"), list):
                    continue
                command = run["command"]
                if command:
                    executables["MCPB.py"] = str(command[0])
        qm_result = bonded.get("pyscf")
        if isinstance(qm_result, dict):
            backend = qm_result.get("small_model_hessian_backend")
            small_model = qm_result.get("small_model")
            if isinstance(small_model, dict):
                external_commands = small_model.get("external_commands")
                if isinstance(external_commands, list):
                    for record in external_commands:
                        if not isinstance(record, dict):
                            continue
                        command = record.get("command")
                        if not isinstance(command, list) or not command:
                            continue
                        if backend in {"gfn2_xtb", "gxtb"}:
                            executables[str(backend)] = str(command[0])
    return executables


def _external_version(executable: str) -> dict[str, object]:
    try:
        result = run_command([executable, "--version"], timeout=5.0)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {"executable": executable, "version": "unknown"}
    text = (result.stdout or result.stderr).strip()
    version = _version_line(text)
    return {
        "executable": executable,
        "version": version,
        "returncode": result.returncode,
    }


def _version_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if "version" in line.lower():
            return line
    return lines[0] if lines else "unknown"
