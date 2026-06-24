"""Build and run tleap scripts for final Amber outputs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
import os
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.leap.forcefields import ForceFieldError, ForceFieldSources, forcefield_sources
from mdprep.leap.ions import IonPlan, IonPlanError, amber_inpcrd_box_volume, build_ion_plan
from mdprep.leap.log_parser import LeapLogError, assert_tleap_success
from mdprep.leap.residues import (
    DisulfideBondCommand,
    LeapInputResult,
    LeapResidueError,
    LigandParameterFiles,
    append_disulfide_conect_records,
    disulfide_bond_commands,
    prepare_leap_input_pdb,
    validate_ligand_parameter_files,
    validate_tleap_ligand_coordinates,
)
from mdprep.leap.runner import TLeapRun, TLeapRunError, run_tleap
from mdprep.ligands.workflow import LigandStageResult
from mdprep.protonation.apply import ProtonationResult
from mdprep.structure.models import PdbStructure


class TLeapBuildError(ValueError):
    """Raised when final tleap system generation cannot complete safely."""


@dataclass(frozen=True)
class TLeapOutputs:
    prmtop: Path
    inpcrd: Path
    pdb: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "prmtop": str(self.prmtop),
            "inpcrd": str(self.inpcrd),
            "pdb": str(self.pdb),
        }


@dataclass(frozen=True)
class TLeapStageResult:
    forcefields: ForceFieldSources
    leap_input: LeapInputResult
    ligands: list[LigandParameterFiles]
    disulfide_bonds: list[DisulfideBondCommand]
    dry_run: TLeapRun
    dry_outputs: TLeapOutputs
    final_run: TLeapRun | None
    final_outputs: TLeapOutputs
    solvation_enabled: bool
    solvation_box_type: str
    buffer_angstrom: float
    neutralization_requested: bool
    salt_concentration_molar: float
    ion_plan: IonPlan | None
    salt_volume_a3: float | None
    warnings: list[str]

    def to_report_dict(self) -> dict[str, object]:
        return {
            "force_fields_sourced": self.forcefields.all_sources,
            "force_field_warnings": self.forcefields.warnings,
            "water_box": self.forcefields.water_box,
            "ligands": [ligand.to_dict() for ligand in self.ligands],
            "disulfide_bond_commands": [bond.to_dict() for bond in self.disulfide_bonds],
            "leap_input": self.leap_input.to_dict(),
            "dry_tleap": self.dry_run.to_dict(),
            "dry_outputs": self.dry_outputs.to_dict(),
            "dry_system_charge": self.dry_run.summary.total_charge,
            "solvation_enabled": self.solvation_enabled,
            "solvation_box_type": self.solvation_box_type,
            "buffer_angstrom": self.buffer_angstrom,
            "neutralization_requested": self.neutralization_requested,
            "neutralizing_ions_added": (
                self.ion_plan.neutralizing_count if self.ion_plan is not None else 0
            ),
            "salt_concentration_molar": self.salt_concentration_molar,
            "salt_ion_pairs_requested": self.ion_plan.salt_pairs if self.ion_plan is not None else 0,
            "salt_volume_a3": self.salt_volume_a3,
            "final_tleap": self.final_run.to_dict() if self.final_run is not None else None,
            "final_outputs": self.final_outputs.to_dict(),
            "warnings": self.warnings,
        }


def run_tleap_stage(
    *,
    structure: PdbStructure,
    manifest: ManifestConfig,
    output_dir: str | Path,
    protonation_result: ProtonationResult,
    ligand_result: LigandStageResult,
) -> TLeapStageResult:
    output = Path(output_dir)
    input_dir = output / "leap" / "input"
    dry_dir = output / "leap" / "dry"
    final_dir = output / "final"
    input_dir.mkdir(parents=True, exist_ok=True)
    dry_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    try:
        sources = forcefield_sources(
            protein_forcefield=manifest.protein.forcefield,
            water_model=manifest.protein.water_model,
            ligands=manifest.ligands,
        )
        leap_input = prepare_leap_input_pdb(
            structure,
            input_dir / "system.leap_input.pdb",
            manifest=manifest,
            ligand_result=ligand_result,
        )
        ligand_files = validate_ligand_parameter_files(
            manifest=manifest,
            structure=leap_input.structure,
            ligand_result=ligand_result,
        )
        disulfide_bonds = disulfide_bond_commands(
            structure=leap_input.structure,
            protonation_result=protonation_result,
        )
        conect_records = append_disulfide_conect_records(leap_input.path, disulfide_bonds)
        dry_outputs = TLeapOutputs(
            prmtop=dry_dir / "system.dry.prmtop",
            inpcrd=dry_dir / "system.dry.inpcrd",
            pdb=dry_dir / "system.dry.pdb",
        )
        dry_script = build_tleap_script(
            sources=sources,
            ligands=ligand_files,
            input_pdb=leap_input.path,
            disulfide_bonds=disulfide_bonds,
            outputs=dry_outputs,
            work_dir=dry_dir,
        )
        dry_input = dry_dir / "tleap.in"
        dry_input.write_text(dry_script, encoding="utf-8")
        dry_run = run_tleap(dry_input, work_dir=dry_dir)
        assert_tleap_success(
            dry_run.summary,
            fail_on_warnings=manifest.validation.fail_on_warnings,
            context="dry",
        )
        _ensure_outputs(dry_outputs)
        dry_coordinate_checks = validate_tleap_ligand_coordinates(
            manifest=manifest,
            reference_structure=leap_input.structure,
            output_pdb=dry_outputs.pdb,
            stage="dry",
        )

        warnings = list(sources.warnings)
        if conect_records:
            warnings.append(
                "Disulfide SG-SG bonds were encoded as PDB CONECT records in the tleap input PDB."
            )
        warnings.extend(leap_input.structure.warnings)
        warnings.extend(warning for warning in dry_run.summary.warnings)
        warnings.extend(
            f"Ligand {check['ligand_id']} dry tleap coordinate max deviation: "
            f"{check['max_coordinate_deviation_angstrom']:.3f} A"
            for check in dry_coordinate_checks
        )
        ion_plan: IonPlan | None = None
        salt_volume_a3: float | None = None
        final_run: TLeapRun | None = None
        if not manifest.solvation.enabled:
            final_outputs = _copy_outputs(dry_outputs, final_dir)
        else:
            final_outputs, final_run, ion_plan, salt_volume_a3 = _run_solvated_build(
                manifest=manifest,
                sources=sources,
                ligands=ligand_files,
                input_pdb=leap_input.path,
                disulfide_bonds=disulfide_bonds,
                dry_charge=dry_run.summary.total_charge,
                output_dir=output,
            )
            warnings.extend(final_run.summary.warnings)

    except (
        ForceFieldError,
        IonPlanError,
        LeapLogError,
        LeapResidueError,
        TLeapRunError,
        FileNotFoundError,
    ) as exc:
        raise TLeapBuildError(str(exc)) from exc

    return TLeapStageResult(
        forcefields=sources,
        leap_input=leap_input,
        ligands=ligand_files,
        disulfide_bonds=disulfide_bonds,
        dry_run=dry_run,
        dry_outputs=dry_outputs,
        final_run=final_run,
        final_outputs=final_outputs,
        solvation_enabled=manifest.solvation.enabled,
        solvation_box_type=manifest.solvation.box,
        buffer_angstrom=manifest.solvation.buffer_angstrom,
        neutralization_requested=manifest.solvation.neutralize,
        salt_concentration_molar=manifest.solvation.salt_concentration_molar,
        ion_plan=ion_plan,
        salt_volume_a3=salt_volume_a3,
        warnings=warnings,
    )


def build_tleap_script(
    *,
    sources: ForceFieldSources,
    ligands: list[LigandParameterFiles],
    input_pdb: str | Path,
    disulfide_bonds: list[DisulfideBondCommand],
    outputs: TLeapOutputs,
    work_dir: str | Path | None = None,
    solvation_command: str | None = None,
    ion_commands: list[str] | None = None,
) -> str:
    lines: list[str] = []
    for source in sources.all_sources:
        lines.append(f"source {source}")
    for ligand in ligands:
        mol2_path = _tleap_path(ligand.final_mol2_path, work_dir)
        frcmod_path = _tleap_path(ligand.final_frcmod_path, work_dir)
        lines.append(f"{ligand.variable_name} = loadmol2 {mol2_path}")
        if ligand.variable_name != ligand.residue_name:
            lines.append(f"{ligand.residue_name} = {ligand.variable_name}")
        lines.append(f"loadamberparams {frcmod_path}")
    lines.append(f"system = loadpdb {_tleap_path(input_pdb, work_dir)}")
    lines.extend(bond.command for bond in disulfide_bonds if bond.command.startswith("bond "))
    if solvation_command is not None:
        lines.append(solvation_command)
    lines.extend(ion_commands or [])
    lines.extend(
        [
            "check system",
            "charge system",
            f"savepdb system {outputs.pdb.name}",
            f"saveamberparm system {outputs.prmtop.name} {outputs.inpcrd.name}",
            "quit",
            "",
        ]
    )
    return "\n".join(lines)


def _tleap_path(path: str | Path, work_dir: str | Path | None) -> str:
    """Return a path that tleap can resolve from its execution directory."""
    path_obj = Path(path)
    if work_dir is None:
        return str(path_obj)
    base = Path(work_dir)
    base_abs = base if base.is_absolute() else Path.cwd() / base
    target_abs = path_obj if path_obj.is_absolute() else Path.cwd() / path_obj
    return os.path.relpath(target_abs.resolve(), base_abs.resolve())


def solvation_command(*, box: str, water_box: str, buffer_angstrom: float) -> str:
    if box == "truncated_octahedron":
        return f"solvateOct system {water_box} {buffer_angstrom:.3f}"
    if box == "rectangular":
        return f"solvateBox system {water_box} {buffer_angstrom:.3f}"
    raise TLeapBuildError(f"Unsupported solvation box: {box}")


def _run_solvated_build(
    *,
    manifest: ManifestConfig,
    sources: ForceFieldSources,
    ligands: list[LigandParameterFiles],
    input_pdb: Path,
    disulfide_bonds: list[DisulfideBondCommand],
    dry_charge: float | None,
    output_dir: Path,
) -> tuple[TLeapOutputs, TLeapRun, IonPlan | None, float | None]:
    if dry_charge is None and manifest.solvation.neutralize:
        raise TLeapBuildError("tleap dry-system charge was not parsed; cannot neutralize safely.")
    solv_dir = output_dir / "leap" / "solvated"
    solv_dir.mkdir(parents=True, exist_ok=True)
    final_outputs = TLeapOutputs(
        prmtop=solv_dir / "system.solvated.prmtop",
        inpcrd=solv_dir / "system.solvated.inpcrd",
        pdb=solv_dir / "system.solvated.pdb",
    )
    solvate = solvation_command(
        box=manifest.solvation.box,
        water_box=sources.water_box,
        buffer_angstrom=manifest.solvation.buffer_angstrom,
    )

    volume_a3: float | None = None
    if manifest.solvation.salt_concentration_molar > 0:
        pre_outputs = TLeapOutputs(
            prmtop=solv_dir / "system.presalt.prmtop",
            inpcrd=solv_dir / "system.presalt.inpcrd",
            pdb=solv_dir / "system.presalt.pdb",
        )
        pre_input = solv_dir / "tleap.presalt.in"
        pre_input.write_text(
            build_tleap_script(
                sources=sources,
                ligands=ligands,
                input_pdb=input_pdb,
                disulfide_bonds=disulfide_bonds,
                outputs=pre_outputs,
                work_dir=solv_dir,
                solvation_command=solvate,
            ),
            encoding="utf-8",
        )
        pre_run = run_tleap(pre_input, work_dir=solv_dir)
        assert_tleap_success(
            pre_run.summary,
            fail_on_warnings=manifest.validation.fail_on_warnings,
            context="pre-salt solvated",
        )
        _ensure_outputs(pre_outputs)
        volume_a3 = amber_inpcrd_box_volume(str(pre_outputs.inpcrd))
        if volume_a3 is None:
            raise TLeapBuildError(
                "Could not determine solvated box volume; cannot add configured salt concentration."
            )

    ion_plan = build_ion_plan(
        total_charge=dry_charge or 0.0,
        neutralize=manifest.solvation.neutralize,
        positive_ion=manifest.solvation.positive_ion,
        negative_ion=manifest.solvation.negative_ion,
        salt_concentration_molar=manifest.solvation.salt_concentration_molar,
        volume_a3=volume_a3,
    )
    script = build_tleap_script(
        sources=sources,
        ligands=ligands,
        input_pdb=input_pdb,
        disulfide_bonds=disulfide_bonds,
        outputs=final_outputs,
        work_dir=solv_dir,
        solvation_command=solvate,
        ion_commands=ion_plan.commands,
    )
    script_path = solv_dir / "tleap.in"
    script_path.write_text(script, encoding="utf-8")
    final_run = run_tleap(script_path, work_dir=solv_dir)
    assert_tleap_success(
        final_run.summary,
        fail_on_warnings=manifest.validation.fail_on_warnings,
        context="solvated",
    )
    _ensure_outputs(final_outputs)
    copied = _copy_outputs(final_outputs, output_dir / "final")
    return copied, final_run, ion_plan, volume_a3


def _copy_outputs(outputs: TLeapOutputs, final_dir: Path) -> TLeapOutputs:
    final_dir.mkdir(parents=True, exist_ok=True)
    copied = TLeapOutputs(
        prmtop=final_dir / "system.prmtop",
        inpcrd=final_dir / "system.inpcrd",
        pdb=final_dir / "system.pdb",
    )
    shutil.copyfile(outputs.prmtop, copied.prmtop)
    shutil.copyfile(outputs.inpcrd, copied.inpcrd)
    shutil.copyfile(outputs.pdb, copied.pdb)
    return copied


def _ensure_outputs(outputs: TLeapOutputs) -> None:
    for path in [outputs.prmtop, outputs.inpcrd, outputs.pdb]:
        if not path.exists():
            raise FileNotFoundError(f"tleap did not produce expected output: {path}")
        if path.stat().st_size == 0:
            raise TLeapBuildError(f"tleap produced an empty output file: {path}")
