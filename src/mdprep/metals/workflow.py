"""Metal preparation stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from mdprep.config.models import ManifestConfig, NonbondedMetalConfig
from mdprep.ligands.workflow import LigandStageResult
from mdprep.metals.coordination import ResolvedMetalSite, resolve_metal_sites
from mdprep.metals.hydrogenate import MetalHydrogenationResult, hydrogenate_for_mcpb
from mdprep.metals.mcpb import McpbSiteResult, run_mcpb_site
from mdprep.metals.nonbonded import (
    NonbondedIonArtifact,
    NonbondedSiteResult,
    prepare_nonbonded_site,
)
from mdprep.protonation.apply import ProtonationResult
from mdprep.structure.models import PdbStructure
from mdprep.structure.pdb import read_pdb

if TYPE_CHECKING:
    from mdprep.refinement.workflow import RefinementResult


class MetalWorkflowError(ValueError):
    """Raised when metal preparation is incomplete or internally inconsistent."""


@dataclass
class MetalStageResult:
    sites: list[ResolvedMetalSite]
    nonbonded_sites: list[NonbondedSiteResult] = field(default_factory=list)
    mcpb_site: McpbSiteResult | None = None
    pre_mcpb_hydrogenation: MetalHydrogenationResult | None = None
    structure: PdbStructure | None = None
    complete: bool = True
    leap_setup_commands: list[str] = field(default_factory=list)
    leap_bond_commands: list[str] = field(default_factory=list)
    c4_atom_types: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_report_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "sites": [site.to_dict() for site in self.sites],
            "nonbonded_sites": [site.to_dict() for site in self.nonbonded_sites],
            "bonded_mcpb_site": self.mcpb_site.to_dict() if self.mcpb_site else None,
            "pre_mcpb_hydrogenation": (
                self.pre_mcpb_hydrogenation.to_dict()
                if self.pre_mcpb_hydrogenation
                else None
            ),
            "leap_setup_commands": self.leap_setup_commands,
            "leap_bond_commands": self.leap_bond_commands,
            "c4_atom_types": self.c4_atom_types,
            "final_metal_stage_pdb": str(self.structure.path) if self.structure else None,
            "warnings": self.warnings,
        }


def run_metal_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
    ligand_result: LigandStageResult,
    protonation_result: ProtonationResult,
    refinement_result: "RefinementResult | None" = None,
) -> MetalStageResult:
    sites = resolve_metal_sites(structure, manifest, validate_mcpb_cutoff=True)
    if not sites:
        return MetalStageResult(sites=[], structure=structure)
    output = Path(output_dir) / "metals"
    output.mkdir(parents=True, exist_ok=True)
    nonbonded_results: list[NonbondedSiteResult] = []
    mcpb_result: McpbSiteResult | None = None
    hydrogenation_result: MetalHydrogenationResult | None = None
    setup_commands: list[str] = []
    bond_commands: list[str] = []
    c4_types: list[str] = []
    warnings: list[str] = []
    final_structure = structure
    parameter_by_type: dict[str, tuple[str, str]] = {}
    provisional_parameter_by_type: dict[str, tuple[str, str]] = {}
    provisional_template_by_resname: dict[str, tuple[str, int, str]] = {}

    provisional_setup_commands: list[str] = []
    bonded_site: ResolvedMetalSite | None = None
    for site in sites:
        site_dir = output / _safe_dir_name(site.config.id)
        if site.config.model == "nonbonded":
            result = prepare_nonbonded_site(
                site,
                water_model=manifest.protein.water_model,
                output_dir=site_dir,
            )
            for artifact in result.ions:
                _register_ion_artifact(
                    artifact,
                    parameter_by_type=parameter_by_type,
                    template_by_resname=None,
                    context="final topology",
                )
                _register_ion_artifact(
                    artifact,
                    parameter_by_type=provisional_parameter_by_type,
                    template_by_resname=provisional_template_by_resname,
                    context="pre-MCPB provisional topology",
                )
                if artifact.parameter.parameter_set == "12_6_4":
                    c4_types.append(artifact.parameter.atom_type)
            nonbonded_results.append(result)
            setup_commands.extend(result.leap_commands)
            provisional_setup_commands.extend(result.leap_commands)
        else:
            if bonded_site is not None:
                raise MetalWorkflowError("Only one bonded MCPB.py site may be prepared per system.")
            bonded_site = site
            assert site.config.mcpb is not None
            family = site.config.mcpb.provisional_nonbonded_parameter_set
            if family is None:
                raise MetalWorkflowError(
                    f"Bonded site {site.config.id!r} has no provisional nonbonded ion model."
                )
            provisional_config = site.config.model_copy(
                update={
                    "model": "nonbonded",
                    "nonbonded": NonbondedMetalConfig(parameter_set=family),
                    "mcpb": None,
                }
            )
            provisional_site = replace(site, config=provisional_config)
            provisional = prepare_nonbonded_site(
                provisional_site,
                water_model=manifest.protein.water_model,
                output_dir=site_dir / "provisional_nonbonded",
            )
            for artifact in provisional.ions:
                _register_ion_artifact(
                    artifact,
                    parameter_by_type=provisional_parameter_by_type,
                    template_by_resname=provisional_template_by_resname,
                    context="pre-MCPB provisional topology",
                )
            provisional_setup_commands.extend(provisional.leap_commands)

    if bonded_site is not None:
        site_dir = output / _safe_dir_name(bonded_site.config.id)
        hydrogenation_result = hydrogenate_for_mcpb(
            structure,
            manifest,
            ligand_result=ligand_result,
            protonation_result=protonation_result,
            metal_setup_commands=_deduplicate_commands(provisional_setup_commands),
            output_dir=site_dir / "hydrogenation",
        )
        hydrated_sites = resolve_metal_sites(
            hydrogenation_result.structure,
            manifest,
            validate_mcpb_cutoff=True,
        )
        hydrated_bonded = next(
            (site for site in hydrated_sites if site.config.id == bonded_site.config.id),
            None,
        )
        if hydrated_bonded is None:
            raise MetalWorkflowError(
                f"Bonded site {bonded_site.config.id!r} could not be resolved after hydrogenation."
            )
        mcpb_result = run_mcpb_site(
            hydrated_bonded,
            structure=hydrogenation_result.structure,
            manifest=manifest,
            ligand_result=ligand_result,
            output_dir=site_dir / "mcpb",
            provisional_prmtop=hydrogenation_result.prmtop_path,
            provisional_inpcrd=hydrogenation_result.inpcrd_path,
            refinement_result=refinement_result,
        )
        if mcpb_result.complete:
            assert mcpb_result.restored_final_pdb_path is not None
            final_structure = read_pdb(mcpb_result.restored_final_pdb_path)
            setup_commands.extend(mcpb_result.leap_setup_commands)
            bond_commands.extend(mcpb_result.leap_bond_commands)
        if (
            mcpb_result.pyscf_result is not None
            and mcpb_result.pyscf_result.small.scientific_warning is not None
        ):
            warnings.append(mcpb_result.pyscf_result.small.scientific_warning)

    complete = mcpb_result is None or mcpb_result.complete
    return MetalStageResult(
        sites=sites,
        nonbonded_sites=nonbonded_results,
        mcpb_site=mcpb_result,
        pre_mcpb_hydrogenation=hydrogenation_result,
        structure=final_structure,
        complete=complete,
        leap_setup_commands=_deduplicate_commands(setup_commands),
        leap_bond_commands=_deduplicate_commands(bond_commands),
        c4_atom_types=sorted(set(c4_types)),
        warnings=warnings,
    )


def promote_mcpb_resp_ligands(
    ligand_result: LigandStageResult,
    metal_result: MetalStageResult,
) -> LigandStageResult:
    """Replace provisional cofactor paths with MCPB's joint-RESP artifacts."""

    site = metal_result.mcpb_site
    if site is None or not site.complete:
        return ligand_result
    artifacts = {item.ligand_id: item for item in site.refitted_ligands}
    promoted = []
    for item in ligand_result.ligands:
        artifact = artifacts.get(item.ligand_id)
        if artifact is None:
            promoted.append(item)
            continue
        if item.charge_method != "mcpb_resp_pyscf":
            raise MetalWorkflowError(
                f"MCPB returned a refitted mol2 for ligand {item.ligand_id!r}, "
                f"but its configured charge method is {item.charge_method!r}."
            )
        if site.final_frcmod_path is None:
            raise MetalWorkflowError(
                f"MCPB RESP ligand {item.ligand_id!r} has no final shared frcmod."
            )
        promoted.append(
            replace(
                item,
                final_mol2_path=artifact.final_mol2_path,
                final_frcmod_path=site.final_frcmod_path,
                validation=None,
                warnings=[
                    *item.warnings,
                    "Final mol2 charges come from the joint embedded PySCF/MCPB RESP "
                    "fit; the recorded AM1-BCC mol2 remains provisional only.",
                ],
                status="ok",
            )
        )
    expected = {
        item.ligand_id
        for item in ligand_result.ligands
        if item.charge_method == "mcpb_resp_pyscf"
    }
    if set(artifacts) != expected:
        raise MetalWorkflowError(
            "MCPB RESP ligand promotion is incomplete: expected "
            f"{sorted(expected)}, obtained {sorted(artifacts)}."
        )
    return replace(ligand_result, ligands=promoted)


def _safe_dir_name(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return safe or "metal_site"


def _deduplicate_commands(commands: list[str]) -> list[str]:
    """Remove identical commands while preserving multiline block members."""

    result: list[str] = []
    seen: set[str] = set()
    seen_blocks: set[tuple[str, ...]] = set()
    index = 0
    while index < len(commands):
        command = commands[index]
        if command == "addAtomTypes {":
            end = index + 1
            while end < len(commands) and commands[end] != "}":
                end += 1
            if end >= len(commands):
                raise MetalWorkflowError("Unterminated addAtomTypes block in generated tleap setup.")
            block = tuple(commands[index : end + 1])
            if block not in seen_blocks:
                seen_blocks.add(block)
                result.extend(block)
            index = end + 1
            continue
        if command not in seen:
            seen.add(command)
            result.append(command)
        index += 1
    return result


def _register_ion_artifact(
    artifact: NonbondedIonArtifact,
    *,
    parameter_by_type: dict[str, tuple[str, str]],
    template_by_resname: dict[str, tuple[str, int, str]] | None,
    context: str,
) -> None:
    parameter = artifact.parameter
    ion = artifact.ion
    signature = (parameter.parameter_set, parameter.frcmod_name)
    previous = parameter_by_type.get(parameter.atom_type)
    if previous is not None and previous != signature:
        raise MetalWorkflowError(
            f"Amber atom type {parameter.atom_type} is requested with conflicting ion models "
            f"{previous} and {signature} in the {context}. A single topology cannot contain both."
        )
    parameter_by_type[parameter.atom_type] = signature
    if template_by_resname is None:
        return
    residue_name = ion.residue.id.resname
    template = (ion.element, ion.charge, ion.atom.name)
    prior_template = template_by_resname.get(residue_name)
    if prior_template is not None and prior_template != template:
        raise MetalWorkflowError(
            f"Metal residue name {residue_name!r} maps to conflicting templates "
            f"{prior_template} and {template} in the {context}. Use distinct PDB residue names."
        )
    template_by_resname[residue_name] = template
