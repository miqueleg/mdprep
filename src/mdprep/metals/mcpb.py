"""Preserved, staged MCPB.py workflow for bonded metal centers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path
import os
import re
import shutil
import sys
from typing import TYPE_CHECKING

from mdprep.ambertools.mol2 import Mol2Error, read_mol2
from mdprep.config.models import ManifestConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.metals.coordination import ResolvedMetalSite
from mdprep.metals.nonbonded import (
    AmberIonParameter,
    NonbondedMetalError,
    resolve_amberhome,
    write_single_ion_mol2,
)
from mdprep.structure.classify import is_standard_protein_residue, is_water_residue
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb

if TYPE_CHECKING:
    from mdprep.ligands.workflow import LigandStageResult
    from mdprep.metals.parameter_comparison import McpbParameterComparisonResult
    from mdprep.metals.pyscf_mcpb import McpbPySCFResult
    from mdprep.refinement.workflow import RefinementResult


class McpbWorkflowError(ValueError):
    """Raised when a bonded MCPB.py workflow cannot be completed safely."""


@dataclass(frozen=True)
class McpbRun:
    step: str
    result: CommandResult
    stdout_path: Path
    stderr_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "command": list(self.result.command),
            "cwd": self.result.cwd,
            "returncode": self.result.returncode,
            "stdout": self.result.stdout,
            "stderr": self.result.stderr,
            "runtime_seconds": self.result.runtime_seconds,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
        }


@dataclass(frozen=True)
class McpbNumberingMap:
    original_by_mcpb_atom_id: dict[int, AtomRecord]
    original_by_mcpb_resid: dict[int, ResidueRecord]
    mcpb_atom_id_by_original_identity: dict[tuple[str, str, int, str | None, str], int]
    mcpb_resid_by_original_key: dict[tuple[str, int, str | None], int]


@dataclass(frozen=True)
class McpbRefittedLigand:
    """One non-protein residue whose final charges come from joint MCPB RESP."""

    ligand_id: str
    original_resname: str
    final_resname: str
    final_mol2_path: Path
    provisional_mol2_path: Path
    atom_names: tuple[str, ...]
    atom_count: int
    fragment_charge: float
    maximum_coordinate_deviation_angstrom: float

    def to_dict(self) -> dict[str, object]:
        return {
            "ligand_id": self.ligand_id,
            "charge_source": "joint_mcpb_resp_pyscf",
            "original_resname": self.original_resname,
            "final_resname": self.final_resname,
            "final_mol2_path": str(self.final_mol2_path),
            "provisional_mol2_path": str(self.provisional_mol2_path),
            "atom_names": list(self.atom_names),
            "atom_count": self.atom_count,
            "fragment_charge": self.fragment_charge,
            "maximum_coordinate_deviation_angstrom": (
                self.maximum_coordinate_deviation_angstrom
            ),
            "fragment_charge_note": (
                "This is a fragment charge from one joint metal-cluster RESP fit; "
                "it is not independently constrained to the isolated formal charge."
            ),
        }


@dataclass(frozen=True)
class McpbSiteResult:
    site_id: str
    workflow: str
    complete: bool
    work_dir: Path
    input_path: Path
    mcpb_input_pdb_path: Path
    restored_final_pdb_path: Path | None
    final_frcmod_path: Path | None
    generated_tleap_path: Path | None
    leap_setup_commands: tuple[str, ...]
    leap_bond_commands: tuple[str, ...]
    residue_renames: tuple[dict[str, object], ...]
    runs: tuple[McpbRun, ...]
    expected_qm_artifacts: tuple[str, ...]
    generated_files: tuple[Path, ...]
    amberhome: Path
    refitted_ligands: tuple[McpbRefittedLigand, ...] = ()
    pyscf_result: "McpbPySCFResult | None" = None
    parameter_comparison: "McpbParameterComparisonResult | None" = None

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "model": "bonded_mcpb",
            "workflow": self.workflow,
            "complete": self.complete,
            "work_dir": str(self.work_dir),
            "input_path": str(self.input_path),
            "mcpb_input_pdb_path": str(self.mcpb_input_pdb_path),
            "restored_final_pdb_path": (
                str(self.restored_final_pdb_path) if self.restored_final_pdb_path else None
            ),
            "final_frcmod_path": str(self.final_frcmod_path) if self.final_frcmod_path else None,
            "generated_tleap_path": (
                str(self.generated_tleap_path) if self.generated_tleap_path else None
            ),
            "leap_setup_commands": list(self.leap_setup_commands),
            "leap_bond_commands": list(self.leap_bond_commands),
            "residue_renames": list(self.residue_renames),
            "runs": [run.to_dict() for run in self.runs],
            "expected_qm_artifacts": list(self.expected_qm_artifacts),
            "generated_files": [str(path) for path in self.generated_files],
            "amberhome": str(self.amberhome),
            "refitted_ligands": [item.to_dict() for item in self.refitted_ligands],
            "pyscf": self.pyscf_result.to_dict() if self.pyscf_result else None,
            "parameter_comparison": (
                self.parameter_comparison.to_dict()
                if self.parameter_comparison is not None
                else None
            ),
        }


def run_mcpb_site(
    site: ResolvedMetalSite,
    *,
    structure: PdbStructure,
    manifest: ManifestConfig,
    ligand_result: "LigandStageResult",
    output_dir: str | Path,
    provisional_prmtop: str | Path | None = None,
    provisional_inpcrd: str | Path | None = None,
    refinement_result: "RefinementResult | None" = None,
) -> McpbSiteResult:
    if site.config.model != "bonded_mcpb" or site.config.mcpb is None:
        raise McpbWorkflowError(f"Metal site {site.config.id!r} is not configured for MCPB.py.")
    mcpb = site.config.mcpb
    executable = which_executable(mcpb.executable)
    if executable is None:
        raise McpbWorkflowError(f"AmberTools executable not found: {mcpb.executable}")
    try:
        amberhome = resolve_amberhome(Path(executable).resolve().parent.parent)
    except NonbondedMetalError as exc:
        raise McpbWorkflowError(str(exc)) from exc
    mcpb_environment = dict(os.environ)
    mcpb_environment["AMBERHOME"] = str(amberhome)
    _validate_mcpb_structure_scope(structure)
    _validate_nonstandard_site_residues(site, manifest)

    work = Path(output_dir)
    work.mkdir(parents=True, exist_ok=True)
    group_name = _safe_group_name(site.config.id)
    input_pdb = work / "system.mcpb_input.pdb"
    numbering = write_mcpb_numbered_pdb(structure, input_pdb)
    ligand_mol2, ligand_frcmods, gaff = _stage_ligand_parameters(
        manifest=manifest,
        ligand_result=ligand_result,
        work_dir=work,
    )
    ion_mol2 = _write_mcpb_ion_mol2(site, numbering=numbering, work_dir=work)
    input_path = work / "mcpb.in"
    input_path.write_text(
        build_mcpb_input(
            site,
            manifest=manifest,
            numbering=numbering,
            original_pdb=input_pdb,
            ion_mol2=ion_mol2,
            ligand_mol2=ligand_mol2,
            ligand_frcmods=ligand_frcmods,
            gaff=gaff,
            group_name=group_name,
        ),
        encoding="utf-8",
    )

    runs: list[McpbRun] = []
    runs.append(
        _run_mcpb(
            executable,
            input_path,
            "1",
            work,
            environment=mcpb_environment,
        )
    )
    for required_input in _required_qm_inputs(
        work,
        group_name=group_name,
        software_version=mcpb.software_version,
    ):
        if not required_input.is_file() or required_input.stat().st_size == 0:
            raise McpbWorkflowError(
                f"MCPB.py step 1 did not produce required QM input: {required_input}"
            )
    expected = _expected_qm_artifacts(
        group_name,
        mcpb.force_constant_method,
        mcpb.software_version,
    )
    if mcpb.workflow == "prepare_inputs":
        return McpbSiteResult(
            site_id=site.config.id,
            workflow=mcpb.workflow,
            complete=False,
            work_dir=work,
            input_path=input_path,
            mcpb_input_pdb_path=input_pdb,
            restored_final_pdb_path=None,
            final_frcmod_path=None,
            generated_tleap_path=None,
            leap_setup_commands=(),
            leap_bond_commands=(),
            residue_renames=(),
            runs=tuple(runs),
            expected_qm_artifacts=expected,
            generated_files=tuple(_generated_files(work)),
            amberhome=amberhome,
        )

    pyscf_result = None
    if mcpb.workflow == "complete":
        _stage_qm_artifacts(mcpb, group_name=group_name, work_dir=work)
    elif mcpb.workflow == "pyscf":
        if provisional_prmtop is None or provisional_inpcrd is None:
            raise McpbWorkflowError(
                "mcpb.workflow: pyscf requires the pre-MCPB Amber prmtop/inpcrd "
                "for electrostatic embedding."
            )
        from mdprep.metals.pyscf_mcpb import McpbPySCFError, run_mcpb_pyscf

        try:
            pyscf_result = run_mcpb_pyscf(
                site,
                group_name=group_name,
                work_dir=work,
                provisional_prmtop=provisional_prmtop,
                provisional_inpcrd=provisional_inpcrd,
                provisional_structure=structure,
                refinement_result=refinement_result,
            )
        except McpbPySCFError as exc:
            raise McpbWorkflowError(str(exc)) from exc
    else:
        raise McpbWorkflowError(f"Unsupported completed MCPB workflow: {mcpb.workflow}")
    step_2 = {
        "seminario": "2s",
        "modified_seminario": "2ms",
        "empirical": "2e",
        "z_matrix": "2z",
    }[mcpb.force_constant_method]
    step_3 = {
        "all_ligating": "3a",
        "backbone_heavy": "3b",
        "backbone_all": "3c",
        "backbone_and_cb": "3d",
    }[mcpb.charge_restraint]
    for step in (step_2, step_3, "4b"):
        runs.append(
            _run_mcpb(
                executable,
                input_path,
                step,
                work,
                environment=mcpb_environment,
            )
        )

    generated_pdb = work / f"{group_name}_mcpbpy.pdb"
    final_frcmod = work / f"{group_name}_mcpbpy.frcmod"
    tleap_path = work / f"{group_name}_tleap.in"
    for required in (generated_pdb, final_frcmod, tleap_path):
        if not required.is_file() or required.stat().st_size == 0:
            raise McpbWorkflowError(f"MCPB.py did not produce required artifact: {required}")
    if mcpb.force_constant_method in {"seminario", "modified_seminario"}:
        from mdprep.metals.parameter_comparison import (
            McpbParameterComparisonError,
            parse_mcpb_seminario_terms,
        )

        try:
            parse_mcpb_seminario_terms(final_frcmod)
        except McpbParameterComparisonError as exc:
            raise McpbWorkflowError(
                "MCPB.py produced invalid Seminario bonded parameters for metal site "
                f"{site.config.id!r}: {exc}"
            ) from exc
    restored_pdb = work / f"{group_name}_mcpbpy.restored_ids.pdb"
    renames = restore_mcpb_residue_identities(
        generated_pdb,
        numbering=numbering,
        output_path=restored_pdb,
    )
    setup_commands, bond_commands = parse_mcpb_tleap_commands(tleap_path, work_dir=work)
    refitted_ligands = collect_mcpb_refitted_ligands(
        manifest=manifest,
        structure=structure,
        ligand_result=ligand_result,
        residue_renames=renames,
        work_dir=work,
        leap_setup_commands=setup_commands,
    )
    parameter_comparison = None
    if mcpb.parameter_comparison is not None:
        from mdprep.metals.parameter_comparison import (
            McpbParameterComparisonError,
            compare_mcpb_parameter_files,
        )

        try:
            parameter_comparison = compare_mcpb_parameter_files(
                final_frcmod,
                config=mcpb.parameter_comparison,
                output_dir=work / "parameter_comparison",
            )
        except McpbParameterComparisonError as exc:
            raise McpbWorkflowError(str(exc)) from exc
    return McpbSiteResult(
        site_id=site.config.id,
        workflow=mcpb.workflow,
        complete=True,
        work_dir=work,
        input_path=input_path,
        mcpb_input_pdb_path=input_pdb,
        restored_final_pdb_path=restored_pdb,
        final_frcmod_path=final_frcmod,
        generated_tleap_path=tleap_path,
        leap_setup_commands=tuple(setup_commands),
        leap_bond_commands=tuple(bond_commands),
        residue_renames=tuple(renames),
        runs=tuple(runs),
        expected_qm_artifacts=expected,
        generated_files=tuple(_generated_files(work)),
        amberhome=amberhome,
        refitted_ligands=tuple(refitted_ligands),
        pyscf_result=pyscf_result,
        parameter_comparison=parameter_comparison,
    )


def collect_mcpb_refitted_ligands(
    *,
    manifest: ManifestConfig,
    structure: PdbStructure,
    ligand_result: "LigandStageResult",
    residue_renames: list[dict[str, object]],
    work_dir: str | Path,
    leap_setup_commands: list[str],
) -> list[McpbRefittedLigand]:
    """Validate and expose MCPB's RESP-fitted mol2 for configured cofactors."""

    configured = [
        ligand
        for ligand in manifest.ligands
        if ligand.charge_method == "mcpb_resp_pyscf"
    ]
    if not configured:
        return []
    items = {item.ligand_id: item for item in ligand_result.ligands}
    output: list[McpbRefittedLigand] = []
    work = Path(work_dir)
    for ligand in configured:
        selector = ligand.selector
        matches = [
            residue
            for residue in structure.residues
            if residue.id.chain_id == selector.chain
            and residue.id.resname == selector.resname
            and residue.id.resid == selector.resid
            and residue.id.icode == selector.icode
        ]
        if len(matches) != 1:
            raise McpbWorkflowError(
                f"Could not map MCPB RESP ligand {ligand.id!r} uniquely in the "
                "hydrogenated input structure."
            )
        residue = matches[0]
        matching_renames = [
            rename
            for rename in residue_renames
            if rename.get("chain") == selector.chain
            and rename.get("resid") == selector.resid
            and rename.get("icode") == selector.icode
            and rename.get("original_resname") == selector.resname
        ]
        if len(matching_renames) != 1:
            raise McpbWorkflowError(
                f"MCPB.py did not report one exact residue rename for RESP ligand "
                f"{ligand.id!r}."
            )
        final_resname = str(matching_renames[0]["final_resname"])
        final_mol2 = (work / f"{final_resname}.mol2").resolve()
        if not final_mol2.is_file() or final_mol2.stat().st_size == 0:
            raise McpbWorkflowError(
                f"MCPB.py did not produce the final RESP-fitted mol2 for ligand "
                f"{ligand.id!r}: {final_mol2}"
            )
        try:
            parsed = read_mol2(final_mol2)
        except Mol2Error as exc:
            raise McpbWorkflowError(
                f"Could not validate MCPB RESP mol2 for ligand {ligand.id!r}: {exc}"
            ) from exc
        expected_names = residue.atom_names()
        observed_names = [atom.name for atom in parsed.atoms]
        if observed_names != expected_names:
            raise McpbWorkflowError(
                f"MCPB RESP mol2 atom-name/order mismatch for ligand {ligand.id!r}: "
                f"{observed_names} != {expected_names}."
            )
        if {atom.subst_name for atom in parsed.atoms} != {final_resname}:
            raise McpbWorkflowError(
                f"MCPB RESP mol2 for ligand {ligand.id!r} does not use its final "
                f"residue name {final_resname!r}."
            )
        deviations = [
            dist(
                (pdb_atom.x, pdb_atom.y, pdb_atom.z),
                (mol2_atom.x, mol2_atom.y, mol2_atom.z),
            )
            for pdb_atom, mol2_atom in zip(residue.atoms, parsed.atoms, strict=True)
        ]
        maximum_deviation = max(deviations, default=0.0)
        if maximum_deviation > 0.002:
            raise McpbWorkflowError(
                f"MCPB RESP mol2 moved ligand {ligand.id!r} by "
                f"{maximum_deviation:.6f} A; only charge/type changes are allowed."
            )
        item = items.get(ligand.id)
        if item is None or item.provisional_mol2_path is None:
            raise McpbWorkflowError(
                f"MCPB RESP ligand {ligand.id!r} has no recorded provisional AM1-BCC mol2."
            )
        load_command = f"{final_resname} = loadmol2 {final_mol2}"
        if load_command not in leap_setup_commands:
            raise McpbWorkflowError(
                f"Final MCPB tleap setup does not load the RESP-fitted mol2 for "
                f"ligand {ligand.id!r}: {load_command}"
            )
        output.append(
            McpbRefittedLigand(
                ligand_id=ligand.id,
                original_resname=selector.resname,
                final_resname=final_resname,
                final_mol2_path=final_mol2,
                provisional_mol2_path=Path(item.provisional_mol2_path).resolve(),
                atom_names=tuple(observed_names),
                atom_count=len(observed_names),
                fragment_charge=float(parsed.total_charge),
                maximum_coordinate_deviation_angstrom=maximum_deviation,
            )
        )
    return output


def write_mcpb_numbered_pdb(
    structure: PdbStructure,
    path: str | Path,
) -> McpbNumberingMap:
    atoms: list[AtomRecord] = []
    original_by_atom: dict[int, AtomRecord] = {}
    original_by_residue: dict[int, ResidueRecord] = {}
    atom_id_by_identity: dict[tuple[str, str, int, str | None, str], int] = {}
    resid_by_key: dict[tuple[str, int, str | None], int] = {}
    atom_id = 0
    for residue_id, residue in enumerate(structure.residues, start=1):
        original_by_residue[residue_id] = residue
        residue_key = (residue.id.chain_id, residue.id.resid, residue.id.icode)
        if residue_key in resid_by_key:
            raise McpbWorkflowError(
                f"Residue identity {residue.id.display()} is ambiguous before MCPB.py numbering."
            )
        resid_by_key[residue_key] = residue_id
        for atom in residue.atoms:
            atom_id += 1
            if atom.atom_identity in atom_id_by_identity:
                raise McpbWorkflowError(f"Duplicate atom identity before MCPB.py: {atom.atom_identity}")
            atom_id_by_identity[atom.atom_identity] = atom_id
            original_by_atom[atom_id] = atom
            atoms.append(
                replace(
                    atom,
                    serial=atom_id,
                    chain_id="A",
                    resid=residue_id,
                    icode=None,
                    altloc=None,
                )
            )
    numbered = PdbStructure(
        path=Path(path),
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=1,
        used_model=1,
        warnings=list(structure.warnings),
    )
    write_pdb(numbered, path)
    return McpbNumberingMap(
        original_by_mcpb_atom_id=original_by_atom,
        original_by_mcpb_resid=original_by_residue,
        mcpb_atom_id_by_original_identity=atom_id_by_identity,
        mcpb_resid_by_original_key=resid_by_key,
    )


def build_mcpb_input(
    site: ResolvedMetalSite,
    *,
    manifest: ManifestConfig,
    numbering: McpbNumberingMap,
    original_pdb: Path,
    ion_mol2: list[Path],
    ligand_mol2: list[Path],
    ligand_frcmods: list[Path],
    gaff: int,
    group_name: str,
) -> str:
    assert site.config.mcpb is not None
    mcpb = site.config.mcpb
    ion_ids = [numbering.mcpb_atom_id_by_original_identity[ion.atom.atom_identity] for ion in site.ions]
    additional_resids = [
        numbering.mcpb_resid_by_original_key[
            (residue.id.chain_id, residue.id.resid, residue.id.icode)
        ]
        for residue in site.additional_residues
    ]
    added_pairs = [
        (
            numbering.mcpb_atom_id_by_original_identity[bond.ion.atom.atom_identity],
            numbering.mcpb_atom_id_by_original_identity[bond.coordinator.atom_identity],
        )
        for bond in site.bonds
        if not bond.discovered_by_cutoff
    ]
    lines = [
        f"original_pdb {original_pdb.name}",
        "ion_ids " + " ".join(str(value) for value in ion_ids),
        "ion_mol2files " + " ".join(path.name for path in ion_mol2),
        f"group_name {group_name}",
        f"cut_off {mcpb.cutoff_angstrom:.6f}",
        f"force_field {manifest.protein.forcefield}",
        f"water_model {manifest.protein.water_model.lower()}",
        f"ion_paraset {mcpb.provisional_nonbonded_parameter_set}",
        f"gaff {gaff}",
        f"software_version {mcpb.software_version}",
        f"smmodel_chg {mcpb.small_model_charge}",
        f"smmodel_spin {mcpb.small_model_spin}",
        f"lgmodel_chg {mcpb.large_model_charge}",
        f"lgmodel_spin {mcpb.large_model_spin}",
        f"scale_factor {mcpb.scale_factor:.8f}",
        f"large_opt {mcpb.large_opt}",
    ]
    if ligand_mol2:
        lines.append("naa_mol2files " + " ".join(path.name for path in ligand_mol2))
    if ligand_frcmods:
        lines.append("frcmod_files " + " ".join(path.name for path in ligand_frcmods))
    if additional_resids:
        lines.append("additional_resids " + " ".join(str(value) for value in additional_resids))
    if added_pairs:
        lines.append(
            "add_bonded_pairs " + " ".join(f"{a}-{b}" for a, b in added_pairs)
        )
    lines.append("")
    return "\n".join(lines)


def restore_mcpb_residue_identities(
    generated_pdb: str | Path,
    *,
    numbering: McpbNumberingMap,
    output_path: str | Path,
) -> list[dict[str, object]]:
    generated = read_pdb(generated_pdb)
    atoms: list[AtomRecord] = []
    renames_by_resid: dict[int, dict[str, object]] = {}
    for atom in generated.atoms:
        original_atom = numbering.original_by_mcpb_atom_id.get(atom.serial or -1)
        original_residue = numbering.original_by_mcpb_resid.get(atom.resid)
        if original_atom is None or original_residue is None:
            raise McpbWorkflowError(
                f"Could not map MCPB.py atom {atom.serial} residue {atom.resid} back to input identity."
            )
        coordinate_deviation = dist(
            (atom.x, atom.y, atom.z),
            (original_atom.x, original_atom.y, original_atom.z),
        )
        if coordinate_deviation > 0.002:
            raise McpbWorkflowError(
                "MCPB.py changed an input atom coordinate by "
                f"{coordinate_deviation:.6f} A for {original_atom.atom_identity}; "
                "the refined fixed geometry must be preserved."
            )
        if atom.resname != original_residue.id.resname:
            renames_by_resid.setdefault(
                atom.resid,
                {
                    "chain": original_residue.id.chain_id,
                    "resid": original_residue.id.resid,
                    "icode": original_residue.id.icode,
                    "original_resname": original_residue.id.resname,
                    "final_resname": atom.resname,
                    "source": "MCPB.py",
                },
            )
        atoms.append(
            replace(
                atom,
                serial=original_atom.serial,
                chain_id=original_residue.id.chain_id,
                resid=original_residue.id.resid,
                icode=original_residue.id.icode,
                element=original_atom.element,
            )
        )
    restored = PdbStructure(
        path=Path(output_path),
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=1,
        used_model=1,
    )
    write_pdb(restored, output_path)
    return [renames_by_resid[key] for key in sorted(renames_by_resid)]


def parse_mcpb_tleap_commands(
    path: str | Path,
    *,
    work_dir: str | Path,
) -> tuple[list[str], list[str]]:
    """Extract only parameter/type setup and bond commands from MCPB.py output."""

    work = Path(work_dir)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    setup: list[str] = []
    bonds: list[str] = []
    in_atom_types = False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "addAtomTypes {":
            in_atom_types = True
            setup.append(line)
            continue
        if in_atom_types:
            setup.append(line)
            if line == "}":
                in_atom_types = False
            continue
        if " = loadmol2 " in line:
            prefix, filename = line.rsplit(" ", 1)
            local = work / filename
            if not local.is_file():
                raise McpbWorkflowError(f"MCPB.py tleap input references missing mol2: {local}")
            setup.append(f"{prefix} {local.resolve()}")
        elif line.startswith("loadamberparams "):
            filename = line.split(maxsplit=1)[1]
            local = work / filename
            setup.append(
                f"loadamberparams {local.resolve()}" if local.is_file() else line
            )
        elif line.startswith("loadoff "):
            filename = line.split(maxsplit=1)[1]
            local = work / filename
            if not local.is_file():
                raise McpbWorkflowError(f"MCPB.py tleap input references missing library: {local}")
            setup.append(f"loadoff {local.resolve()}")
        elif line.startswith("bond mol."):
            bonds.append("bond system." + line[len("bond mol.") :].replace(" mol.", " system."))
    if not setup:
        raise McpbWorkflowError(f"No MCPB.py parameter setup commands were found in {path}.")
    if not bonds:
        raise McpbWorkflowError(f"No bonded metal-site commands were found in {path}.")
    return setup, bonds


def _run_mcpb(
    executable: str,
    input_path: Path,
    step: str,
    work_dir: Path,
    *,
    environment: dict[str, str] | None = None,
) -> McpbRun:
    if step in {"2s", "2ms"}:
        compatibility_script = Path(__file__).with_name("mcpb_compat.py").resolve()
        command = [
            sys.executable,
            str(compatibility_script),
            "--mcpb-script",
            executable,
            "--",
            "-i",
            input_path.name,
            "-s",
            step,
        ]
    else:
        command = [executable, "-i", input_path.name, "-s", step]
    if environment is None:
        result = run_command(command, cwd=work_dir)
    else:
        result = run_command(command, cwd=work_dir, env=environment)
    label = step.replace("/", "_")
    stdout_path = work_dir / f"mcpb.step_{label}.stdout.txt"
    stderr_path = work_dir / f"mcpb.step_{label}.stderr.txt"
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise McpbWorkflowError(
            f"MCPB.py step {step} failed with exit code {result.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return McpbRun(step=step, result=result, stdout_path=stdout_path, stderr_path=stderr_path)


def _write_mcpb_ion_mol2(
    site: ResolvedMetalSite,
    *,
    numbering: McpbNumberingMap,
    work_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    for index, ion in enumerate(site.ions, start=1):
        atom_type = f"{ion.element}{ion.charge}+"
        parameter = AmberIonParameter(
            element=ion.element,
            charge=ion.charge,
            atom_type=atom_type,
            mass=0.0,
            rmin_over_2_angstrom=0.0,
            epsilon_kcal_mol=0.0,
            parameter_set="mcpb_input",
            water_model="",
            frcmod_name="",
            frcmod_path=Path("."),
        )
        numbered_atom_id = numbering.mcpb_atom_id_by_original_identity[ion.atom.atom_identity]
        numbered_resid = numbering.mcpb_resid_by_original_key[
            (ion.residue.id.chain_id, ion.residue.id.resid, ion.residue.id.icode)
        ]
        numbered_atom = replace(ion.atom, serial=numbered_atom_id, chain_id="A", resid=numbered_resid, icode=None)
        numbered_residue = ResidueRecord(
            id=ResidueId("A", ion.residue.id.resname, numbered_resid),
            atoms=[numbered_atom],
            record_names={numbered_atom.record_name},
            original_index=numbered_resid - 1,
        )
        numbered_ion = replace(ion, atom=numbered_atom, residue=numbered_residue)
        path = work_dir / f"ion_{index}_{ion.residue.id.resname}.mol2"
        write_single_ion_mol2(numbered_ion, parameter, path)
        paths.append(path)
    return paths


def _stage_ligand_parameters(
    *,
    manifest: ManifestConfig,
    ligand_result: "LigandStageResult",
    work_dir: Path,
) -> tuple[list[Path], list[Path], int]:
    by_id = {item.ligand_id: item for item in ligand_result.ligands}
    families = {ligand.atom_types for ligand in manifest.ligands}
    if len(families) > 1:
        raise McpbWorkflowError(
            "MCPB.py accepts one GAFF family per run; bonded metal preparation cannot mix GAFF and GAFF2."
        )
    gaff = 0 if not families else (1 if next(iter(families)) == "gaff" else 2)
    mol2_paths: list[Path] = []
    frcmod_paths: list[Path] = []
    seen_resnames: set[str] = set()
    for ligand in manifest.ligands:
        item = by_id.get(ligand.id)
        if item is None or item.final_mol2_path is None or item.final_frcmod_path is None:
            raise McpbWorkflowError(f"Ligand {ligand.id!r} has no final parameters for MCPB.py.")
        resname = ligand.selector.resname
        if resname not in seen_resnames:
            mol2_target = work_dir / f"{resname}.mol2"
            shutil.copyfile(item.final_mol2_path, mol2_target)
            mol2_paths.append(mol2_target)
            seen_resnames.add(resname)
        frcmod_target = work_dir / f"ligand_{_safe_group_name(ligand.id)}.frcmod"
        shutil.copyfile(item.final_frcmod_path, frcmod_target)
        frcmod_paths.append(frcmod_target)
    return mol2_paths, frcmod_paths, gaff


def _stage_qm_artifacts(config: object, *, group_name: str, work_dir: Path) -> None:
    artifacts = config.artifacts  # type: ignore[attr-defined]
    assert artifacts is not None
    copies = [(artifacts.large_mk_log, work_dir / f"{group_name}_large_mk.log")]
    if artifacts.small_fc_log:
        copies.append((artifacts.small_fc_log, work_dir / f"{group_name}_small_fc.log"))
    if artifacts.small_opt_fchk:
        copies.append((artifacts.small_opt_fchk, work_dir / f"{group_name}_small_opt.fchk"))
    for source, target in copies:
        source_path = Path(source)
        if not source_path.is_file() or source_path.stat().st_size == 0:
            raise McpbWorkflowError(f"Configured MCPB.py QM artifact is missing or empty: {source_path}")
        if source_path.resolve() != target.resolve():
            shutil.copyfile(source_path, target)


def _expected_qm_artifacts(
    group_name: str,
    method: str,
    software_version: str,
) -> tuple[str, ...]:
    names = [f"{group_name}_large_mk.log"]
    if method == "z_matrix" or (
        method in {"seminario", "modified_seminario"} and software_version == "gms"
    ):
        names.append(f"{group_name}_small_fc.log")
    if method in {"seminario", "modified_seminario"} and software_version != "gms":
        names.append(f"{group_name}_small_opt.fchk")
    return tuple(names)


def _required_qm_inputs(
    work_dir: Path,
    *,
    group_name: str,
    software_version: str,
) -> tuple[Path, Path, Path]:
    suffix = ".inp" if software_version == "gms" else ".com"
    return (
        work_dir / f"{group_name}_small_opt{suffix}",
        work_dir / f"{group_name}_small_fc{suffix}",
        work_dir / f"{group_name}_large_mk{suffix}",
    )


def _validate_nonstandard_site_residues(
    site: ResolvedMetalSite,
    manifest: ManifestConfig,
) -> None:
    """Require parameters for every non-protein/non-water MCPB site residue."""

    ligand_keys = {
        (ligand.selector.chain, ligand.selector.resid, ligand.selector.icode)
        for ligand in manifest.ligands
    }
    site_residues = {
        (
            bond.coordinator_residue.id.chain_id,
            bond.coordinator_residue.id.resid,
            bond.coordinator_residue.id.icode,
        ): bond.coordinator_residue
        for bond in site.bonds
    }
    site_residues.update(
        {
            (residue.id.chain_id, residue.id.resid, residue.id.icode): residue
            for residue in site.additional_residues
        }
    )
    missing = [
        residue.id.display()
        for key, residue in site_residues.items()
        if not is_standard_protein_residue(residue)
        and not is_water_residue(residue)
        and key not in ligand_keys
    ]
    if missing:
        raise McpbWorkflowError(
            "Every non-protein/non-water residue in an MCPB.py site must be configured and "
            "parameterized under ligands; missing: " + ", ".join(sorted(missing))
        )


def _validate_mcpb_structure_scope(structure: PdbStructure) -> None:
    protein_chains = {
        residue.id.chain_id
        for residue in structure.residues
        if is_standard_protein_residue(residue)
    }
    if len(protein_chains) > 1:
        raise McpbWorkflowError(
            "Bonded MCPB.py preparation currently supports one protein chain because MCPB.py "
            "discards PDB chain IDs while constructing capped QM models. Use the nonbonded model "
            "or prepare a reviewed one-chain MCPB input."
        )


def _safe_group_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_")
    if not safe:
        safe = "metal_site"
    if safe[0].isdigit():
        safe = "site_" + safe
    return safe


def _generated_files(work_dir: Path) -> list[Path]:
    return sorted((path for path in work_dir.iterdir() if path.is_file()), key=lambda path: path.name)


def _build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    return [
        ResidueRecord(
            id=ResidueId(chain_id=key[0], resname=key[1], resid=key[2], icode=key[3]),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, (key, residue_atoms) in enumerate(grouped.items())
    ]
