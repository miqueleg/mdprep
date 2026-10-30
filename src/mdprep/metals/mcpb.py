"""Preserved, staged MCPB.py workflow for bonded metal centers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import dist
from pathlib import Path
import os
import re
import shutil
import sys
from typing import TYPE_CHECKING, Collection, Sequence

from mdprep.ambertools.mol2 import Mol2Error, read_mol2
from mdprep.config.models import ManifestConfig
from mdprep.external.discovery import which_executable
from mdprep.external.runner import CommandResult, run_command
from mdprep.metals.coordination import ResolvedMetalIon, ResolvedMetalSite
from mdprep.metals.nonbonded import (
    AmberIonParameter,
    load_amber_ion_parameter,
    NonbondedMetalError,
    ion_atom_type,
    resolve_amberhome,
    write_single_ion_mol2,
)
from mdprep.residues.workflow import kept_residue_names
from mdprep.structure.classify import (
    WATER_RESIDUES,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord, build_residues
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
    # Keyed by the whole residue identity, chain/resname/resid/icode, the
    # same tuple ResidueId is. Keyed on chain and number alone, a ligand
    # that a prediction numbered 1 inside the protein's own chain looked
    # like the protein's first residue, and the build stopped calling
    # them ambiguous -- which they were, but only to this key.
    mcpb_resid_by_original_key: dict[tuple[str, str, int, str | None], int]


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
class McpbSiteCopy:
    """A copy of the fitted site, built from the fit's templates.

    Not fitted: the copy's residues take the names -- and with them the
    atom types and charges -- of the reference residues they correspond to,
    and its metal is bonded to its own donors with the same terms.
    """

    site_id: str
    parameters_from: str
    residue_renames: tuple[dict[str, object], ...]
    leap_bond_commands: tuple[str, ...]
    refitted_ligands: tuple[McpbRefittedLigand, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "parameters_from": self.parameters_from,
            "residue_renames": list(self.residue_renames),
            "leap_bond_commands": list(self.leap_bond_commands),
            "refitted_ligands": [item.to_dict() for item in self.refitted_ligands],
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
    # Copies of this site in other copies of the protein, and the structure
    # with their residues renamed as well; the fit's own is above.
    copies: tuple[McpbSiteCopy, ...] = ()
    final_pdb_path: Path | None = None

    @property
    def all_refitted_ligands(self) -> tuple[McpbRefittedLigand, ...]:
        return self.refitted_ligands + tuple(
            item for copy in self.copies for item in copy.refitted_ligands
        )

    @property
    def all_residue_renames(self) -> tuple[dict[str, object], ...]:
        copied = {
            (rename["chain"], rename["resid"], rename["icode"])
            for copy in self.copies
            for rename in copy.residue_renames
        }
        return tuple(
            rename
            for rename in self.residue_renames
            if (rename["chain"], rename["resid"], rename["icode"]) not in copied
        ) + tuple(rename for copy in self.copies for rename in copy.residue_renames)

    @property
    def all_leap_bond_commands(self) -> tuple[str, ...]:
        return self.leap_bond_commands + tuple(
            command for copy in self.copies for command in copy.leap_bond_commands
        )

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
            "copies": [copy.to_dict() for copy in self.copies],
            "final_pdb_path": str(self.final_pdb_path) if self.final_pdb_path else None,
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
    copies: Sequence[ResolvedMetalSite] = (),
) -> McpbSiteResult:
    """Fit one bonded site with MCPB.py, and build its copies from the fit.

    ``copies`` are the sites of other copies of the protein that reuse this
    site's parameters (``parameters_from``). They are not fitted; after the
    fit their residues are given its templates and their metals its bonds.
    """

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
    kept = kept_residue_names(manifest)
    for item in (site, *copies):
        _validate_site_in_one_chain(item, protein_names=kept)
    _validate_nonstandard_site_residues(site, manifest)

    work = Path(output_dir)
    work.mkdir(parents=True, exist_ok=True)
    group_name = _safe_group_name(site.config.id)
    input_pdb = work / "system.mcpb_input.pdb"
    numbering = write_mcpb_numbered_pdb(
        structure, input_pdb, ion_names=_mcpb_ion_names(site)
    )
    ligand_mol2, ligand_frcmods, families = _stage_ligand_parameters(
        manifest=manifest,
        ligand_result=ligand_result,
        work_dir=work,
    )
    residue_mol2, residue_frcmods, residue_families = _stage_kept_residue_parameters(
        manifest, amberhome=amberhome, work_dir=work
    )
    gaff = _gaff_version(families | residue_families)
    ligand_mol2 = ligand_mol2 + residue_mol2
    ligand_frcmods = (
        ligand_frcmods
        + residue_frcmods
        + _stage_coordinating_ion_frcmods(site, manifest, work_dir=work)
        + _stage_bound_water_frcmod(site, work_dir=work)
    )
    ion_mol2 = _write_mcpb_ion_mol2(site, numbering=numbering, work_dir=work)
    ligand_mol2 = (
        ligand_mol2
        + _write_mcpb_coordinating_ion_mol2(site, numbering=numbering, work_dir=work)
        + _write_mcpb_ion_charge_mol2(site, numbering=numbering, work_dir=work)
        + _write_mcpb_water_mol2(site, manifest=manifest, work_dir=work)
    )
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
            (seminario_run,) = [run for run in runs if run.step == step_2]
            negative = negative_seminario_terms(seminario_run.result.stdout, numbering)
            raise McpbWorkflowError(
                "MCPB.py produced invalid Seminario bonded parameters for metal site "
                f"{site.config.id!r}: {exc}"
                + (
                    ". Negative, and written as zero by MCPB.py: "
                    + "; ".join(negative)
                    + ". The Hessian has no restoring force there. At a geometry "
                    "that is not a minimum of the QM method -- "
                    "a crystal site with only its hydrogens relaxed -- a metal's "
                    "angles, its softest terms, are the first to go. See "
                    "docs/metals.md, 'A zero force constant is reported, not shipped'."
                    if negative
                    else ""
                )
            ) from exc
    restored_pdb = work / f"{group_name}_mcpbpy.restored_ids.pdb"
    renames = restore_mcpb_residue_identities(
        generated_pdb,
        numbering=numbering,
        output_path=restored_pdb,
    )
    # A kept residue went to MCPB.py as a mol2 so it could be modelled. Its
    # bonded model loads only the templates of the residues it renamed, but a
    # unit of the kept residue's own name from MCPB.py would replace the
    # library whose head and tail bond it into the chain wherever it is not
    # on the metal, so none is taken.
    setup_commands, bond_commands = parse_mcpb_tleap_commands(
        tleap_path, work_dir=work, skip_units=kept
    )
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
    result = McpbSiteResult(
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
        final_pdb_path=restored_pdb,
    )
    if not copies:
        return result
    return apply_mcpb_site_copies(
        result,
        site=site,
        copies=copies,
        structure=structure,
        manifest=manifest,
        numbering=numbering,
        output_path=work / f"{group_name}_mcpbpy.with_copies.pdb",
    )


def apply_mcpb_site_copies(
    result: McpbSiteResult,
    *,
    site: ResolvedMetalSite,
    copies: Sequence[ResolvedMetalSite],
    structure: PdbStructure,
    manifest: ManifestConfig,
    numbering: McpbNumberingMap,
    output_path: str | Path,
) -> McpbSiteResult:
    """Give each copy of a fitted site the fit's templates and bonds.

    The fit renamed the reference site's residues -- each histidine, the
    cofactor, the bound water and halide, the metal -- to templates carrying
    its atom types and charges, and bonded the metal to its donors. A copy
    reuses all of it: each of its residues takes the name of the reference
    residue it corresponds to, and its metal is bonded to its own donors
    with the same atom names. Correspondence is by position in the site's
    lists, which a copy states in the reference's order, never by residue
    number, because copies are often numbered differently.

    Anything that would make the copy something other than the reference --
    a residue whose atoms are not the template's, a template bond to a
    residue the copy does not list -- is refused rather than adapted.
    """

    templates = {
        command.split(" = loadmol2 ", 1)[0].strip()
        for command in result.leap_setup_commands
        if " = loadmol2 " in command
    }
    reference_residues = _site_residues(site)
    reference_keys = {id(residue) for residue in reference_residues}
    renames_by_key = {
        (rename["chain"], rename["resid"], rename["icode"]): rename
        for rename in result.residue_renames
    }
    if result.restored_final_pdb_path is None:
        raise McpbWorkflowError("The MCPB.py fit has no restored structure to copy from.")
    restored = read_pdb(result.restored_final_pdb_path)
    restored_by_serial = {atom.serial: atom for atom in restored.atoms}
    index_of = {id(residue): index for index, residue in enumerate(structure.residues, start=1)}

    # What the fit made of each reference residue: its new name, if a
    # template, and the name of each of its atoms afterwards.
    template_of: dict[int, str] = {}
    atom_names_of: dict[int, dict[str, str]] = {}
    for residue in reference_residues:
        rename = renames_by_key.get((residue.id.chain_id, residue.id.resid, residue.id.icode))
        if rename is not None and rename["final_resname"] in templates:
            template_of[id(residue)] = str(rename["final_resname"])
        atom_names_of[id(residue)] = {
            atom.name: restored_by_serial[atom.serial].name
            for atom in residue.atoms
            if atom.serial in restored_by_serial
        }

    # The fit's bonds that belong to the site: the metal's to its donors, and
    # the peptide links MCPB.py writes between a renamed amino acid and the
    # residues either side of it, since a template loaded from mol2 knows
    # nothing of the chain it sits in. A link is carried to a copy as a step
    # along its chain, and the residue it lands on has to be the
    # counterpart of the one the fit linked to. Bonds between two residues
    # outside the site -- disulfides -- are the whole system's already.
    site_bonds: list[tuple[ResidueRecord, str, ResidueRecord, str]] = []
    links: list[tuple[ResidueRecord, str, int, str, ResidueRecord, bool]] = []
    for command in result.leap_bond_commands:
        ends = _bond_command_ends(command)
        if ends is None:
            continue
        (first_index, first_atom), (second_index, second_atom) = ends
        first = numbering.original_by_mcpb_resid.get(first_index)
        second = numbering.original_by_mcpb_resid.get(second_index)
        if first is None or second is None:
            raise McpbWorkflowError(f"MCPB.py bond command names an unknown residue: {command}")
        first_in = id(first) in reference_keys
        second_in = id(second) in reference_keys
        if first_in and second_in:
            site_bonds.append((first, first_atom, second, second_atom))
        elif first_in or second_in:
            linked, linked_atom, other, other_atom = (
                (first, first_atom, second, second_atom)
                if first_in
                else (second, second_atom, first, first_atom)
            )
            step = index_of[id(other)] - index_of[id(linked)]
            if (
                abs(step) != 1
                or other.id.chain_id != linked.id.chain_id
                or not _is_chain_residue(other, manifest)
            ):
                raise McpbWorkflowError(
                    f"MCPB.py bonded the site to a residue it does not list: {command}"
                )
            links.append((linked, linked_atom, step, other_atom, other, first_in))

    atoms = list(restored.atoms)
    position = {atom.serial: index for index, atom in enumerate(atoms)}
    built: list[McpbSiteCopy] = []
    for copy in copies:
        copy_residues = _site_residues(copy)
        if len(copy_residues) != len(reference_residues):
            raise McpbWorkflowError(
                f"Metal site {copy.config.id!r} lists {len(copy_residues)} residues where "
                f"{site.config.id!r} lists {len(reference_residues)}; a copy names the "
                "counterpart of every residue of the site it copies."
            )
        counterpart: dict[int, ResidueRecord] = {}
        for reference, own in zip(reference_residues, copy_residues):
            previous = counterpart.setdefault(id(reference), own)
            if previous is not own:
                raise McpbWorkflowError(
                    f"Metal site {copy.config.id!r} pairs {reference.id.display()} with "
                    f"both {previous.id.display()} and {own.id.display()}."
                )
        renames: list[dict[str, object]] = []
        for reference in reference_residues:
            own = counterpart[id(reference)]
            names = atom_names_of[id(reference)]
            expected = set(names.values())
            observed = {names.get(atom.name, atom.name) for atom in own.atoms}
            if observed != expected:
                raise McpbWorkflowError(
                    f"{own.id.display()} cannot take the template fitted for "
                    f"{reference.id.display()}: it has atoms {sorted(observed - expected)} "
                    f"that the template lacks and lacks {sorted(expected - observed)}. A "
                    "copy has to be the same residue, protonated the same way."
                )
            final_name = template_of.get(id(reference))
            for atom in own.atoms:
                index = position.get(atom.serial)
                if index is None:
                    raise McpbWorkflowError(
                        f"{own.id.display()} is missing from the MCPB.py output structure."
                    )
                atoms[index] = replace(
                    atoms[index],
                    resname=final_name or atoms[index].resname,
                    name=names.get(atom.name, atom.name),
                )
            if final_name is not None:
                renames.append(
                    {
                        "chain": own.id.chain_id,
                        "resid": own.id.resid,
                        "icode": own.id.icode,
                        "original_resname": own.id.resname,
                        "final_resname": final_name,
                        "source": (
                            f"mdprep: the MCPB.py template fitted for "
                            f"{reference.id.display()} in site {site.config.id}"
                        ),
                    }
                )
        bonds = [
            f"bond system.{index_of[id(counterpart[id(first)])]}.{first_atom} "
            f"system.{index_of[id(counterpart[id(second)])]}.{second_atom}"
            for first, first_atom, second, second_atom in site_bonds
        ]
        for site_residue, site_atom, step, other_atom, other, site_first in links:
            own = counterpart[id(site_residue)]
            own_index = index_of[id(own)]
            neighbour_index = own_index + step
            neighbour = (
                structure.residues[neighbour_index - 1]
                if 1 <= neighbour_index <= len(structure.residues)
                else None
            )
            if (
                neighbour is None
                or neighbour.id.chain_id != own.id.chain_id
                or not _is_chain_residue(neighbour, manifest)
                or _one_letter(neighbour.id.resname) != _one_letter(other.id.resname)
            ):
                raise McpbWorkflowError(
                    f"The fit links {site_residue.id.display()} to its neighbour "
                    f"{other.id.display()}, and the residue in that place next to "
                    f"{own.id.display()} is "
                    f"{neighbour.id.display() if neighbour is not None else 'missing'}; "
                    "a copy of the site needs the same chain around it."
                )
            ends_in_order = (
                (f"system.{own_index}.{site_atom}", f"system.{neighbour_index}.{other_atom}")
                if site_first
                else (f"system.{neighbour_index}.{other_atom}", f"system.{own_index}.{site_atom}")
            )
            bonds.append(f"bond {ends_in_order[0]} {ends_in_order[1]}")
        bonds = tuple(bonds)
        built.append(
            McpbSiteCopy(
                site_id=copy.config.id,
                parameters_from=site.config.id,
                residue_renames=tuple(renames),
                leap_bond_commands=bonds,
                refitted_ligands=tuple(
                    _copied_refits(result, manifest, counterpart, reference_residues, structure)
                ),
            )
        )

    copied = PdbStructure(
        path=Path(output_path),
        atoms=atoms,
        residues=build_residues(atoms),
        model_count=1,
        used_model=1,
    )
    write_pdb(copied, output_path)
    return replace(result, copies=tuple(built), final_pdb_path=Path(output_path))


def _is_chain_residue(residue: ResidueRecord, manifest: ManifestConfig) -> bool:
    """A residue of a protein chain: a standard one, or one kept as it is."""

    return is_standard_protein_residue(residue) or residue.id.resname in kept_residue_names(
        manifest
    )


def _one_letter(resname: str) -> str:
    from mdprep.structure.oligomer import ONE_LETTER

    return ONE_LETTER.get(resname, resname)


def _site_residues(site: ResolvedMetalSite) -> list[ResidueRecord]:
    """The site's residues in a fixed order: ions, coordinators, additional."""

    found: list[ResidueRecord] = []
    for residue in (
        *(ion.residue for ion in site.ions),
        *(bond.coordinator_residue for bond in site.bonds),
        *site.additional_residues,
    ):
        if all(residue is not seen for seen in found):
            found.append(residue)
    return found


_BOND_COMMAND = re.compile(r"^bond system\.(\d+)\.(\S+) system\.(\d+)\.(\S+)$")


def _bond_command_ends(command: str) -> tuple[tuple[int, str], tuple[int, str]] | None:
    match = _BOND_COMMAND.match(command.strip())
    if match is None:
        return None
    return (int(match[1]), match[2]), (int(match[3]), match[4])


def _copied_refits(
    result: McpbSiteResult,
    manifest: ManifestConfig,
    counterpart: dict[int, ResidueRecord],
    reference_residues: list[ResidueRecord],
    structure: PdbStructure,
) -> list[McpbRefittedLigand]:
    """The fitted cofactor template, for each copy of the cofactor."""

    refits = {item.ligand_id: item for item in result.refitted_ligands}
    copied: list[McpbRefittedLigand] = []
    for ligand in manifest.ligands:
        source = refits.get(ligand.parameters_from or "")
        if source is None:
            continue
        own = None
        for reference in reference_residues:
            candidate = counterpart[id(reference)]
            if (
                candidate.id.chain_id == ligand.selector.chain
                and candidate.id.resid == ligand.selector.resid
                and candidate.id.icode == ligand.selector.icode
                and candidate.id.resname == ligand.selector.resname
            ):
                own = candidate
        if own is None:
            continue
        if own.atom_names() != list(source.atom_names):
            raise McpbWorkflowError(
                f"Ligand {ligand.id!r} cannot take {source.ligand_id!r}'s fitted charges: "
                f"its atoms are {own.atom_names()}, the template's {list(source.atom_names)}."
            )
        copied.append(
            replace(
                source,
                ligand_id=ligand.id,
                original_resname=ligand.selector.resname,
                maximum_coordinate_deviation_angstrom=0.0,
            )
        )
    return copied


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

    # A copy takes its reference's fitted template after the fit, and MCPB.py
    # renamed nothing of it.
    configured = [
        ligand
        for ligand in manifest.ligands
        if ligand.charge_method == "mcpb_resp_pyscf" and ligand.parameters_from is None
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
    *,
    ion_names: dict[tuple[str, str, int, str | None], str] | None = None,
) -> McpbNumberingMap:
    """Renumber a structure into the contiguous form MCPB.py expects.

    ``ion_names`` renames both the residue and its single atom for the metal
    ions. MCPB.py reads an ion's charge by its ATOM name in one place and by
    its RESIDUE name in another, and builds a van der Waals key from the
    element symbol plus that charge, so all three have to be the same bare
    symbol: FE, giving Fe2 for Fe(II). A deposition may call the atom
    anything, and Amber's own libraries name the residue after the oxidation
    state, so neither can be used as it stands.

    The rename applies only to these working files. The original atom is kept
    in the numbering map, so the structure handed back afterwards still
    carries the names the input used.
    """

    renames = ion_names or {}
    atoms: list[AtomRecord] = []
    original_by_atom: dict[int, AtomRecord] = {}
    original_by_residue: dict[int, ResidueRecord] = {}
    atom_id_by_identity: dict[tuple[str, str, int, str | None, str], int] = {}
    resid_by_key: dict[tuple[str, str, int, str | None], int] = {}
    atom_id = 0
    for residue_id, residue in enumerate(structure.residues, start=1):
        original_by_residue[residue_id] = residue
        residue_key = (
            residue.id.chain_id,
            residue.id.resname,
            residue.id.resid,
            residue.id.icode,
        )
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
            renamed = renames.get(residue_key)
            atoms.append(
                replace(
                    atom,
                    serial=atom_id,
                    name=atom.name if renamed is None else renamed,
                    resname=atom.resname if renamed is None else renamed,
                    chain_id="A",
                    resid=residue_id,
                    icode=None,
                    altloc=None,
                )
            )
    numbered = PdbStructure(
        path=Path(path),
        atoms=atoms,
        residues=build_residues(atoms),
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
            (residue.id.chain_id, residue.id.resname, residue.id.resid, residue.id.icode)
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
        residues=build_residues(atoms),
        model_count=1,
        used_model=1,
    )
    write_pdb(restored, output_path)
    return [renames_by_resid[key] for key in sorted(renames_by_resid)]


def parse_mcpb_tleap_commands(
    path: str | Path,
    *,
    work_dir: str | Path,
    skip_units: Collection[str] = (),
) -> tuple[list[str], list[str]]:
    """Extract only parameter/type setup and bond commands from MCPB.py output.

    ``skip_units`` are units the final build defines itself, whose
    ``loadmol2`` lines are left out.
    """

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
            if line.split(" = loadmol2 ", 1)[0].strip() in skip_units:
                continue
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


def negative_seminario_terms(stdout: str, numbering: McpbNumberingMap) -> list[str]:
    """The Seminario terms MCPB.py found negative and wrote as zero.

    Each is named by its MCPB.py atom types and the atoms the site's types
    stand for, as the structure numbers them, with the value it came out at.
    """

    atoms_by_type: dict[str, list[str]] = {}
    for match in re.finditer(
        r"^\s*\d+-\S+@(\d+)-\S+\s*:\s*\S+\s*-->\s*(\S+)\s*$", stdout, flags=re.MULTILINE
    ):
        atom = numbering.original_by_mcpb_atom_id.get(int(match.group(1)))
        if atom is not None:
            atoms_by_type.setdefault(match.group(2), []).append(
                f"{atom.chain_id or '<blank>'}:{atom.resname}{atom.resid}"
                f"{atom.icode or ''} {atom.name}"
            )
    found: list[str] = []
    term = value = None
    for line in stdout.splitlines():
        heading = re.match(r"\s*For (?:bond|angle|dihedral)type: (\S+?)\.\s", line + " ")
        if heading:
            term, value = heading.group(1), None
            continue
        average = re.match(r"\s*The average force constant is: (\S+)", line)
        if average:
            value = average.group(1)
        if term is not None and "is negative, treat is as 0.00" in line:
            atoms = ", ".join("/".join(atoms_by_type.get(kind, [kind])) for kind in term.split("-"))
            found.append(f"{term} ({atoms}) at {value or 'a negative value'}")
    return found


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
        atom_type = ion_atom_type(ion.element, ion.charge)
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
            (ion.residue.id.chain_id, ion.residue.id.resname, ion.residue.id.resid, ion.residue.id.icode)
        ]
        numbered_atom = replace(
            ion.atom,
            serial=numbered_atom_id,
            # The same element-symbol name the numbered PDB now carries.
            name=ion.element.strip().upper(),
            chain_id="A",
            resid=numbered_resid,
            icode=None,
        )
        numbered_residue = ResidueRecord(
            id=ResidueId("A", ion.element.strip().upper(), numbered_resid),
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
    mol2_paths: list[Path] = []
    frcmod_paths: list[Path] = []
    seen_resnames: set[str] = set()
    for ligand in manifest.ligands:
        # A copy's template is its reference's, staged once under the name.
        if ligand.parameters_from is not None:
            continue
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
    return mol2_paths, frcmod_paths, families


def _stage_kept_residue_parameters(
    manifest: ManifestConfig, *, amberhome: Path, work_dir: Path
) -> tuple[list[Path], list[Path], set[str]]:
    """Each kept residue as a mol2 and an frcmod, for MCPB.py to read.

    MCPB.py tells an amino acid by its atoms, not its name, so a kept
    residue of a chain is modelled as one -- its side chain on the metal in
    the small model, its backbone capping a neighbour's in the large one --
    and its atom types and charges are read from a mol2, as a ligand's are.
    Every kept residue is staged, since the large model reaches the residues
    beside those on the metal. One parameterised from its SMILES has its
    mol2 and frcmod already; one built from an AmberTools template has its
    unit written out by tleap and takes its family's frcmod. A template of
    the protein force field itself, HYP, MCPB.py reads with the rest of it.
    """

    from mdprep.residues.templates import AMBER_RESIDUE_TEMPLATES
    from mdprep.residues.workflow import (
        ResidueStageError,
        final_resname,
        residue_directory,
        template_mol2,
    )

    forcefield = manifest.protein.forcefield
    mol2_paths: list[Path] = []
    frcmod_paths: list[Path] = []
    families: set[str] = set()
    for block in manifest.noncanonical_residues:
        name = final_resname(block)
        if block.parameters == "amber_library":
            assert block.template is not None
            template = AMBER_RESIDUE_TEMPLATES[block.template]
            frcmod_name = template.frcmod_file(forcefield)
            if frcmod_name is None:
                continue
            try:
                exported = template_mol2(
                    template,
                    protein_forcefield=forcefield,
                    work_dir=work_dir / "residue_templates" / name,
                )
            except ResidueStageError as exc:
                raise McpbWorkflowError(str(exc)) from exc
            source_mol2 = exported
            source_frcmod = amberhome / "dat" / "leap" / "parm" / frcmod_name
        else:
            directory = residue_directory(manifest.project.output_dir, block.resname)
            source_mol2 = directory / f"{name}.mol2"
            source_frcmod = directory / f"{name}.frcmod"
            families.add(block.atom_types)
        for source in (source_mol2, source_frcmod):
            if not source.is_file():
                raise McpbWorkflowError(
                    f"Kept residue {name} has no {source.name} to give MCPB.py: {source}"
                )
        mol2_target = work_dir / f"{name}.mol2"
        if source_mol2.resolve() != mol2_target.resolve():
            shutil.copyfile(source_mol2, mol2_target)
        frcmod_target = work_dir / f"residue_{name}.frcmod"
        shutil.copyfile(source_frcmod, frcmod_target)
        mol2_paths.append(mol2_target)
        frcmod_paths.append(frcmod_target)
    return mol2_paths, frcmod_paths, families


def _gaff_version(families: set[str]) -> int:
    """MCPB.py's ``gaff`` setting for the GAFF families staged, of which it takes one."""

    if len(families) > 1:
        raise McpbWorkflowError(
            "MCPB.py accepts one GAFF family per run; bonded metal preparation cannot mix "
            f"GAFF and GAFF2 (the ligands and kept residues use {', '.join(sorted(families))})."
        )
    if not families:
        return 0
    return 1 if next(iter(families)) == "gaff" else 2


def _mcpb_ion_names(
    site: ResolvedMetalSite,
) -> dict[tuple[str, str, int, str | None], str]:
    """The name MCPB.py needs for each of the site's metal ions.

    One name for the residue and its atom alike, because MCPB.py reads the
    charge by the atom name in one place and by the residue name in another,
    and pairs it with the element symbol to find van der Waals parameters.
    """

    return {
        (
            ion.residue.id.chain_id,
            ion.residue.id.resname,
            ion.residue.id.resid,
            ion.residue.id.icode,
        ): (
            ion.element.strip().upper()
        )
        for ion in site.ions
    }


def _write_mcpb_ion_charge_mol2(
    site: ResolvedMetalSite,
    *,
    numbering: McpbNumberingMap,
    work_dir: Path,
) -> list[Path]:
    """Declare each metal ion's charge where MCPB.py will look for it.

    MCPB.py builds its charge table from the force field's amino acids and
    from naa_mol2files, and the Amber force fields carry no ions at all, so an
    ion's charge reaches MCPB.py only through a mol2 listed there. Without one
    it raises KeyError on the ion's atom name while writing the large model's
    ESP input.

    The charge written is the oxidation state declared in the manifest, which
    is the user stating it rather than anything inferred. The residue name is
    the element symbol, matching the atom name used in the numbered PDB, since
    that is the key MCPB.py looks up.
    """

    paths: list[Path] = []
    seen: set[str] = set()
    for ion in site.ions:
        name = ion.element.strip().upper()
        if name in seen:
            continue
        seen.add(name)
        numbered_resid = numbering.mcpb_resid_by_original_key[
            (ion.residue.id.chain_id, ion.residue.id.resname, ion.residue.id.resid, ion.residue.id.icode)
        ]
        numbered_atom = replace(
            ion.atom,
            serial=numbering.mcpb_atom_id_by_original_identity[ion.atom.atom_identity],
            name=name,
            chain_id="A",
            resid=numbered_resid,
            icode=None,
        )
        numbered_residue = ResidueRecord(
            id=ResidueId("A", name, numbered_resid),
            atoms=[numbered_atom],
            record_names={numbered_atom.record_name},
            original_index=numbered_resid - 1,
        )
        parameter = AmberIonParameter(
            element=ion.element,
            charge=ion.charge,
            atom_type=ion_atom_type(ion.element, ion.charge),
            mass=0.0,
            rmin_over_2_angstrom=0.0,
            epsilon_kcal_mol=0.0,
            parameter_set="mcpb_charge_declaration",
            water_model="",
            frcmod_name="",
            frcmod_path=Path("."),
        )
        path = work_dir / f"{name}.mol2"
        write_single_ion_mol2(
            ResolvedMetalIon(
                site_id=site.config.id,
                element=ion.element,
                charge=ion.charge,
                atom=numbered_atom,
                residue=numbered_residue,
            ),
            parameter,
            path,
        )
        paths.append(path)
    return paths


# MCPB.py's frcmod reader takes an atom type from a fixed two-character field
# (pymsmt.lib.readmass is `line[0:2]`), so a three-character Amber ion type
# cannot be expressed at all: Cl- is read as Cl with a mass of "-  35.45", and
# MCPB.py then raises KeyError on Cl- while writing the initial frcmod. F- and
# I- happen to fit; Cl- and Br- do not.
MCPB_MAX_ATOM_TYPE_LENGTH = 2


def mcpb_ion_atom_type(element: str, charge: int) -> str:
    """The atom type to give a coordinating ion inside MCPB.py's input.

    Amber's own type where it fits in MCPB.py's two-character field, and an
    uppercase element alias where it does not. The alias is only a name: it is
    emitted with the same Li/Merz mass and Lennard-Jones parameters as the
    type it stands for, and only inside the MCPB working directory. Everything
    outside -- tleap, the final topology -- keeps Amber's own type.
    """

    amber_type = ion_atom_type(element, charge)
    if len(amber_type) <= MCPB_MAX_ATOM_TYPE_LENGTH:
        return amber_type
    alias = element.strip().upper()
    if len(alias) > MCPB_MAX_ATOM_TYPE_LENGTH:
        raise McpbWorkflowError(
            f"{element} needs an atom type of at most "
            f"{MCPB_MAX_ATOM_TYPE_LENGTH} characters for MCPB.py, and neither "
            f"{amber_type} nor {alias} fits."
        )
    return alias


def _stage_coordinating_ion_frcmods(
    site: ResolvedMetalSite,
    manifest: ManifestConfig,
    *,
    work_dir: Path,
) -> list[Path]:
    """Write the ion parameters MCPB.py needs, under a type it can read.

    The values are Amber's own, read out of the Li/Merz file the final build
    will use, so the alias changes the spelling of the type and nothing about
    the chemistry.
    """

    assert site.config.mcpb is not None
    if not site.config.coordinating_ions:
        return []
    amberhome = resolve_amberhome()
    parameter_set = site.config.mcpb.provisional_nonbonded_parameter_set
    lines = [
        "mdprep: Amber ion parameters under a two-character type for MCPB.py",
        "MASS",
    ]
    nonbonded: list[str] = []
    seen: set[str] = set()
    for ion in site.config.coordinating_ions:
        try:
            parameter = load_amber_ion_parameter(
                element=ion.element,
                charge=ion.charge,
                water_model=manifest.protein.water_model,
                parameter_set=parameter_set,
                amberhome=amberhome,
            )
        except NonbondedMetalError as exc:
            raise McpbWorkflowError(
                f"Coordinating ion {ion.element} has no Amber parameters for "
                f"the {parameter_set} set: {exc}"
            ) from exc
        alias = mcpb_ion_atom_type(ion.element, ion.charge)
        if alias in seen:
            continue
        seen.add(alias)
        lines.append(f"{alias} {parameter.mass:.2f}")
        nonbonded.append(
            f"  {alias}   {parameter.rmin_over_2_angstrom:.4f}  "
            f"{parameter.epsilon_kcal_mol:.8f}   "
            f"{parameter.atom_type} from {parameter.frcmod_name}"
        )
    if not seen:
        return []
    lines.extend(["", "NONBON", *nonbonded, ""])
    path = work_dir / "mdprep_coordinating_ions.frcmod"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [path]


def _write_mcpb_coordinating_ion_mol2(
    site: ResolvedMetalSite,
    *,
    numbering: McpbNumberingMap,
    work_dir: Path,
) -> list[Path]:
    """One-atom mol2 files for the site's coordinating ions.

    MCPB.py asks for a mol2 for every non-standard residue bonded to the
    metal, including an ion Amber already parameterises -- it needs the atom
    type and the charge, not new parameters. antechamber cannot supply one,
    because its semi-empirical step refuses a single atom, so it is written
    directly with the ion atom type tleap will use for the same residue.

    The coordinates come from the site's own bond coordinators, so the mol2
    describes the atom actually in the structure rather than a copy of the
    selector.
    """

    assert site.config.mcpb is not None
    coordinator_by_key = {
        (
            bond.coordinator_residue.id.chain_id,
            bond.coordinator_residue.id.resname,
            bond.coordinator_residue.id.resid,
            bond.coordinator_residue.id.icode,
        ): bond.coordinator_residue
        for bond in site.bonds
    }
    paths: list[Path] = []
    seen_resnames: set[str] = set()
    for ion_config in site.config.coordinating_ions:
        selector = ion_config.selector
        key = (selector.chain, selector.resname, selector.resid, selector.icode)
        residue = coordinator_by_key.get(key)
        if residue is None:
            raise McpbWorkflowError(
                f"Coordinating ion {selector.resname} {selector.resid} is not "
                f"bonded to any ion of metal site {site.config.id}. Add it to "
                "mcpb.bonds, or drop it from coordinating_ions if it is not "
                "part of the site."
            )
        if len(residue.atoms) != 1:
            raise McpbWorkflowError(
                f"A coordinating ion must be a single atom; {residue.id.display()} "
                f"has {len(residue.atoms)}."
            )
        resname = residue.id.resname
        if resname in seen_resnames:
            continue
        seen_resnames.add(resname)

        numbered_resid = numbering.mcpb_resid_by_original_key[key]
        atom = residue.atoms[0]
        numbered_atom = replace(
            atom,
            serial=numbering.mcpb_atom_id_by_original_identity[atom.atom_identity],
            chain_id="A",
            resid=numbered_resid,
            icode=None,
        )
        numbered_residue = ResidueRecord(
            id=ResidueId("A", resname, numbered_resid),
            atoms=[numbered_atom],
            record_names={numbered_atom.record_name},
            original_index=numbered_resid - 1,
        )
        parameter = AmberIonParameter(
            element=ion_config.element,
            charge=ion_config.charge,
            atom_type=mcpb_ion_atom_type(ion_config.element, ion_config.charge),
            mass=0.0,
            rmin_over_2_angstrom=0.0,
            epsilon_kcal_mol=0.0,
            parameter_set="mcpb_input",
            water_model="",
            frcmod_name="",
            frcmod_path=Path("."),
        )
        path = work_dir / f"{resname}.mol2"
        write_single_ion_mol2(
            ResolvedMetalIon(
                site_id=site.config.id,
                element=ion_config.element,
                charge=ion_config.charge,
                atom=numbered_atom,
                residue=numbered_residue,
            ),
            parameter,
            path,
        )
        paths.append(path)
    return paths


# TIP3P, as tleap's WAT unit defines it: the bound water keeps the atom
# types and charges of every other water until MCPB refits the site.
_TIP3P_WATER = {"O": ("OW", -0.834), "H1": ("HW", 0.417), "H2": ("HW", 0.417)}


def _stage_bound_water_frcmod(site: ResolvedMetalSite, *, work_dir: Path) -> list[Path]:
    """The H-O-H angle a water bound to the metal needs and TIP3P never had.

    TIP3P is rigid in Amber: frcmod.tip3p holds its shape with an H-H
    pseudo-bond for SHAKE and defines no angle. MCPB.py gives the bound
    oxygen a new atom type and copies every term the old type had -- so the
    O-H bond came across, and tleap then stopped on HW-Y6-HW, an angle with
    no parameter anywhere. The bound water is no longer a rigid TIP3P
    molecule, so it gets the angle Amber's own parm94/parm99 define for the
    same geometry, and MCPB.py carries it over to the new type like the bond.
    """

    if not any(
        bond.coordinator_residue.id.resname in WATER_RESIDUES
        or bond.coordinator_residue.id.resname == "WAT"
        for bond in site.bonds
    ):
        return []
    path = work_dir / "mdprep_bound_water.frcmod"
    path.write_text(
        "Water bound to a metal: the TIP3P angle from parm94/parm99\n"
        "MASS\n\n"
        "BOND\n\n"
        "ANGLE\n"
        "HW-OW-HW    100.00      104.52    TIP3P water geometry, parm99.dat\n\n"
        "DIHE\n\n"
        "NONBON\n\n",
        encoding="utf-8",
    )
    return [path]


def _write_mcpb_water_mol2(
    site: ResolvedMetalSite,
    *,
    manifest: ManifestConfig,
    work_dir: Path,
) -> list[Path]:
    """A template for a water bound to the metal, which MCPB.py requires.

    MCPB.py treats every non-amino-acid residue bonded to the metal as a
    non-standard residue and asks for its mol2 -- a water included. The
    MCPB.py tutorial makes one with tleap's savemol2; it is written here
    directly instead, from the bound water's own coordinates, for the same
    reason the coordinating-ion template is: the atom types and charges are
    all MCPB needs, and they are TIP3P's.

    A four-point water cannot be a member of the QM model: its extra point
    has no element, no nucleus and no place in a Hessian. So a water bound to
    a bonded metal requires TIP3P, and saying so here beats MCPB.py failing
    on a residue it cannot type.
    """

    waters = []
    seen: set[str] = set()
    for bond in site.bonds:
        residue = bond.coordinator_residue
        if residue.id.resname not in WATER_RESIDUES and residue.id.resname != "WAT":
            continue
        if residue.id.resname in seen:
            continue
        seen.add(residue.id.resname)
        waters.append(residue)
    if not waters:
        return []
    if manifest.protein.water_model != "TIP3P":
        raise McpbWorkflowError(
            f"Metal site {site.config.id} has a water bound to the metal, which "
            "becomes part of the MCPB quantum model. That needs a three-point "
            f"water; {manifest.protein.water_model} has an extra point with no "
            "nucleus. Use TIP3P (with ff14SB) for this system."
        )
    paths: list[Path] = []
    for residue in waters:
        atoms = {atom.name.strip(): atom for atom in residue.atoms}
        missing = sorted(set(_TIP3P_WATER) - set(atoms))
        if missing:
            raise McpbWorkflowError(
                f"Bound water {residue.id.display()} has no {', '.join(missing)}; "
                "it should have been built as a TIP3P water before MCPB."
            )
        resname = residue.id.resname
        lines = [
            "@<TRIPOS>MOLECULE",
            resname,
            "    3     2     1     0     0",
            "SMALL",
            "USER_CHARGES",
            "",
            "",
            "@<TRIPOS>ATOM",
        ]
        for index, name in enumerate(("O", "H1", "H2"), start=1):
            atom = atoms[name]
            atom_type, charge = _TIP3P_WATER[name]
            lines.append(
                f"{index:7d} {name:<4} {atom.x:12.4f}{atom.y:12.4f}{atom.z:12.4f} "
                f"{atom_type:<5}{1:5d} {resname:<6}{charge:12.6f}"
            )
        lines += [
            "@<TRIPOS>BOND",
            "     1     1     2 1",
            "     2     1     3 1",
            "@<TRIPOS>SUBSTRUCTURE",
            f"     1 {resname:<4}        1 TEMP              0 ****  ****    0 ROOT",
            "",
        ]
        path = work_dir / f"{resname}.mol2"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


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
    # A coordinating ion is declared on the site, not under ligands, because
    # it is neither: an ion entry is a metal and a halide is not, while
    # antechamber refuses a single atom so it cannot be a GAFF cofactor.
    # Amber parameterises it directly from its ion library, so it is already
    # as parameterized as it needs to be.
    ligand_keys.update(
        (ion.selector.chain, ion.selector.resid, ion.selector.icode)
        for ion in site.config.coordinating_ions
    )
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
    kept = kept_residue_names(manifest)
    missing = [
        residue.id.display()
        for key, residue in site_residues.items()
        if not is_standard_protein_residue(residue)
        and not is_water_residue(residue)
        and residue.id.resname not in kept
        and key not in ligand_keys
    ]
    if missing:
        raise McpbWorkflowError(
            "Every non-protein/non-water residue in an MCPB.py site must be configured and "
            "parameterized under ligands, or kept under noncanonical_residues; missing: "
            + ", ".join(sorted(missing))
        )


def _validate_site_in_one_chain(
    site: ResolvedMetalSite, *, protein_names: Collection[str] = ()
) -> None:
    """Refuse a site whose protein donors come from more than one chain.

    MCPB.py reads one long chain: mdprep renumbers the structure for it, and
    MCPB.py caps each coordinating residue with the residues numbered either
    side of it, treating two donors a few numbers apart as one stretch of
    backbone. Inside one chain those are real neighbours; across a chain
    boundary they are not, so a site at an interface between two copies is
    not something MCPB.py can model correctly here.
    """

    chains = {
        residue.id.chain_id
        for residue in (
            *(bond.coordinator_residue for bond in site.bonds),
            *site.additional_residues,
        )
        if is_standard_protein_residue(residue) or residue.id.resname in protein_names
    }
    if len(chains) > 1:
        raise McpbWorkflowError(
            f"Metal site {site.config.id!r} is bound by residues of chains "
            f"{', '.join(sorted(chains))}. A bonded MCPB site has to sit within one "
            "protein chain, because MCPB.py caps its residues from their neighbours in "
            "one numbering; model this interface site nonbonded."
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
