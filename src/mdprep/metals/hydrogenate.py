"""Dry tleap hydrogenation required before bonded MCPB.py model generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from math import dist
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.leap.forcefields import forcefield_sources
from mdprep.leap.log_parser import assert_tleap_success
from mdprep.leap.residues import (
    append_disulfide_conect_records,
    disulfide_bond_commands,
    prepare_leap_input_pdb,
    validate_ligand_parameter_files,
    validate_tleap_ligand_coordinates,
)
from mdprep.leap.runner import TLeapRun, run_tleap
from mdprep.ligands.workflow import LigandStageResult
from mdprep.metals.coordination import resolve_metal_sites
from mdprep.protonation.apply import ProtonationResult, is_hydrogen_atom
from mdprep.structure.classify import WATER_RESIDUES
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


class MetalHydrogenationError(ValueError):
    """Raised when the pre-MCPB hydrogenated structure is unsafe or incomplete."""


@dataclass(frozen=True)
class MetalHydrogenationResult:
    input_pdb_path: Path
    script_path: Path
    raw_output_pdb_path: Path
    restored_output_pdb_path: Path
    prmtop_path: Path
    inpcrd_path: Path
    run: TLeapRun
    structure: PdbStructure
    donor_protonation_checks: tuple[dict[str, object], ...]
    maximum_input_atom_coordinate_deviation_angstrom: float

    def to_dict(self) -> dict[str, object]:
        return {
            "input_pdb_path": str(self.input_pdb_path),
            "script_path": str(self.script_path),
            "raw_output_pdb_path": str(self.raw_output_pdb_path),
            "restored_output_pdb_path": str(self.restored_output_pdb_path),
            "prmtop_path": str(self.prmtop_path),
            "inpcrd_path": str(self.inpcrd_path),
            "tleap": self.run.to_dict(),
            "donor_protonation_checks": list(self.donor_protonation_checks),
            "maximum_input_atom_coordinate_deviation_angstrom": (
                self.maximum_input_atom_coordinate_deviation_angstrom
            ),
        }


def hydrogenate_for_mcpb(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    ligand_result: LigandStageResult,
    protonation_result: ProtonationResult,
    metal_setup_commands: list[str],
    output_dir: str | Path,
) -> MetalHydrogenationResult:
    # Local import prevents a module-level cycle: the final tleap builder also
    # consumes MetalStageResult.
    from mdprep.leap.builder import TLeapOutputs, build_tleap_script

    work = Path(output_dir)
    input_dir = work / "input"
    work.mkdir(parents=True, exist_ok=True)
    leap_input = prepare_leap_input_pdb(
        structure,
        input_dir / "system.pre_mcpb.pdb",
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
    disulfides = disulfide_bond_commands(
        structure=leap_input.structure,
        protonation_result=protonation_result,
    )
    append_disulfide_conect_records(leap_input.path, disulfides)
    outputs = TLeapOutputs(
        prmtop=work / "system.pre_mcpb.prmtop",
        inpcrd=work / "system.pre_mcpb.inpcrd",
        pdb=work / "system.pre_mcpb.raw.pdb",
    )
    script_path = work / "tleap.in"
    script_path.write_text(
        build_tleap_script(
            sources=sources,
            ligands=ligand_files,
            input_pdb=leap_input.path,
            disulfide_bonds=disulfides,
            outputs=outputs,
            work_dir=work,
            setup_commands=metal_setup_commands,
        ),
        encoding="utf-8",
    )
    run = run_tleap(script_path, work_dir=work)
    assert_tleap_success(
        run.summary,
        fail_on_warnings=manifest.validation.fail_on_warnings,
        context="pre-MCPB hydrogenation",
    )
    for path in (outputs.prmtop, outputs.inpcrd, outputs.pdb):
        if not path.is_file() or path.stat().st_size == 0:
            raise MetalHydrogenationError(f"Pre-MCPB tleap did not produce required output: {path}")
    raw = read_pdb(outputs.pdb)
    restored_path = work / "system.pre_mcpb.hydrogenated.pdb"
    restored = _restore_residue_identities(
        raw,
        reference=leap_input.structure,
        output_path=restored_path,
    )
    maximum_coordinate_deviation = _validate_preserved_input_coordinates(
        structure,
        restored,
    )
    validate_tleap_ligand_coordinates(
        manifest=manifest,
        reference_structure=structure,
        output_pdb=restored_path,
        stage="pre-MCPB hydrogenation",
    )
    checks = _validate_histidine_donors(restored, manifest)
    return MetalHydrogenationResult(
        input_pdb_path=leap_input.path,
        script_path=script_path,
        raw_output_pdb_path=outputs.pdb,
        restored_output_pdb_path=restored_path,
        prmtop_path=outputs.prmtop,
        inpcrd_path=outputs.inpcrd,
        run=run,
        structure=restored,
        donor_protonation_checks=tuple(checks),
        maximum_input_atom_coordinate_deviation_angstrom=(
            maximum_coordinate_deviation
        ),
    )


def _validate_preserved_input_coordinates(
    reference: PdbStructure,
    observed: PdbStructure,
    *,
    tolerance_angstrom: float = 0.002,
) -> float:
    observed_by_identity = {
        _preservation_identity(atom): atom for atom in observed.atoms
    }
    maximum = 0.0
    for atom in reference.atoms:
        candidate = observed_by_identity.get(_preservation_identity(atom))
        if candidate is None:
            raise MetalHydrogenationError(
                "Pre-MCPB tleap removed an input atom before MCPB.py: "
                f"{atom.atom_identity}."
            )
        deviation = dist(
            (atom.x, atom.y, atom.z),
            (candidate.x, candidate.y, candidate.z),
        )
        maximum = max(maximum, deviation)
        if deviation > tolerance_angstrom:
            raise MetalHydrogenationError(
                "Pre-MCPB tleap moved an existing atom by "
                f"{deviation:.6f} A: {atom.atom_identity}. Refined hydrogen "
                "coordinates must be preserved for the MCPB/QMMESP model."
            )
    return maximum


def _preservation_identity(
    atom: AtomRecord,
) -> tuple[str, str, int, str | None, str]:
    # tleap standardizes retained HOH/H2O/TIP3/OPC input residues to WAT.
    # This is a representation rename, not atom loss, so compare waters using
    # one canonical residue name while preserving chain/residue/atom identity.
    resname = "WAT" if atom.resname in WATER_RESIDUES else atom.resname
    return (atom.chain_id, resname, atom.resid, atom.icode, atom.name)


def _restore_residue_identities(
    observed: PdbStructure,
    *,
    reference: PdbStructure,
    output_path: Path,
) -> PdbStructure:
    if len(observed.residues) != len(reference.residues):
        raise MetalHydrogenationError(
            "Pre-MCPB tleap changed the residue count during hydrogenation: "
            f"{len(reference.residues)} -> {len(observed.residues)}."
        )
    atoms: list[AtomRecord] = []
    serial = 0
    for expected, actual in zip(reference.residues, observed.residues, strict=True):
        if actual.id.resname != expected.id.resname:
            raise MetalHydrogenationError(
                f"Pre-MCPB tleap unexpectedly renamed {expected.id.display()} to "
                f"{actual.id.resname}; residue-state changes must be resolved before this stage."
            )
        expected_names = set(expected.atom_names())
        actual_names = set(actual.atom_names())
        if not expected_names.issubset(actual_names):
            missing = sorted(expected_names - actual_names)
            raise MetalHydrogenationError(
                f"Pre-MCPB tleap removed atoms from {expected.id.display()}: {missing}"
            )
        expected_by_name = {atom.name: atom for atom in expected.atoms}
        residue_record_name = (
            "HETATM" if expected.record_names == {"HETATM"} else "ATOM"
        )
        for atom in actual.atoms:
            serial += 1
            reference_atom = expected_by_name.get(atom.name)
            atoms.append(
                replace(
                    atom,
                    serial=serial,
                    chain_id=expected.id.chain_id,
                    resid=expected.id.resid,
                    icode=expected.id.icode,
                    element=(
                        reference_atom.element
                        if reference_atom is not None
                        else atom.element
                    ),
                    record_name=(
                        reference_atom.record_name
                        if reference_atom is not None
                        else residue_record_name
                    ),
                    occupancy=(
                        reference_atom.occupancy
                        if reference_atom is not None
                        else atom.occupancy
                    ),
                    bfactor=(
                        reference_atom.bfactor
                        if reference_atom is not None
                        else atom.bfactor
                    ),
                )
            )
    structure = PdbStructure(
        path=output_path,
        atoms=atoms,
        residues=_build_residues(atoms),
        model_count=1,
        used_model=1,
        warnings=list(reference.warnings),
    )
    write_pdb(structure, output_path)
    return structure


def _validate_histidine_donors(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    sites = resolve_metal_sites(structure, manifest, validate_mcpb_cutoff=True)
    for site in sites:
        if site.config.model != "bonded_mcpb":
            continue
        for bond in site.bonds:
            residue = bond.coordinator_residue
            donor = bond.coordinator
            if residue.id.resname not in {"HID", "HIE"}:
                continue
            expected_donor = "NE2" if residue.id.resname == "HID" else "ND1"
            protonated_nitrogen = "ND1" if residue.id.resname == "HID" else "NE2"
            if donor.name != expected_donor:
                raise MetalHydrogenationError(
                    f"Hydrogenated {residue.id.display()} is incompatible with declared donor "
                    f"{donor.name}; {residue.id.resname} requires {expected_donor}."
                )
            donor_h = _nearby_hydrogens(donor, residue)
            other = next(
                atom for atom in residue.atoms if atom.name == protonated_nitrogen
            )
            other_h = _nearby_hydrogens(other, residue)
            ok = donor_h == 0 and other_h == 1
            check = {
                "metal_site_id": site.config.id,
                "residue": residue.id.to_dict(),
                "state": residue.id.resname,
                "donor_atom": donor.name,
                "donor_hydrogens_within_1.25A": donor_h,
                "protonated_nitrogen": protonated_nitrogen,
                "protonated_nitrogen_hydrogens_within_1.25A": other_h,
                "ok": ok,
            }
            checks.append(check)
            if not ok:
                raise MetalHydrogenationError(
                    f"Pre-MCPB tleap produced an invalid metal-binding tautomer for "
                    f"{residue.id.display()}: donor {donor.name} has {donor_h} nearby H and "
                    f"{protonated_nitrogen} has {other_h}; expected 0 and 1."
                )
    return checks


def _nearby_hydrogens(atom: AtomRecord, residue: ResidueRecord) -> int:
    return sum(
        1
        for candidate in residue.atoms
        if is_hydrogen_atom(candidate)
        and dist((atom.x, atom.y, atom.z), (candidate.x, candidate.y, candidate.z)) <= 1.25
    )


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
