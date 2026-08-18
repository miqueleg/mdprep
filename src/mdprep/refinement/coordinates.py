"""Transfer optimized Amber coordinates back to stable input residue identities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

from mdprep.structure.classify import is_standard_protein_residue
from mdprep.refinement.mapping import (
    RefinementMappingError,
    compatible_resnames,
    is_virtual_site,
    map_residues_by_coordinates,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord


class CoordinateTransferError(ValueError):
    """Raised when optimized coordinates cannot be mapped without guessing."""


@dataclass(frozen=True)
class CoordinateTransferResult:
    structure: PdbStructure
    restored_residue_names: tuple[dict[str, object], ...]
    generated_hydrogen_count: int
    excluded_virtual_site_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "restored_residue_names": list(self.restored_residue_names),
            "generated_hydrogen_count": self.generated_hydrogen_count,
            "excluded_virtual_site_count": self.excluded_virtual_site_count,
            "output_atom_count": len(self.structure.atoms),
        }


def transfer_optimized_coordinates(
    *,
    reference_structure: PdbStructure,
    leap_input_structure: PdbStructure,
    topology_structure: PdbStructure,
    coordinates_angstrom: tuple[tuple[float, float, float], ...],
    output_path: str | Path,
) -> CoordinateTransferResult:
    """Return a hydrogenated optimized structure with pre-protonation names.

    tleap-generated hydrogens are retained here. With
    ``post_refinement_protonation: reuse_initial`` they remain the authoritative
    optimized coordinates for subsequent MCPB/charge/final-tleap stages. With
    the legacy ``rerun`` policy, the normal protonation stage may remove them
    before final parameterization.
    """

    if len(coordinates_angstrom) != len(topology_structure.atoms):
        raise CoordinateTransferError(
            "Optimized coordinate count does not match provisional tleap PDB atom count: "
            f"{len(coordinates_angstrom)} != {len(topology_structure.atoms)}"
        )
    try:
        reference_to_leap = map_residues_by_coordinates(
            reference_structure,
            leap_input_structure,
        )
        leap_to_topology = map_residues_by_coordinates(
            leap_input_structure,
            topology_structure,
        )
    except RefinementMappingError as exc:
        raise CoordinateTransferError(str(exc)) from exc
    leap_pair_by_input_index = {
        pair.input_index: pair for pair in leap_to_topology
    }

    topology_atom_index = {
        id(atom): index for index, atom in enumerate(topology_structure.atoms)
    }
    new_atoms: list[AtomRecord] = []
    old_to_new_serial: dict[int, int] = {}
    residue_last_serial: dict[int, int] = {}
    restored_names: list[dict[str, object]] = []
    generated_hydrogens = 0
    excluded_virtual_sites = 0

    for residue_index, reference_pair in enumerate(reference_to_leap):
        reference = reference_pair.input_residue
        leap_input = reference_pair.topology_residue
        topology_pair = leap_pair_by_input_index[reference_pair.topology_index]
        topology = topology_pair.topology_residue
        if not compatible_resnames(reference.id.resname, leap_input.id.resname):
            raise CoordinateTransferError(
                f"Residue {residue_index + 1} changed identity before tleap: "
                f"{reference.id.display()} -> {leap_input.id.display()}"
            )
        if not compatible_resnames(leap_input.id.resname, topology.id.resname):
            raise CoordinateTransferError(
                f"Residue {residue_index + 1} changed identity in tleap: "
                f"{leap_input.id.display()} -> {topology.id.display()}"
            )
        if reference.id.resname != leap_input.id.resname:
            restored_names.append(
                {
                    "chain_id": reference.id.chain_id,
                    "resid": reference.id.resid,
                    "icode": reference.id.icode,
                    "optimized_provisional_resname": leap_input.id.resname,
                    "restored_pre_protonation_resname": reference.id.resname,
                    "reason": "rerun requested protonation protocol after QM/MM refinement",
                }
            )
        reference_by_name = {atom.name: atom for atom in reference.atoms}
        if len(reference_by_name) != len(reference.atoms):
            raise CoordinateTransferError(
                f"Reference residue {reference.id.display()} has duplicate atom names"
            )
        record_name = "ATOM" if is_standard_protein_residue(reference) else next(
            iter(reference.record_names), "HETATM"
        )
        for topology_atom in topology.atoms:
            atom_index = topology_atom_index[id(topology_atom)]
            if is_virtual_site(topology_atom):
                excluded_virtual_sites += 1
                continue
            x, y, z = coordinates_angstrom[atom_index]
            reference_atom = reference_by_name.get(topology_atom.name)
            serial = len(new_atoms) + 1
            if reference_atom is not None:
                atom_record_name = reference_atom.record_name
                occupancy = reference_atom.occupancy
                bfactor = reference_atom.bfactor
                original_line = reference_atom.original_line
                if reference_atom.serial is not None:
                    old_to_new_serial[reference_atom.serial] = serial
            else:
                atom_record_name = record_name
                occupancy = 1.0
                bfactor = 0.0
                original_line = ""
                if (topology_atom.element or "").strip().upper() == "H":
                    generated_hydrogens += 1
            new_atoms.append(
                replace(
                    topology_atom,
                    serial=serial,
                    resname=reference.id.resname,
                    chain_id=reference.id.chain_id,
                    resid=reference.id.resid,
                    icode=reference.id.icode,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=occupancy,
                    bfactor=bfactor,
                    record_name=atom_record_name,  # type: ignore[arg-type]
                    original_line=original_line,
                )
            )
            residue_last_serial[residue_index] = serial

    conect_bonds = {
        (min(old_to_new_serial[left], old_to_new_serial[right]),
         max(old_to_new_serial[left], old_to_new_serial[right]))
        for left, right in reference_structure.conect_bonds
        if left in old_to_new_serial and right in old_to_new_serial
    }
    reference_residue_index_by_serial = {
        atom.serial: index
        for index, residue in enumerate(reference_structure.residues)
        for atom in residue.atoms
        if atom.serial is not None
    }
    ter_after_serials = {
        residue_last_serial[reference_residue_index_by_serial[serial]]
        for serial in reference_structure.ter_after_serials
        if serial in reference_residue_index_by_serial
        and reference_residue_index_by_serial[serial] in residue_last_serial
    }
    output = Path(output_path)
    structure = PdbStructure(
        path=output,
        atoms=new_atoms,
        residues=_build_residues(new_atoms),
        model_count=1,
        used_model=1,
        warnings=list(reference_structure.warnings),
        conect_bonds=conect_bonds,
        ter_after_serials=ter_after_serials,
    )
    return CoordinateTransferResult(
        structure=structure,
        restored_residue_names=tuple(restored_names),
        generated_hydrogen_count=generated_hydrogens,
        excluded_virtual_site_count=excluded_virtual_sites,
    )


def _build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    return [
        ResidueRecord(
            id=ResidueId(chain_id=chain, resname=resname, resid=resid, icode=icode),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, ((chain, resname, resid, icode), residue_atoms) in enumerate(grouped.items())
    ]
