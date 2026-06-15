"""Residue classification helpers."""

from __future__ import annotations

from mdprep.structure.models import ResidueRecord


STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "LYN",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

WATER_RESIDUES = {"HOH", "WAT", "H2O", "TIP3", "OPC"}

TITRATABLE_RESIDUES = {
    "ASP",
    "ASH",
    "GLU",
    "GLH",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "LYS",
    "LYN",
    "ARG",
    "CYS",
    "CYM",
    "CYX",
}

HISTIDINE_RESIDUES = {"HIS", "HID", "HIE", "HIP"}


def is_standard_protein_residue(residue: ResidueRecord) -> bool:
    return residue.id.resname in STANDARD_PROTEIN_RESIDUES


def is_water_residue(residue: ResidueRecord) -> bool:
    return residue.id.resname in WATER_RESIDUES


def is_histidine(residue: ResidueRecord) -> bool:
    return residue.id.resname in HISTIDINE_RESIDUES


def is_titratable_residue(residue: ResidueRecord) -> bool:
    return residue.id.resname in TITRATABLE_RESIDUES


def is_likely_ligand_or_cofactor(residue: ResidueRecord) -> bool:
    return (
        "HETATM" in residue.record_names
        and not is_water_residue(residue)
        and not is_standard_protein_residue(residue)
    )


def likely_ligands_or_cofactors(residues: list[ResidueRecord]) -> list[ResidueRecord]:
    return [residue for residue in residues if is_likely_ligand_or_cofactor(residue)]

