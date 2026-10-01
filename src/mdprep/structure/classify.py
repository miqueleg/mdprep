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

# Elements that appear in PDB files as free, single-atom ions. Transition and
# main-group metals that coordinate a protein are separated from the bulk
# counterions so that a manifest planner can ask about them differently: a
# structural Zn needs an explicit oxidation state and a metal-site model, a
# crystallisation Na+ does not.
COORDINATING_METAL_ELEMENTS = {
    "ZN", "FE", "CU", "MN", "CO", "NI", "MG", "CA", "MO", "W", "V", "CD", "HG",
}

BULK_ION_ELEMENTS = {"NA", "K", "LI", "RB", "CS", "CL", "BR", "I", "F"}


def is_standard_protein_residue(residue: ResidueRecord) -> bool:
    # Residue names alone are insufficient: free proline substrates and other
    # amino-acid-like ligands are valid HETATM residues.  Polymer identity is
    # carried by ATOM records (and preserved TER boundaries).
    return (
        residue.id.resname in STANDARD_PROTEIN_RESIDUES
        and "ATOM" in residue.record_names
    )


def is_water_residue(residue: ResidueRecord) -> bool:
    return residue.id.resname in WATER_RESIDUES


def is_histidine(residue: ResidueRecord) -> bool:
    return is_standard_protein_residue(residue) and residue.id.resname in HISTIDINE_RESIDUES


def is_titratable_residue(residue: ResidueRecord) -> bool:
    return is_standard_protein_residue(residue) and residue.id.resname in TITRATABLE_RESIDUES


def is_nonstandard_nonwater_residue(residue: ResidueRecord) -> bool:
    return not is_water_residue(residue) and not is_standard_protein_residue(residue)


def is_likely_ligand_or_cofactor(residue: ResidueRecord) -> bool:
    return is_nonstandard_nonwater_residue(residue)


def likely_ligands_or_cofactors(residues: list[ResidueRecord]) -> list[ResidueRecord]:
    return [residue for residue in residues if is_likely_ligand_or_cofactor(residue)]


def _single_atom_element(residue: ResidueRecord) -> str | None:
    """Element symbol of a one-atom residue, upper-cased, else None."""

    if len(residue.atoms) != 1:
        return None
    atom = residue.atoms[0]
    element = (atom.element or "").strip().upper()
    return element or None


def is_metal_ion_residue(residue: ResidueRecord) -> bool:
    """A free single-atom metal that a metal site would have to describe.

    Bulk counterions are deliberately excluded: they are solvent, and asking a
    user for the oxidation state of every crystallisation Na+ would bury the
    one question that matters.
    """

    if is_water_residue(residue) or is_standard_protein_residue(residue):
        return False
    return _single_atom_element(residue) in COORDINATING_METAL_ELEMENTS


def is_bulk_ion_residue(residue: ResidueRecord) -> bool:
    if is_water_residue(residue) or is_standard_protein_residue(residue):
        return False
    return _single_atom_element(residue) in BULK_ION_ELEMENTS


def metal_ion_residues(residues: list[ResidueRecord]) -> list[ResidueRecord]:
    return [residue for residue in residues if is_metal_ion_residue(residue)]
