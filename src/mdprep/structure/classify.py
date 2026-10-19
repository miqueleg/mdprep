"""Residue classification helpers."""

from __future__ import annotations

from typing import Sequence

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


# Residue names tleap already answers to once the protein, water and ion force
# fields are sourced.  `loadpdb` resolves every residue by name against those
# templates, and for an amino-acid name it maps the name to its terminal form
# first -- a lysine that opens a chain becomes NLYS -- which nothing a manifest
# loads can intercept.  So a free amino acid declared as a ligand is built as a
# piece of protein: tleap stops on the first ligand-named hydrogen it cannot
# type, and `LYS = <ligand unit>` would meanwhile point every real lysine in
# the protein at the ligand.
#
# It lives here, beside the sets it is built from, because two readers need
# it: the tleap builder, which refuses such a ligand, and the decision
# planner, which refuses the structure before asking the user anything.
_NUCLEIC_RESIDUES = (
    {f"D{base}{end}" for base in "ACGT" for end in ("", "3", "5", "N")}
    | {f"R{base}{end}" for base in "ACGU" for end in ("", "3", "5", "N")}
    | set("ACGTU")
)

_AMBER_ION_RESIDUES = (
    COORDINATING_METAL_ELEMENTS
    | BULK_ION_ELEMENTS
    | {
        f"{element}{charge}"
        for element in COORDINATING_METAL_ELEMENTS
        for charge in ("1", "2", "3")
    }
    | {f"{element}+" for element in BULK_ION_ELEMENTS}
    | {f"{element}-" for element in BULK_ION_ELEMENTS}
)

LEAP_RESERVED_RESIDUE_NAMES = frozenset(
    STANDARD_PROTEIN_RESIDUES
    | {f"{end}{name}" for end in ("N", "C") for name in STANDARD_PROTEIN_RESIDUES}
    | WATER_RESIDUES
    | _NUCLEIC_RESIDUES
    | _AMBER_ION_RESIDUES
)


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


def reserved_residue_name_reason(resname: str) -> str:
    """Which force field template claims ``resname``, in words.

    A ligand that collides with an amino acid is a different mistake from a
    strand of DNA read as a pile of ligands, and the advice differs, so the
    refusal says which one it found rather than only that a name was taken.
    """

    name = resname.strip().upper()
    if name in STANDARD_PROTEIN_RESIDUES or name in {
        f"{end}{residue}"
        for end in ("N", "C")
        for residue in STANDARD_PROTEIN_RESIDUES
    }:
        return "an amino acid"
    if name in WATER_RESIDUES:
        return "water"
    if name in _NUCLEIC_RESIDUES:
        return "a nucleic acid residue"
    if name in _AMBER_ION_RESIDUES:
        return "a monatomic ion"
    return "a force field residue"


def describe_reserved_residue_names(collisions: Sequence[tuple[str, str]]) -> str:
    """The message both mdprep and the GUI show for a reserved ligand name.

    One wording, written once, so the page and the command line cannot tell
    the user two different things about the same structure.
    """

    listed = ", ".join(
        f"{resname} ({reserved_residue_name_reason(resname)}; {label})"
        for label, resname in collisions
    )
    many = len(collisions) > 1
    reasons = {reserved_residue_name_reason(resname) for _, resname in collisions}
    message = (
        f"Residue name{'s' if many else ''} the Amber force fields already "
        f"define: {listed}. tleap resolves every residue by name against the "
        "force field templates, so it would build "
        f"{'these residues' if many else 'this residue'} from that template "
        "instead of from parameters of its own."
    )
    if "an amino acid" in reasons:
        message += (
            " An amino-acid name would also redirect every matching residue in "
            "the protein to it."
        )
    message += (
        f" Rename {'them' if many else 'it'} in the input structure to a code "
        "no force field uses (LIG, or any unused three-character name), then "
        "run mdprep again."
    )
    if any(
        reserved_residue_name_reason(resname) == "a nucleic acid residue"
        for _, resname in collisions
    ):
        message += (
            " If this structure contains DNA or RNA, renaming will not help: "
            "mdprep prepares proteins, and has no nucleic acid force field to "
            "build those residues with."
        )
    return message


def free_residue_name(taken: Sequence[str] | set[str]) -> str:
    """A three-character residue code no force field and no caller is using.

    Offered to the user as the replacement for a name the force fields own.
    LIG first, because it is what a reader expects an unparameterised
    heterogen to be called.
    """

    unavailable = {name.strip().upper() for name in taken} | set(
        LEAP_RESERVED_RESIDUE_NAMES
    )
    candidates = ["LIG"] + [f"LG{digit}" for digit in "123456789"]
    candidates += [f"L{index:02d}" for index in range(100)]
    for candidate in candidates:
        if candidate not in unavailable:
            return candidate
    raise ValueError("No free three-character residue name is left")
