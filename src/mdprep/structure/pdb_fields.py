"""What a fixed-column PDB record can express, and how to fit mmCIF into it.

A PDB file gives a residue name three columns, a chain one, an atom name four
and a residue number four.  mmCIF has no such limits, so prediction output
routinely carries names that cannot be written back out: Boltz names a second
SMILES ligand ``LIG2`` and names its chain after the entity rather than with a
letter.

That is not a cosmetic problem.  Everything downstream of mdprep reads fixed
columns -- tleap, antechamber, MCPB.py and RDKit's PDB parser -- so a name
written one character too wide pushes every later field one column right.  The
chain id lands in the residue-number field and the coordinates lose their first
digit, and the first thing to complain is RDKit, about a residue number that now
has half a chain id inside it.

So wide names are mapped onto expressible ones while the structure is being
read, before any other part of mdprep sees them, and the mapping is reported so
a selector written against the original file can be translated by hand.
"""

from __future__ import annotations

from dataclasses import replace

from mdprep.structure.classify import (
    HISTIDINE_RESIDUES,
    STANDARD_PROTEIN_RESIDUES,
    TITRATABLE_RESIDUES,
    WATER_RESIDUES,
)
from mdprep.structure.models import AtomRecord
from mdprep.structure.pdb import PERIODIC_ELEMENTS

RESNAME_WIDTH = 3
CHAIN_ID_WIDTH = 1
ATOM_NAME_WIDTH = 4
RESID_MIN = -999
RESID_MAX = 9999

# Chain ids a PDB file can hold, in the order they are handed out.
CHAIN_ID_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" "abcdefghijklmnopqrstuvwxyz" "0123456789"
)

# A renamed ligand must not land on a name that means something to mdprep:
# classification is by residue name, so a ligand called WAT becomes solvent and
# one called HIS becomes a titratable protein residue.
RESERVED_RESNAMES = frozenset(
    STANDARD_PROTEIN_RESIDUES | WATER_RESIDUES | TITRATABLE_RESIDUES | HISTIDINE_RESIDUES
)


class PdbFieldError(ValueError):
    """A value cannot be written into the PDB column it belongs in."""


def looks_like_ion_resname(name: str) -> bool:
    """Would this residue name be read as a monatomic ion?

    Amber names ion residues after the element and its oxidation state -- FE2,
    CU1, ZN -- and mdprep's element inference strips the digits and compares
    what is left against the periodic table.  A renamed ligand that happened to
    match would silently become a metal site.
    """

    stripped = name.upper().rstrip("0123456789+-")
    return bool(stripped) and stripped in PERIODIC_ELEMENTS


def fit_atoms_to_pdb_fields(
    atoms: list[AtomRecord],
) -> tuple[list[AtomRecord], list[str]]:
    """Rename whatever is too wide for a PDB record, and say what was renamed.

    Residue numbers and atom names are not renamed: a number has no spare
    spelling and an atom name is matched against force-field templates, so
    both are reported as errors instead.
    """

    warnings: list[str] = []
    chain_map = _chain_id_map(atoms, warnings)
    resname_map = _resname_map(atoms, warnings)
    _check_unmappable_fields(atoms)

    if not chain_map and not resname_map:
        return atoms, warnings

    return [
        replace(
            atom,
            chain_id=chain_map.get(atom.chain_id, atom.chain_id),
            resname=resname_map.get(atom.resname, atom.resname),
            # The original line is a PDB line, and these no longer came from
            # one. Readers fall back to the fields when it is empty.
            original_line="",
        )
        for atom in atoms
    ], warnings


def _chain_id_map(atoms: list[AtomRecord], warnings: list[str]) -> dict[str, str]:
    present = list(dict.fromkeys(atom.chain_id for atom in atoms))
    wide = [chain for chain in present if len(chain) > CHAIN_ID_WIDTH]
    if not wide:
        return {}

    taken = {chain for chain in present if len(chain) <= CHAIN_ID_WIDTH and chain.strip()}
    mapping: dict[str, str] = {}
    for chain in wide:
        # Its own initial first, so A_heavy stays near A and a reader can still
        # guess which chain is which.
        candidates = [character for character in chain if character.isalnum()]
        candidates.extend(CHAIN_ID_ALPHABET)
        replacement = next(
            (candidate for candidate in candidates if candidate not in taken), None
        )
        if replacement is None:
            raise PdbFieldError(
                f"Chain {chain!r} needs a single-character PDB chain id, but all "
                f"{len(CHAIN_ID_ALPHABET)} of them are already used by this "
                "structure. mdprep cannot write it as a PDB file."
            )
        taken.add(replacement)
        mapping[chain] = replacement
        warnings.append(
            f"Chain {chain!r} is wider than the single character a PDB file "
            f"allows, so mdprep renamed it to {replacement!r}."
        )
    return mapping


def _resname_map(atoms: list[AtomRecord], warnings: list[str]) -> dict[str, str]:
    present = list(dict.fromkeys(atom.resname for atom in atoms))
    wide = [resname for resname in present if len(resname) > RESNAME_WIDTH]
    if not wide:
        return {}

    taken = {resname for resname in present if len(resname) <= RESNAME_WIDTH}
    taken |= RESERVED_RESNAMES
    mapping: dict[str, str] = {}
    for resname in wide:
        replacement = _free_resname(resname, taken)
        taken.add(replacement)
        mapping[resname] = replacement
        warnings.append(
            f"Residue name {resname!r} is longer than the three characters a PDB "
            f"file allows, so mdprep renamed it to {replacement!r}. Selectors in "
            f"the manifest use {replacement!r}."
        )
    return mapping


def ion_element(resname: str) -> str | None:
    """The element an ion residue name is built from, if it is one.

    Amber names ion residues after the element and its oxidation state -- ZN,
    FE2, CU1 -- so the name is not a label but the only record of which metal
    this is.
    """

    stripped = resname.upper().rstrip("0123456789+-")
    return stripped if stripped and stripped in PERIODIC_ELEMENTS else None


def _free_resname(resname: str, taken: set[str]) -> str:
    """Pick an unused three-character stand-in for one wide residue name.

    The first three characters when they are free, because LIG2 reading as LIG
    is obvious to a student; a numbered fallback otherwise.
    """

    # Reserved here rather than only in the caller's set, so the function is
    # safe to call on its own: it is the one deciding what a residue is named,
    # and a name that means something to mdprep is never a free choice.
    unavailable = set(taken) | RESERVED_RESNAMES

    # An ion keeps its element. Shortening ZN22 to Z01 would fit the column
    # and throw away the fact that it is zinc, which is what tleap matches
    # against its ion templates and what mdprep reads the metal from.
    element = ion_element(resname)
    if element is not None and len(element) <= RESNAME_WIDTH:
        room = RESNAME_WIDTH - len(element)
        suffixes = [""] + [str(number) for number in range(1, 10)] if room else [""]
        for suffix in suffixes:
            candidate = element + suffix
            if candidate not in unavailable:
                return candidate
        raise PdbFieldError(
            f"Ion residue name {resname!r} does not fit three characters, and "
            f"every name based on {element} is already in use."
        )

    def usable(candidate: str) -> bool:
        return candidate not in unavailable and not looks_like_ion_resname(candidate)

    head = resname[:RESNAME_WIDTH].upper()
    if usable(head):
        return head
    initial = head[0] if head[:1].isalpha() else "L"
    for prefix in (initial, "L", "X"):
        for number in range(1, 100):
            candidate = f"{prefix}{number:02d}"
            if usable(candidate):
                return candidate
    raise PdbFieldError(
        f"Residue name {resname!r} does not fit the three characters a PDB file "
        "allows, and mdprep could not find an unused replacement."
    )


def _check_unmappable_fields(atoms: list[AtomRecord]) -> None:
    for atom in atoms:
        if len(atom.name) > ATOM_NAME_WIDTH:
            raise PdbFieldError(
                f"Atom name {atom.name!r} in residue {atom.resname}{atom.resid} is "
                f"longer than the {ATOM_NAME_WIDTH} characters a PDB file allows. "
                "mdprep cannot shorten it, because atom names are matched against "
                "force-field templates."
            )
        if not RESID_MIN <= atom.resid <= RESID_MAX:
            raise PdbFieldError(
                f"Residue number {atom.resid} is outside the range a PDB file can "
                f"hold ({RESID_MIN} to {RESID_MAX}). Renumber the residues in the "
                "input structure."
            )
