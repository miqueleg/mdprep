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
# Columns 31-54 give each coordinate eight characters at three decimals, so
# the sign costs a digit: -999.999 fits and -1000.0 does not.
COORDINATE_MIN = -999.999
COORDINATE_MAX = 9999.999

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
) -> tuple[list[AtomRecord], list[str], frozenset[str]]:
    """Rename whatever is too wide for a PDB record, and say what was renamed.

    Residue numbers and atom names are not renamed: a number has no spare
    spelling and an atom name is matched against force-field templates, so
    both are reported as errors instead.
    """

    warnings: list[str] = []
    chain_map = _chain_id_map(atoms, warnings)
    resname_map = _resname_map(atoms, warnings)
    _check_unmappable_fields(atoms)

    assigned = frozenset(resname_map.values())
    if not chain_map and not resname_map:
        return settle_monatomic_ion_elements(atoms), warnings, assigned

    return settle_monatomic_ion_elements([
        replace(
            atom,
            chain_id=chain_map.get(atom.chain_id, atom.chain_id),
            resname=resname_map.get(atom.resname, atom.resname),
            # The original line is a PDB line, and these no longer came from
            # one. Readers fall back to the fields when it is empty.
            original_line="",
        )
        for atom in atoms
    ]), warnings, assigned


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
    monatomic = _monatomic_elements(atoms)
    # LIG2 and LIG3 share their first three characters, and letting whichever
    # came first take the bare LIG leaves a reader unable to say which one it
    # was. When a head is contested nobody gets it.
    heads: dict[str, int] = {}
    for resname in wide:
        head = resname[:RESNAME_WIDTH].upper()
        heads[head] = heads.get(head, 0) + 1
    mapping: dict[str, str] = {}
    for resname in wide:
        element = monatomic.get(resname)
        replacement = _free_resname(
            resname,
            taken,
            element=element,
            allow_head=heads[resname[:RESNAME_WIDTH].upper()] == 1,
        )
        taken.add(replacement)
        mapping[resname] = replacement
        reason = (
            f" Named after the element of its only atom, because a metal is "
            f"identified by its residue name from here on."
            if element is not None
            else ""
        )
        warnings.append(
            f"Residue name {resname!r} is longer than the three characters a PDB "
            f"file allows, so mdprep renamed it to {replacement!r}.{reason} "
            f"Selectors in the manifest use {replacement!r}."
        )
    return mapping


def settle_monatomic_ion_elements(atoms: list[AtomRecord]) -> list[AtomRecord]:
    """Give a one-atom ion the element its residue name states.

    The atom's own name is not enough. A residue FE holding an atom called
    FE1 is iron, but by the column convention the name says fluorine, and
    tleap writes the element column from the Amber type -- F2+ -- which
    says fluorine too. Both halves agree on the wrong answer, so the iron
    stopped matching itself across tleap and the refinement could not map
    it. The residue name is the one place that still says Fe, and for a
    residue holding a single atom it cannot be describing anything else.
    """

    counts: dict[tuple[str, str, int, str | None], int] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.resname, atom.resid, atom.icode)
        counts[key] = counts.get(key, 0) + 1
    settled: list[AtomRecord] = []
    for atom in atoms:
        key = (atom.chain_id, atom.resname, atom.resid, atom.icode)
        element = ion_element(atom.resname) if counts[key] == 1 else None
        if element is not None:
            # Always written in the canonical spelling, not only when it
            # disagrees: the same ion arriving as FE from one reader and Fe
            # from another is how two paths end up comparing unequal.
            atom = replace(atom, element=_canonical(element))
        settled.append(atom)
    return settled


def _canonical(symbol: str) -> str:
    return symbol[0].upper() + symbol[1:].lower()


def _monatomic_elements(atoms: list[AtomRecord]) -> dict[str, str]:
    """Residue names that always hold exactly one atom, and of what element.

    Prediction output names its non-polymers by position rather than by
    chemistry -- Boltz calls the iron LIG1 and the cofactor beside it LIG2 --
    so the name carries nothing worth preserving, while the single atom
    inside says exactly what the residue is. Everything downstream reads a
    metal out of its residue name, so that is the name to give it.
    """

    composition: dict[str, set[tuple[int, str]]] = {}
    counts: dict[tuple[str, str, int, str | None], int] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.resname, atom.resid, atom.icode)
        counts[key] = counts.get(key, 0) + 1
    for atom in atoms:
        key = (atom.chain_id, atom.resname, atom.resid, atom.icode)
        element = (atom.element or "").strip()
        composition.setdefault(atom.resname, set()).add(
            (counts[key], element.upper())
        )
    result: dict[str, str] = {}
    for resname, seen in composition.items():
        if len(seen) != 1:
            continue
        size, element = next(iter(seen))
        if size == 1 and element in PERIODIC_ELEMENTS and len(element) <= RESNAME_WIDTH:
            result[resname] = element
    return result


def ion_element(resname: str) -> str | None:
    """The element an ion residue name is built from, if it is one.

    Amber names ion residues after the element and its oxidation state -- ZN,
    FE2, CU1 -- so the name is not a label but the only record of which metal
    this is.
    """

    stripped = resname.upper().rstrip("0123456789+-")
    return stripped if stripped and stripped in PERIODIC_ELEMENTS else None


def _free_resname(
    resname: str,
    taken: set[str],
    *,
    element: str | None = None,
    allow_head: bool = True,
) -> str:
    """Pick an unused three-character stand-in for one wide residue name.

    ``element`` names the residue when it holds a single atom of a known
    element, which is the only thing worth keeping about a name a predictor
    assigned by position. Otherwise the first three characters when they are
    free, because LIG2 reading as LIG is obvious to a student, then the
    alphabetic part shortened around a trailing number so LIG2 and LIG3 stay
    told apart, and a numbered fallback last.
    """

    # Reserved here rather than only in the caller's set, so the function is
    # safe to call on its own: it is the one deciding what a residue is named,
    # and a name that means something to mdprep is never a free choice.
    unavailable = set(taken) | RESERVED_RESNAMES

    # An ion keeps its element. Shortening ZN22 to Z01 would fit the column
    # and throw away the fact that it is zinc, which is what tleap matches
    # against its ion templates and what mdprep reads the metal from.
    element = element.upper() if element else ion_element(resname)
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

    upper = resname.upper()
    head = upper[:RESNAME_WIDTH]
    if allow_head and usable(head):
        return head
    # LIG1 and LIG2 share a head, and L01/L02 loses the one character that
    # told them apart. Keep the trailing number and shorten what precedes it.
    digits = upper[len(upper.rstrip("0123456789")) :]
    letters = upper[: len(upper) - len(digits)]
    if digits and letters:
        room = RESNAME_WIDTH - len(digits)
        if room >= 1:
            for shortened in (letters[:room], letters[0] + letters[-1] if room >= 2 else ""):
                candidate = (shortened + digits)[:RESNAME_WIDTH]
                if len(candidate) == RESNAME_WIDTH and usable(candidate):
                    return candidate
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
        # Residue numbers are deliberately not checked here. Four columns do
        # not hold a solvated box's residue count, and the writer continues
        # in hybrid-36 past 9,999 exactly as tleap does -- so a number this
        # once rejected is one the writer can represent and the reader can
        # read back. It is the writer that refuses what it cannot spell.
        if atom.resid < RESID_MIN:
            raise PdbFieldError(
                f"Residue number {atom.resid} is below the smallest a PDB file "
                f"can hold ({RESID_MIN}). Renumber the residues in the input "
                "structure."
            )
