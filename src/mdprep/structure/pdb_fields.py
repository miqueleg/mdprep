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

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

from mdprep.structure.classify import (
    COORDINATING_METAL_ELEMENTS,
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

    if chain_map or resname_map:
        atoms = [
            replace(
                atom,
                chain_id=chain_map.get(atom.chain_id, atom.chain_id),
                resname=resname_map.get(atom.resname, atom.resname),
                # The original line is a PDB line, and these no longer came
                # from one. Readers fall back to the fields when it is empty.
                original_line="",
            )
            for atom in atoms
        ]
    atoms, split_names = split_cofactor_metals(atoms, warnings)
    assigned = frozenset(resname_map.values()) | split_names
    return settle_monatomic_ion_elements(atoms), warnings, assigned


def split_cofactor_metals(
    atoms: list[AtomRecord],
    warnings: list[str],
) -> tuple[list[AtomRecord], frozenset[str]]:
    """Take a metal atom out of the cofactor it was deposited in.

    A heme is deposited as one residue, HEM, with its iron among its 43
    atoms, and a cobalamin or a chlorophyll the same way. Nothing downstream
    can parameterise that residue whole: antechamber's semi-empirical charges
    do not treat iron, and a metal is modelled as a metal site, which MCPB.py
    builds around a one-atom metal residue. So the metal is moved into a
    residue of its own as the structure is read -- named after its element,
    with the cofactor's chain and number, written after it -- and the
    cofactor without it is an ordinary ligand that coordinates it: a
    porphyrin with four nitrogens on the iron.

    The new name is assigned, not deposited. FE is the PDB's code for
    Fe(III), and a heme's iron is either; nothing may read an oxidation state
    into a name mdprep chose.

    A residue holding more than one metal atom -- an iron-sulfur cluster --
    is left whole: splitting it would not make it something mdprep can
    parameterise. So is one whose remainder would fall apart: a vanadate is
    four oxygens bonded to nothing but the vanadium, and taking it out
    leaves four loose oxygens under one residue name, not a ligand. That is
    told by every remaining atom being on its own.

    A modified residue of a protein chain is read the same way -- a cysteine
    deposited with the zinc it binds, as one residue -- with two differences.
    The metal is written after the chain rather than after the residue,
    because a one-atom residue between two residues of a chain is where
    tleap would end the chain. And the residue is left whole if what remains
    would be in more than one piece: a metal bonded to atoms beyond the amino
    acid, as the mercury of S-(methylmercury)cysteine is to its methyl, is an
    organometallic group, which is neither a metal site nor a residue mdprep
    can parameterise, and the planner says so.
    """

    keys: list[tuple[str, str, int, str | None]] = []
    members: dict[tuple[str, str, int, str | None], list[AtomRecord]] = {}
    for atom in atoms:
        key = (atom.chain_id, atom.resname, atom.resid, atom.icode)
        if key not in members:
            keys.append(key)
            members[key] = []
        members[key].append(atom)

    chained: set[tuple[str, str, int, str | None]] = set()
    split: dict[tuple[str, str, int, str | None], tuple[AtomRecord, AtomRecord]] = {}
    for key in keys:
        chain_id, resname, resid, icode = key
        residue_atoms = members[key]
        if len(residue_atoms) < 2 or resname in STANDARD_PROTEIN_RESIDUES | WATER_RESIDUES:
            continue
        metals = [atom for atom in residue_atoms if _element_of(atom) in COORDINATING_METAL_ELEMENTS]
        if len(metals) != 1:
            continue
        (metal,) = metals
        element = _element_of(metal)
        if _in_chain(key, keys, members):
            chained.add(key)
        remainder = [
            atom for atom in residue_atoms
            if atom is not metal and _element_of(atom) not in {"H", "D"}
        ]
        fragments = _fragment_count(remainder)
        # Every atom on its own: an oxoanion, whose atoms are bonded to the
        # metal and to nothing else. A cofactor with a disordered gap still
        # has fragments of several atoms, and is split as before.
        if len(remainder) > 1 and fragments == len(remainder):
            warnings.append(
                f"{_label(key)} holds a metal atom, {metal.name.strip()} "
                f"({_canonical(element)}), whose removal would leave {fragments} "
                "fragments bonded to nothing but it -- an oxoanion such as "
                "vanadate -- so it was left whole."
            )
            continue
        if key in chained and fragments > 1:
            warnings.append(
                f"{_label(key)}, a residue of its chain, holds a metal atom, "
                f"{metal.name.strip()} ({_canonical(element)}), bonded to atoms beyond "
                f"the amino acid: taking it out would leave {fragments} pieces of the "
                "residue. An organometallic group, so it was left whole."
            )
            continue
        new_key = (chain_id, element, resid, icode)
        if new_key in members:
            raise PdbFieldError(
                f"{_label(key)} holds a metal atom, {metal.name.strip()} "
                f"({_canonical(element)}), that mdprep moves into a residue of its "
                f"own, {_label(new_key)}, but the "
                "structure already has a residue by that name and number. "
                "Renumber one of them in the input."
            )
        split[key] = (
            metal,
            replace(
                metal,
                resname=element,
                record_name="HETATM",
                element=_canonical(element),
                original_line="",
            ),
        )
        warnings.append(
            f"{_label(key)} holds a metal atom, {metal.name.strip()} ({_canonical(element)}). "
            f"mdprep moved it into a residue of its own, {_label(new_key)}, so it "
            "can be modelled as a metal site; the rest of "
            + (
                f"{resname} is a residue of the chain that coordinates it, and the "
                "metal is written after the chain."
                if key in chained
                else f"{resname} is a ligand that coordinates it."
            )
            + f" Selectors in the manifest use {_label(new_key)}."
        )
    if not split:
        return atoms, frozenset()

    # Every other atom keeps its place; the metal goes right after the last
    # atom of its cofactor, or of its chain.
    last_in_chain = {atom.chain_id: id(atom) for atom in atoms}
    moved_out = {id(metal) for metal, _ in split.values()}
    after: dict[int, list[AtomRecord]] = {}
    for key, (_, moved) in split.items():
        anchor = last_in_chain[key[0]] if key in chained else id(members[key][-1])
        after.setdefault(anchor, []).append(moved)
    result: list[AtomRecord] = []
    for atom in atoms:
        if id(atom) not in moved_out:
            result.append(atom)
        result.extend(after.get(id(atom), ()))
    return result, frozenset(moved.resname for _, moved in split.values())


# A peptide C-N bond is 1.33 A; this is the limit the rest of mdprep uses.
_PEPTIDE_BOND_MAX_ANGSTROM = 2.0


def _in_chain(
    key: tuple[str, str, int, str | None],
    keys: list[tuple[str, str, int, str | None]],
    members: dict[tuple[str, str, int, str | None], list[AtomRecord]],
) -> bool:
    """Whether this residue's backbone is peptide-bonded to another of its chain."""

    own = {atom.name.strip(): atom for atom in members[key]}
    if not {"N", "CA", "C"} <= set(own):
        return False
    for other in keys:
        if other == key or other[0] != key[0]:
            continue
        for atom in members[other]:
            name = atom.name.strip()
            if (name == "N" and _within(own["C"], atom, _PEPTIDE_BOND_MAX_ANGSTROM)) or (
                name == "C" and _within(own["N"], atom, _PEPTIDE_BOND_MAX_ANGSTROM)
            ):
                return True
    return False


def _within(first: AtomRecord, second: AtomRecord, limit: float) -> bool:
    return (
        (first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2
        <= limit * limit
    )


@contextmanager
def expected_split_metal_numbering() -> Iterator[None]:
    """Read a PDB mdprep wrote with OpenMM, without its residue-number warning.

    A metal read out of its cofactor keeps the cofactor's number, so the two
    are consecutive residues sharing one -- which OpenMM's PDB reader reports
    every time it reads such a file, and still reads as the two residues they
    are. Expected, so not passed on as if something were wrong.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="WARNING: two consecutive residues with same number"
        )
        yield


def _element_of(atom: AtomRecord) -> str:
    return (atom.element or "").strip().upper()


# Covalent radii, RDKit's, for reading bonds from distances: whether the atoms
# left in a cofactor without its metal are still one molecule, and where a
# kept residue's geometry is too strained to read its bonds from
# (residues/capping.py). Anything not listed is taken as carbon.
COVALENT_RADII = {
    "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "SI": 1.11, "P": 1.07, "S": 1.05, "CL": 1.02, "AS": 1.19, "SE": 1.20,
    "BR": 1.20, "I": 1.39,
}


def _fragment_count(atoms: list[AtomRecord]) -> int:
    """How many separate molecules these atoms form, bonded by distance."""

    heavy = [atom for atom in atoms if _element_of(atom) not in {"H", "D"}]
    if len(heavy) < 2:
        return len(heavy)
    parent = list(range(len(heavy)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, first in enumerate(heavy):
        for j in range(i + 1, len(heavy)):
            second = heavy[j]
            limit = (
                COVALENT_RADII.get(_element_of(first), 0.76)
                + COVALENT_RADII.get(_element_of(second), 0.76)
                + 0.45
            )
            distance_squared = (
                (first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2
            )
            if distance_squared <= limit * limit:
                parent[root(i)] = root(j)
    return len({root(index) for index in range(len(heavy))})


def _label(key: tuple[str, str, int, str | None]) -> str:
    chain_id, resname, resid, icode = key
    return f"{chain_id}:{resname}{resid}{icode or ''}"


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
