"""Copies of a protein in one structure, and which residue is which in each.

A predicted dimer and a crystal asymmetric unit with four chains pose the same
problem. Every question mdprep asks about a ligand, a metal or a coordinating
residue is asked once per copy, and copies rarely number their residues
alike: a crystal copy starts later, misses a loop or files its cofactor under
the next free number, and a prediction puts every ligand in a chain of its
own. Matching residues by number is therefore wrong about as often as it is
right -- 2FCT numbers chain A's alpha-ketoglutarate 897 and chain B's 898.

So residues are matched the way a person would match them:

* protein residues by aligning the chain sequences, which survives offsets,
  gaps and unresolved termini;
* a heterogen with the one of the same name, with the same atom names, that
  sits in the same place relative to the protein around it once that
  neighbourhood is superposed -- so a lid that moved between copies does not
  break the match;
* a metal only when its whole first shell matches as well, donor for donor
  and atom for atom, and a ligand on a metal only when it binds that metal's
  counterpart the same way. A donor bound in one copy only has to be near
  enough in another (SHELL_MATCH_ANGSTROM), since a prediction rarely puts
  the same oxygen at the same distance from both irons.

Anything that does not match is reported with the reason, never guessed at.
Its questions are then asked on their own.

Pure Python on purpose: the input builder runs this in the browser, where
there is no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import sqrt
from typing import Any, Sequence

from mdprep.structure.classify import (
    is_bulk_ion_residue,
    is_likely_ligand_or_cofactor,
    is_metal_ion_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.coordination import (
    COORDINATION_CUTOFF_ANGSTROM,
    metal_bound_waters,
    metal_donor_atoms,
)
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import ResidueId, ResidueRecord


# One-letter codes, with the protonation and Amber variants of a residue
# named as the residue itself: HID in one copy and HIS in another are the
# same position in the same protein.
ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D",
    "CYS": "C", "CYM": "C", "CYX": "C", "GLN": "Q", "GLU": "E", "GLH": "E",
    "GLY": "G", "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "LYN": "K", "MET": "M",
    "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Two chains are copies of one protein when this share of the shorter one
# aligns residue for residue. A crystal copy that misses a loop still scores
# 1.0 -- unresolved residues are not in either sequence -- and a point mutant
# 0.99; two different subunits of a complex score far below it.
COPY_IDENTITY = 0.9

# A heterogen's neighbourhood: the paired protein residues with an atom this
# close to it, superposed to compare its position in two copies.
NEIGHBOURHOOD_ANGSTROM = 10.0

# How far a heterogen may sit from its counterpart once the neighbourhood is
# superposed. A single atom -- a metal, a halide, a bound water -- gets the
# tighter limit, because two waters on one metal are under 3 A apart and a
# looser one could swap them.
ATOM_MATCH_ANGSTROM = 1.5
LIGAND_MATCH_ANGSTROM = 2.5

# A donor within the first-shell cutoff of one copy's metal is in the shell
# of another copy's when its counterpart is this close to that metal. Copies
# of one site are not drawn alike: a predicted dimer puts the same oxygen
# 2.5 A from one iron and 2.7 A from the other, either side of the 2.6 A
# cutoff, and that is one site, not two. Further out it is not bound there,
# and the copies are different sites -- a water that left, a carboxylate
# that turned away.
SHELL_MATCH_ANGSTROM = 3.0

# A heterogen chain touching a protein chain this closely belongs to it.
CONTACT_ANGSTROM = 4.5
# Failing any contact, the nearest protein chain within this distance.
NEAR_ANGSTROM = 6.0


@dataclass(frozen=True)
class ChainAlignment:
    """One chain aligned onto the first chain of its group."""

    reference: str
    chain: str
    # (reference residue, residue in this chain), identical residue types only.
    pairs: tuple[tuple[ResidueId, ResidueId], ...]
    identity: float

    @property
    def renumbered(self) -> bool:
        """Whether a residue in this copy carries a different number."""

        return any(
            (reference.resid, reference.icode) != (copy.resid, copy.icode)
            for reference, copy in self.pairs
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "chain": self.chain,
            "identity": round(self.identity, 4),
            "paired_residues": len(self.pairs),
            "numbering_differs": self.renumbered,
        }


@dataclass(frozen=True)
class CopiedDonor:
    """An atom another copy of a metal's site binds, a little further out here."""

    residue: ResidueId
    atom_name: str
    # The metal of the copy that binds it within the first-shell cutoff, and
    # how close it is to that metal.
    bound_to: ResidueId
    distance_there: float


@dataclass(frozen=True)
class OligomerAnalysis:
    """Which chains are copies of one protein, and how their residues pair."""

    protein_chains: tuple[str, ...]
    # Chains that are copies of one protein, in input order. The first chain
    # of each group is its reference: the copy whose questions are asked.
    groups: tuple[tuple[str, ...], ...]
    alignments: dict[str, ChainAlignment]
    # The protein chain -- the copy -- every residue belongs to. A heterogen
    # in a chain of its own (as a prediction writes every ligand) belongs to
    # the protein chain it sits against; one touching no protein to none. A
    # water outside the protein chains belongs to the copy it sits against,
    # whatever chain it was written in.
    chain_of_residue: dict[ResidueId, str | None]
    chain_of_nonprotein_chain: dict[str, str | None]
    # Residue in a copy -> the residue it corresponds to in the reference.
    counterparts: dict[ResidueId, ResidueId] = field(default_factory=dict)
    # Heterogens of a copy that have no counterpart, and why.
    unmatched: dict[ResidueId, str] = field(default_factory=dict)
    # Metal -> the atoms its site is confirmed with beyond the first-shell
    # cutoff, because a copy of the site binds their counterparts within it.
    # Every copy of a site then offers the same bonds, each at its own
    # distance.
    copied_donors: dict[ResidueId, tuple[CopiedDonor, ...]] = field(default_factory=dict)

    @property
    def homomeric(self) -> bool:
        """Whether any protein appears in more than one copy."""

        return any(len(group) > 1 for group in self.groups)

    def reference_chain(self, chain: str | None) -> str | None:
        """The chain whose questions stand for ``chain``'s."""

        if chain is None:
            return None
        for group in self.groups:
            if chain in group:
                return group[0]
        return chain

    def counterpart(self, residue_id: ResidueId) -> ResidueId | None:
        return self.counterparts.get(residue_id)

    def copies_of(self, reference: ResidueId) -> list[ResidueId]:
        """Every residue that corresponds to ``reference``, in input order."""

        return [copy for copy, source in self.counterparts.items() if source == reference]

    def counterpart_beside(self, reference: ResidueId, residue: ResidueId) -> ResidueId | None:
        """The residue standing for ``reference`` in the copy ``residue`` is in."""

        chain = self.chain_of_residue.get(residue)
        if chain is not None and self.chain_of_residue.get(reference) == chain:
            return reference
        for copy in self.copies_of(reference):
            if self.chain_of_residue.get(copy) == chain:
                return copy
        return None

    def heterogen_counterparts(self) -> dict[ResidueId, ResidueId]:
        protein = {
            copy for alignment in self.alignments.values() for _, copy in alignment.pairs
        }
        return {
            copy: reference
            for copy, reference in self.counterparts.items()
            if copy not in protein
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "protein_chains": list(self.protein_chains),
            "groups": [list(group) for group in self.groups],
            "alignments": {
                chain: alignment.to_dict() for chain, alignment in self.alignments.items()
            },
            "chain_of_nonprotein_chain": dict(self.chain_of_nonprotein_chain),
            "counterparts": {
                copy.display(): reference.display()
                for copy, reference in self.heterogen_counterparts().items()
            },
            "unmatched": {
                residue.display(): reason for residue, reason in self.unmatched.items()
            },
        }


def protein_sequence(residues: Sequence[ResidueRecord]) -> str:
    return "".join(ONE_LETTER.get(residue.id.resname, "X") for residue in residues)


def align_chains(
    reference: Sequence[ResidueRecord], other: Sequence[ResidueRecord]
) -> tuple[tuple[tuple[ResidueId, ResidueId], ...], float]:
    """Identical residues of two chains, paired by sequence, and the identity.

    The matching blocks of the two sequences give the pairs, so a copy that
    misses a loop pairs on both sides of the gap and a mutated position is
    simply left out. Identity is the paired share of the shorter chain.
    """

    if not reference or not other:
        return (), 0.0
    matcher = SequenceMatcher(
        None, protein_sequence(reference), protein_sequence(other), autojunk=False
    )
    pairs: list[tuple[ResidueId, ResidueId]] = []
    for start_reference, start_other, length in matcher.get_matching_blocks():
        for offset in range(length):
            pairs.append(
                (
                    reference[start_reference + offset].id,
                    other[start_other + offset].id,
                )
            )
    return tuple(pairs), len(pairs) / min(len(reference), len(other))


def analyse_oligomer(residues: Sequence[ResidueRecord]) -> OligomerAnalysis:
    """Group the protein chains into copies and pair their residues."""

    by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in residues:
        if is_standard_protein_residue(residue) and residue.id.chain_id:
            by_chain.setdefault(residue.id.chain_id, []).append(residue)
    chains = tuple(by_chain)

    groups: list[list[str]] = []
    alignments: dict[str, ChainAlignment] = {}
    for chain in chains:
        placed = False
        for group in groups:
            pairs, identity = align_chains(by_chain[group[0]], by_chain[chain])
            shorter = min(len(by_chain[group[0]]), len(by_chain[chain]))
            if identity >= COPY_IDENTITY and len(pairs) >= min(10, shorter):
                group.append(chain)
                alignments[chain] = ChainAlignment(group[0], chain, pairs, identity)
                placed = True
                break
        if not placed:
            groups.append([chain])

    chain_of_nonprotein = _assign_nonprotein_chains(residues, set(chains))
    owner_of_water = _assign_waters(residues, set(chains))
    chain_of_residue: dict[ResidueId, str | None] = {}
    for residue in residues:
        chain = residue.id.chain_id
        chain_of_residue[residue.id] = (
            chain
            if chain in by_chain
            else owner_of_water.get(residue.id, chain_of_nonprotein.get(chain))
        )

    analysis = OligomerAnalysis(
        protein_chains=chains,
        groups=tuple(tuple(group) for group in groups),
        alignments=alignments,
        chain_of_residue=chain_of_residue,
        chain_of_nonprotein_chain=chain_of_nonprotein,
    )
    if analysis.homomeric:
        _pair_copies(analysis, residues)
    return analysis


# --------------------------------------------------------------------------
# Which protein chain a heterogen chain belongs to
# --------------------------------------------------------------------------


def _assign_nonprotein_chains(
    residues: Sequence[ResidueRecord], protein_chains: set[str]
) -> dict[str, str | None]:
    """The protein chain each chain without protein sits against.

    A prediction writes every ligand as a chain of its own, so keeping one
    copy of a dimer has to know that chain C is chain A's cofactor and chain
    D chain B's. Counted in contacts, so a ligand at an interface goes to the
    chain it touches most. Waters are left out: a chain of crystal waters
    spans every copy and is never anyone's.
    """

    grid = _protein_grid(residues, protein_chains)

    members: dict[str, list[ResidueRecord]] = {}
    for residue in residues:
        chain = residue.id.chain_id
        if chain in protein_chains or is_water_residue(residue):
            continue
        members.setdefault(chain, []).append(residue)
    for residue in residues:
        if residue.id.chain_id not in protein_chains and is_water_residue(residue):
            members.setdefault(residue.id.chain_id, [])

    assigned: dict[str, str | None] = {}
    for chain, chain_residues in members.items():
        contacts: dict[str, int] = {}
        nearest: dict[str, float] = {}
        for residue in chain_residues:
            for atom in residue.atoms:
                if is_hydrogen(atom):
                    continue
                for owner, distance in grid.near(atom, NEAR_ANGSTROM):
                    nearest[owner] = min(nearest.get(owner, distance), distance)
                    if distance <= CONTACT_ANGSTROM:
                        contacts[owner] = contacts.get(owner, 0) + 1
        if contacts:
            assigned[chain] = max(
                contacts, key=lambda owner: (contacts[owner], -nearest[owner])
            )
        elif nearest:
            assigned[chain] = min(nearest, key=lambda owner: nearest[owner])
        else:
            assigned[chain] = None

    # A halide on an iron that is itself a chain of its own touches no
    # protein; it belongs wherever the iron does.
    unassigned = [chain for chain, owner in assigned.items() if owner is None and members[chain]]
    if unassigned:
        placed = _Grid(CONTACT_ANGSTROM)
        for chain, owner in assigned.items():
            if owner is None:
                continue
            for residue in members[chain]:
                for atom in residue.atoms:
                    placed.add(atom, owner)
        for chain in unassigned:
            best: tuple[float, str] | None = None
            for residue in members[chain]:
                for atom in residue.atoms:
                    for owner, distance in placed.near(atom, CONTACT_ANGSTROM):
                        if best is None or distance < best[0]:
                            best = (distance, owner)
            if best is not None:
                assigned[chain] = best[1]
    return assigned


def _assign_waters(
    residues: Sequence[ResidueRecord], protein_chains: set[str]
) -> dict[ResidueId, str]:
    """The protein chain each water outside the protein chains sits against.

    One water at a time, not with its chain. A dimer put together by hand
    can leave the second copy's waters in a chain with the first copy's
    ligand, and the water on the second copy's iron then counted as the
    first copy's: the two iron sites differed by a water each and could not
    share a fit. A water touching no protein is left to its chain.
    """

    grid = _protein_grid(residues, protein_chains)
    owners: dict[ResidueId, str] = {}
    for residue in residues:
        if residue.id.chain_id in protein_chains or not is_water_residue(residue):
            continue
        contacts: dict[str, int] = {}
        nearest: dict[str, float] = {}
        for atom in residue.atoms:
            if is_hydrogen(atom):
                continue
            for owner, distance in grid.near(atom, NEAR_ANGSTROM):
                nearest[owner] = min(nearest.get(owner, distance), distance)
                if distance <= CONTACT_ANGSTROM:
                    contacts[owner] = contacts.get(owner, 0) + 1
        if contacts:
            owners[residue.id] = max(
                contacts, key=lambda owner: (contacts[owner], -nearest[owner])
            )
        elif nearest:
            owners[residue.id] = min(nearest, key=lambda owner: nearest[owner])
    return owners


def _protein_grid(residues: Sequence[ResidueRecord], protein_chains: set[str]) -> "_Grid":
    grid = _Grid(CONTACT_ANGSTROM)
    for residue in residues:
        if residue.id.chain_id in protein_chains and is_standard_protein_residue(residue):
            for atom in residue.atoms:
                if not is_hydrogen(atom):
                    grid.add(atom, residue.id.chain_id)
    return grid


# --------------------------------------------------------------------------
# Heterogen counterparts
# --------------------------------------------------------------------------


def _pair_copies(analysis: OligomerAnalysis, residues: Sequence[ResidueRecord]) -> None:
    residue_by_id = {residue.id: residue for residue in residues}
    metals = [residue for residue in residues if is_metal_ion_residue(residue)]
    # As far out as a copy's shell reaches, so that a water bound to one
    # copy's metal is paired with its counterpart a little further out.
    bound_waters = {
        residue.id
        for residue in metal_bound_waters(
            residues, metals, cutoff_angstrom=SHELL_MATCH_ANGSTROM
        )
    }

    def is_item(residue: ResidueRecord) -> bool:
        if is_standard_protein_residue(residue):
            return False
        if is_water_residue(residue):
            return residue.id in bound_waters
        return (
            is_metal_ion_residue(residue)
            or is_bulk_ion_residue(residue)
            or is_likely_ligand_or_cofactor(residue)
        )

    items_by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in residues:
        chain = analysis.chain_of_residue.get(residue.id)
        if chain is not None and is_item(residue):
            items_by_chain.setdefault(chain, []).append(residue)

    for alignment in analysis.alignments.values():
        for reference, copy in alignment.pairs:
            analysis.counterparts[copy] = reference
        reference_items = items_by_chain.get(alignment.reference, [])
        copy_items = items_by_chain.get(alignment.chain, [])
        candidates: list[tuple[float, ResidueId, ResidueId]] = []
        for reference_item in reference_items:
            frame = _neighbourhood_frame(reference_item, alignment, residue_by_id)
            if frame is None:
                continue
            limit = (
                ATOM_MATCH_ANGSTROM
                if len(reference_item.atoms) == 1 or is_water_residue(reference_item)
                else LIGAND_MATCH_ANGSTROM
            )
            for copy_item in copy_items:
                if copy_item.id.resname != reference_item.id.resname:
                    continue
                if copy_item.atom_names() != reference_item.atom_names():
                    continue
                distance = _deviation(copy_item, reference_item, frame)
                if distance <= limit:
                    candidates.append((distance, copy_item.id, reference_item.id))
        taken_copy: set[ResidueId] = set()
        taken_reference: set[ResidueId] = set()
        for _, copy_id, reference_id in sorted(candidates, key=lambda item: item[0]):
            if copy_id in taken_copy or reference_id in taken_reference:
                continue
            taken_copy.add(copy_id)
            taken_reference.add(reference_id)
            analysis.counterparts[copy_id] = reference_id

    _require_matching_metal_shells(analysis, residues, residue_by_id)

    for chain, chain_items in items_by_chain.items():
        if analysis.reference_chain(chain) == chain:
            continue
        reference = analysis.reference_chain(chain)
        for residue in chain_items:
            if is_water_residue(residue) or is_bulk_ion_residue(residue):
                continue
            if residue.id in analysis.counterparts or residue.id in analysis.unmatched:
                continue
            twin = _same_place_other_name(
                residue,
                items_by_chain.get(reference, []),
                analysis.alignments[chain],
                residue_by_id,
            )
            analysis.unmatched[residue.id] = (
                f"it sits where {twin.id.display()} does, with the same atoms, but is "
                f"called {residue.id.resname} rather than {twin.id.resname} -- copies "
                "share parameters under one residue name, so give both the same one "
                "in the input"
                if twin is not None
                else f"no {residue.id.resname} with the same atoms sits in the same "
                f"place in chain {reference}"
            )


def _same_place_other_name(
    residue: ResidueRecord,
    reference_items: Sequence[ResidueRecord],
    alignment: ChainAlignment,
    residue_by_id: dict[ResidueId, ResidueRecord],
) -> ResidueRecord | None:
    """The reference copy's heterogen in ``residue``'s place, under another name.

    A dimer assembled by hand often names the second copy's ligands apart
    (THC and THC2, FE1 and FE2). Pairing is by name, since tleap builds every
    residue of one name from one template, so these are not copies -- and
    "nothing with the same atoms in the same place" said nothing of why.
    """

    found: list[tuple[float, ResidueRecord]] = []
    for item in reference_items:
        if item.id.resname == residue.id.resname or item.atom_names() != residue.atom_names():
            continue
        frame = _neighbourhood_frame(item, alignment, residue_by_id)
        if frame is None:
            continue
        limit = ATOM_MATCH_ANGSTROM if len(item.atoms) == 1 else LIGAND_MATCH_ANGSTROM
        distance = _deviation(residue, item, frame)
        if distance <= limit:
            found.append((distance, item))
    return min(found, key=lambda pair: pair[0])[1] if found else None


def _require_matching_metal_shells(
    analysis: OligomerAnalysis,
    residues: Sequence[ResidueRecord],
    residue_by_id: dict[ResidueId, ResidueRecord],
) -> None:
    """Keep a metal pairing only if its whole first shell pairs too.

    A bonded site copied into another chain takes its templates and bonds
    with it, so the copy's iron has to be bound by the counterparts of the
    same atoms -- the same histidine nitrogens, the same alpha-ketoglutarate
    oxygens, the water in the same place. One donor more or less and it is a
    different site, whose questions are asked on their own. A ligand bound to
    a metal follows its metal, since its charges may come from that site's
    fit.

    An atom within the first-shell cutoff of one copy's metal has to be
    within SHELL_MATCH_ANGSTROM of every other copy's. The site is then the
    atoms any of its copies binds, and each copy is confirmed with all of
    them: the ones a little further out in one copy are its
    ``copied_donors``.
    """

    metals = [residue for residue in residues if is_metal_ion_residue(residue)]
    # Every donor atom within reach of each metal, and how far it is.
    reach = {
        metal.id: {
            (donor.id, atom.name): distance
            for donor, atom, distance in metal_donor_atoms(
                residues, metal, cutoff_angstrom=SHELL_MATCH_ANGSTROM
            )
        }
        for metal in metals
    }

    def bound(metal: ResidueId) -> dict[tuple[ResidueId, str], float]:
        return {
            atom: distance
            for atom, distance in reach[metal].items()
            if distance <= COORDINATION_CUTOFF_ANGSTROM
        }

    def as_reference(
        atom: tuple[ResidueId, str], metal: ResidueId, reference: ResidueId
    ) -> tuple[ResidueId | None, str]:
        """One of ``metal``'s donor atoms, named as the reference copy's."""

        if metal == reference:
            return atom
        return (analysis.counterparts.get(atom[0]), atom[1])

    def distance_to(residue: ResidueId | None, name: str, metal: ResidueId) -> str:
        atom = _named_atom(residue_by_id, residue, name)
        if atom is None:
            return "is missing"
        return f"is {_atom_distance(residue_by_id[metal].atoms[0], atom):.2f} A away"

    copies: dict[ResidueId, list[ResidueId]] = {}
    for copy_id, reference_id in analysis.counterparts.items():
        copy = residue_by_id.get(copy_id)
        if copy is not None and reference_id in reach and is_metal_ion_residue(copy):
            copies.setdefault(reference_id, []).append(copy_id)

    for reference_id, members in copies.items():
        while True:
            # The site, in the reference's atoms: every atom a copy binds,
            # with the metal that binds it closest.
            site: dict[
                tuple[ResidueId | None, str], tuple[ResidueId, tuple[ResidueId, str], float]
            ] = {}
            for metal in (reference_id, *members):
                for atom, distance in bound(metal).items():
                    key = as_reference(atom, metal, reference_id)
                    if key not in site or distance < site[key][2]:
                        site[key] = (metal, atom, distance)
            differences: dict[ResidueId, list[str]] = {}
            for member in members:
                own = {as_reference(atom, member, reference_id) for atom in reach[member]}
                found: list[str] = []
                # Whatever it binds, the reference reaches...
                for atom, distance in bound(member).items():
                    key = as_reference(atom, member, reference_id)
                    if key[0] is not None and key in reach[reference_id]:
                        continue
                    found.append(
                        f"it binds {atom[0].display()} {atom[1]} at {distance:.2f} A, "
                        + (
                            f"whose counterpart {distance_to(key[0], key[1], reference_id)} "
                            f"from {reference_id.display()}"
                            if key[0] is not None
                            else "which has no counterpart in chain "
                            f"{analysis.chain_of_residue.get(reference_id)}"
                        )
                    )
                # ...and whatever the others bind, it reaches.
                for key, (binder, atom, distance) in site.items():
                    if binder == member or key[0] is None or key not in reach[reference_id]:
                        continue
                    if key in own:
                        continue
                    here = analysis.counterpart_beside(key[0], member)
                    found.append(
                        f"{binder.display()} binds {atom[0].display()} {atom[1]} at "
                        f"{distance:.2f} A, "
                        + (
                            f"whose counterpart here {distance_to(here, key[1], member)}"
                            if here is not None
                            else "which has no counterpart here"
                        )
                    )
                if found:
                    differences[member] = found
            if not differences:
                break
            for member, found in differences.items():
                del analysis.counterparts[member]
                analysis.unmatched[member] = (
                    f"its first shell differs from {reference_id.display()}'s "
                    f"({'; '.join(found)})"
                )
            members = [member for member in members if member not in differences]
        if not members:
            continue
        for metal in (reference_id, *members):
            own = {
                as_reference(atom, metal, reference_id): (atom, distance)
                for atom, distance in reach[metal].items()
            }
            further_out = sorted(
                (
                    (own[key][1], CopiedDonor(*own[key][0], bound_to=binder, distance_there=there))
                    for key, (binder, _, there) in site.items()
                    if own[key][1] > COORDINATION_CUTOFF_ANGSTROM
                ),
                key=lambda item: item[0],
            )
            if further_out:
                analysis.copied_donors[metal] = tuple(donor for _, donor in further_out)

    # Which metal each ligand binds, if any: through its own shell, or
    # through what its copies bind.
    bound_to: dict[ResidueId, set[ResidueId]] = {}
    for metal in metals:
        for residue, _ in bound(metal.id):
            bound_to.setdefault(residue, set()).add(metal.id)
        for donor in analysis.copied_donors.get(metal.id, ()):
            bound_to.setdefault(donor.residue, set()).add(metal.id)
    for copy_id, reference_id in list(analysis.counterparts.items()):
        copy = residue_by_id.get(copy_id)
        if copy is None or is_standard_protein_residue(copy) or is_metal_ion_residue(copy):
            continue
        copy_metals = bound_to.get(copy_id, set())
        reference_metals = bound_to.get(reference_id, set())
        if not copy_metals and not reference_metals:
            continue
        mapped = {analysis.counterparts.get(metal) for metal in copy_metals}
        if mapped != reference_metals:
            del analysis.counterparts[copy_id]
            if not is_water_residue(copy) and not is_bulk_ion_residue(copy):
                analysis.unmatched[copy_id] = (
                    f"it binds a metal differently from {reference_id.display()}"
                )


def _named_atom(
    residue_by_id: dict[ResidueId, ResidueRecord], residue: ResidueId | None, name: str
):
    record = residue_by_id.get(residue) if residue is not None else None
    if record is None:
        return None
    return next((atom for atom in record.atoms if atom.name == name), None)


def _atom_distance(first, second) -> float:
    return sqrt(
        (first.x - second.x) ** 2 + (first.y - second.y) ** 2 + (first.z - second.z) ** 2
    )


def site_donor_atoms(
    residues: Sequence[ResidueRecord],
    metal: ResidueRecord,
    analysis: OligomerAnalysis | None,
) -> list[tuple[ResidueRecord, Any, float]]:
    """A metal's first shell as its site is confirmed, nearest first.

    The donors within the first-shell cutoff, and the atoms that a copy of
    the site binds and that sit a little further out here
    (``OligomerAnalysis.copied_donors``): every copy of a site is then asked
    about, and bonded to, the same atoms, each at its own distance.
    """

    found = metal_donor_atoms(residues, metal)
    copied = analysis.copied_donors.get(metal.id, ()) if analysis is not None else ()
    if not copied:
        return found
    residue_by_id = {residue.id: residue for residue in residues}
    for donor in copied:
        atom = _named_atom(residue_by_id, donor.residue, donor.atom_name)
        if atom is not None:
            found.append(
                (residue_by_id[donor.residue], atom, _atom_distance(metal.atoms[0], atom))
            )
    found.sort(key=lambda item: item[2])
    return found


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


Vector = tuple[float, float, float]
Matrix = tuple[Vector, Vector, Vector]


@dataclass(frozen=True)
class _Frame:
    """A rigid transform taking a copy's coordinates onto the reference."""

    rotation: Matrix
    mobile_centre: Vector
    target_centre: Vector

    def apply(self, point: Vector) -> Vector:
        x = point[0] - self.mobile_centre[0]
        y = point[1] - self.mobile_centre[1]
        z = point[2] - self.mobile_centre[2]
        r = self.rotation
        return (
            r[0][0] * x + r[0][1] * y + r[0][2] * z + self.target_centre[0],
            r[1][0] * x + r[1][1] * y + r[1][2] * z + self.target_centre[1],
            r[2][0] * x + r[2][1] * y + r[2][2] * z + self.target_centre[2],
        )


def _neighbourhood_frame(
    item: ResidueRecord,
    alignment: ChainAlignment,
    residue_by_id: dict[ResidueId, ResidueRecord],
) -> _Frame | None:
    """Superpose the copy's protein around ``item`` onto the reference's."""

    centre = _centroid([(atom.x, atom.y, atom.z) for atom in item.atoms])
    near: list[tuple[Vector, Vector]] = []
    every: list[tuple[Vector, Vector]] = []
    limit = NEIGHBOURHOOD_ANGSTROM * NEIGHBOURHOOD_ANGSTROM
    for reference_id, copy_id in alignment.pairs:
        reference = residue_by_id.get(reference_id)
        copy = residue_by_id.get(copy_id)
        if reference is None or copy is None:
            continue
        reference_ca = _atom_xyz(reference, "CA")
        copy_ca = _atom_xyz(copy, "CA")
        if reference_ca is None or copy_ca is None:
            continue
        every.append((copy_ca, reference_ca))
        if any(
            (atom.x - centre[0]) ** 2 + (atom.y - centre[1]) ** 2 + (atom.z - centre[2]) ** 2
            <= limit
            for atom in reference.atoms
        ):
            near.append((copy_ca, reference_ca))
    pairs = near if len(near) >= 4 else every
    if len(pairs) < 3:
        return None
    return kabsch([mobile for mobile, _ in pairs], [target for _, target in pairs])


def _deviation(copy: ResidueRecord, reference: ResidueRecord, frame: _Frame) -> float:
    """Largest atom-to-atom distance once ``copy`` is moved onto ``reference``.

    Atoms are compared by name, which both residues share by construction.
    """

    by_name = {atom.name: atom for atom in reference.atoms}
    worst = 0.0
    for atom in copy.atoms:
        target = by_name[atom.name]
        moved = frame.apply((atom.x, atom.y, atom.z))
        worst = max(
            worst,
            sqrt(
                (moved[0] - target.x) ** 2
                + (moved[1] - target.y) ** 2
                + (moved[2] - target.z) ** 2
            ),
        )
    return worst


def kabsch(mobile: Sequence[Vector], target: Sequence[Vector]) -> _Frame:
    """The rotation and translation that best fit ``mobile`` onto ``target``.

    Horn's quaternion form of the least-squares fit, with the 4x4 symmetric
    eigenproblem solved by Jacobi rotations -- a few dozen lines that need no
    linear algebra library and cannot return a reflection.
    """

    if len(mobile) != len(target) or not mobile:
        raise ValueError("kabsch needs two equal, non-empty point sets")
    mobile_centre = _centroid(mobile)
    target_centre = _centroid(target)
    s = [[0.0] * 3 for _ in range(3)]
    for m, t in zip(mobile, target):
        a = (m[0] - mobile_centre[0], m[1] - mobile_centre[1], m[2] - mobile_centre[2])
        b = (t[0] - target_centre[0], t[1] - target_centre[1], t[2] - target_centre[2])
        for i in range(3):
            for j in range(3):
                s[i][j] += a[i] * b[j]
    sxx, sxy, sxz = s[0]
    syx, syy, syz = s[1]
    szx, szy, szz = s[2]
    n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
    values, vectors = _jacobi_eigen(n)
    best = max(range(4), key=lambda index: values[index])
    q0, q1, q2, q3 = (vectors[row][best] for row in range(4))
    rotation = (
        (
            q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
        ),
        (
            2 * (q1 * q2 + q0 * q3),
            q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
            2 * (q2 * q3 - q0 * q1),
        ),
        (
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
        ),
    )
    return _Frame(rotation, mobile_centre, target_centre)


def _jacobi_eigen(
    matrix: list[list[float]], *, sweeps: int = 50
) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and column eigenvectors of a small symmetric matrix."""

    size = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(sweeps):
        off = sum(a[i][j] ** 2 for i in range(size) for j in range(size) if i != j)
        if off < 1e-22:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
                c = 1.0 / sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(size):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(size)], v


def _centroid(points: Sequence[Vector]) -> Vector:
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def _atom_xyz(residue: ResidueRecord, name: str) -> Vector | None:
    for atom in residue.atoms:
        if atom.name == name:
            return (atom.x, atom.y, atom.z)
    return None


class _Grid:
    """Atoms hashed into cubes, for neighbour searches without numpy."""

    def __init__(self, cell: float) -> None:
        self.cell = cell
        self.cells: dict[tuple[int, int, int], list[tuple[float, float, float, str]]] = {}

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (int(x // self.cell), int(y // self.cell), int(z // self.cell))

    def add(self, atom, owner: str) -> None:
        self.cells.setdefault(self._key(atom.x, atom.y, atom.z), []).append(
            (atom.x, atom.y, atom.z, owner)
        )

    def near(self, atom, radius: float) -> list[tuple[str, float]]:
        """(owner, distance) for every stored atom within ``radius``."""

        reach = int(radius // self.cell) + 1
        cx, cy, cz = self._key(atom.x, atom.y, atom.z)
        found: list[tuple[str, float]] = []
        limit = radius * radius
        for dx in range(-reach, reach + 1):
            for dy in range(-reach, reach + 1):
                for dz in range(-reach, reach + 1):
                    for x, y, z, owner in self.cells.get((cx + dx, cy + dy, cz + dz), ()):
                        squared = (x - atom.x) ** 2 + (y - atom.y) ** 2 + (z - atom.z) ** 2
                        if squared <= limit:
                            found.append((owner, sqrt(squared)))
        return found
