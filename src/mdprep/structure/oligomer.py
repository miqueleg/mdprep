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
  counterpart the same way.

Anything that does not match is reported with the reason, never guessed at.
Its questions are then asked on their own.

Pure Python on purpose: the input builder runs this in the browser, where
there is no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from math import sqrt
from typing import Sequence

from mdprep.structure.classify import (
    is_bulk_ion_residue,
    is_likely_ligand_or_cofactor,
    is_metal_ion_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.coordination import metal_bound_waters, metal_donor_atoms
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
class OligomerAnalysis:
    """Which chains are copies of one protein, and how their residues pair."""

    protein_chains: tuple[str, ...]
    # Chains that are copies of one protein, in input order. The first chain
    # of each group is its reference: the copy whose questions are asked.
    groups: tuple[tuple[str, ...], ...]
    alignments: dict[str, ChainAlignment]
    # The protein chain -- the copy -- every residue belongs to. A heterogen
    # in a chain of its own (as a prediction writes every ligand) belongs to
    # the protein chain it sits against; one touching no protein to none.
    chain_of_residue: dict[ResidueId, str | None]
    chain_of_nonprotein_chain: dict[str, str | None]
    # Residue in a copy -> the residue it corresponds to in the reference.
    counterparts: dict[ResidueId, ResidueId] = field(default_factory=dict)
    # Heterogens of a copy that have no counterpart, and why.
    unmatched: dict[ResidueId, str] = field(default_factory=dict)

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
    chain_of_residue: dict[ResidueId, str | None] = {}
    for residue in residues:
        chain = residue.id.chain_id
        chain_of_residue[residue.id] = (
            chain if chain in by_chain else chain_of_nonprotein.get(chain)
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

    grid = _Grid(CONTACT_ANGSTROM)
    for residue in residues:
        if residue.id.chain_id in protein_chains and is_standard_protein_residue(residue):
            for atom in residue.atoms:
                if not _is_hydrogen(atom):
                    grid.add(atom, residue.id.chain_id)

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
                if _is_hydrogen(atom):
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


# --------------------------------------------------------------------------
# Heterogen counterparts
# --------------------------------------------------------------------------


def _pair_copies(analysis: OligomerAnalysis, residues: Sequence[ResidueRecord]) -> None:
    residue_by_id = {residue.id: residue for residue in residues}
    metals = [residue for residue in residues if is_metal_ion_residue(residue)]
    bound_waters = {residue.id for residue in metal_bound_waters(residues, metals)}

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
            analysis.unmatched[residue.id] = (
                f"no {residue.id.resname} with the same atoms sits in the same place "
                f"in chain {reference}"
            )


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
    """

    def shell(metal: ResidueRecord) -> set[tuple[ResidueId, str]]:
        return {
            (donor.id, atom.name) for donor, atom, _ in metal_donor_atoms(residues, metal)
        }

    for copy_id, reference_id in list(analysis.counterparts.items()):
        copy = residue_by_id.get(copy_id)
        reference = residue_by_id.get(reference_id)
        if copy is None or reference is None or not is_metal_ion_residue(copy):
            continue
        mapped = {
            (analysis.counterparts.get(donor), name) for donor, name in shell(copy)
        }
        expected = shell(reference)
        if mapped != expected:
            del analysis.counterparts[copy_id]
            missing = sorted(
                f"{donor.display()} {name}"
                for donor, name in expected - mapped
                if donor is not None
            )
            extra = sorted(
                f"{donor.display()} {name}"
                for donor, name in shell(copy)
                if (analysis.counterparts.get(donor), name) not in expected
            )
            detail = "; ".join(
                part
                for part in (
                    f"bound in {reference_id.display()} only: {', '.join(missing)}"
                    if missing
                    else "",
                    f"bound here only: {', '.join(extra)}" if extra else "",
                )
                if part
            )
            analysis.unmatched[copy_id] = (
                f"its first shell differs from {reference_id.display()}'s ({detail})"
            )

    # Which metal each ligand binds, if any.
    metals = [residue for residue in residues if is_metal_ion_residue(residue)]
    bound_to: dict[ResidueId, set[ResidueId]] = {}
    for metal in metals:
        for donor, _, _ in metal_donor_atoms(residues, metal):
            bound_to.setdefault(donor.id, set()).add(metal.id)
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


def _is_hydrogen(atom) -> bool:
    element = (atom.element or "").strip().upper()
    if element:
        return element in {"H", "D"}
    return atom.name.strip().startswith("H")


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
