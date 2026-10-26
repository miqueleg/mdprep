"""Detect what a deposited structure is missing.

A crystal structure almost never arrives complete: side chains with no density
are deposited truncated, and disordered loops are left out entirely. `tleap`
cannot build either, so a run that ignores them fails late, after the expensive
stages, with an error about an unrecognised residue.

Finding them up front lets the question be asked before any of that work
starts. Nothing here repairs anything; it only reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Iterable, Sequence

from mdprep.structure.classify import is_standard_protein_residue
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import ResidueId, ResidueRecord


# Heavy atoms of each standard residue, backbone included. OXT is omitted
# because it is present only at a chain terminus.
BACKBONE_HEAVY_ATOMS = frozenset({"N", "CA", "C", "O"})

SIDE_CHAIN_HEAVY_ATOMS: dict[str, frozenset[str]] = {
    "ALA": frozenset({"CB"}),
    "ARG": frozenset({"CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"}),
    "ASN": frozenset({"CB", "CG", "OD1", "ND2"}),
    "ASP": frozenset({"CB", "CG", "OD1", "OD2"}),
    "CYS": frozenset({"CB", "SG"}),
    "GLN": frozenset({"CB", "CG", "CD", "OE1", "NE2"}),
    "GLU": frozenset({"CB", "CG", "CD", "OE1", "OE2"}),
    "GLY": frozenset(),
    "HIS": frozenset({"CB", "CG", "ND1", "CD2", "CE1", "NE2"}),
    "ILE": frozenset({"CB", "CG1", "CG2", "CD1"}),
    "LEU": frozenset({"CB", "CG", "CD1", "CD2"}),
    "LYS": frozenset({"CB", "CG", "CD", "CE", "NZ"}),
    "MET": frozenset({"CB", "CG", "SD", "CE"}),
    "PHE": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"}),
    "PRO": frozenset({"CB", "CG", "CD"}),
    "SER": frozenset({"CB", "OG"}),
    "THR": frozenset({"CB", "OG1", "CG2"}),
    "TRP": frozenset(
        {"CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
    ),
    "TYR": frozenset({"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}),
    "VAL": frozenset({"CB", "CG1", "CG2"}),
}

# Protonation and disulfide variants differ only in hydrogens, so they share a
# heavy-atom complement with their parent residue.
PROTONATION_VARIANTS = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",
    "CYM": "CYS", "CYX": "CYS",
}

# A peptide C-N bond is about 1.33 A. Beyond this the chain is not continuous.
PEPTIDE_BOND_MAX_ANGSTROM = 1.5


@dataclass(frozen=True)
class IncompleteResidue:
    residue: ResidueRecord
    missing_atoms: tuple[str, ...]

    def describe(self) -> str:
        return f"{self.residue.id.display()} missing {', '.join(self.missing_atoms)}"


@dataclass(frozen=True)
class ChainBreak:
    before: ResidueRecord
    after: ResidueRecord
    reason: str

    def describe(self) -> str:
        return (
            f"{self.before.id.display()} -> {self.after.id.display()} ({self.reason})"
        )


def expected_heavy_atoms(resname: str) -> frozenset[str] | None:
    """Heavy atoms of a standard residue, or None if it is not one."""

    name = PROTONATION_VARIANTS.get(resname.strip().upper(), resname.strip().upper())
    side_chain = SIDE_CHAIN_HEAVY_ATOMS.get(name)
    if side_chain is None:
        return None
    return BACKBONE_HEAVY_ATOMS | side_chain


def missing_heavy_atoms(residue: ResidueRecord) -> tuple[str, ...]:
    """Heavy atoms a standard residue should have but does not."""

    expected = expected_heavy_atoms(residue.id.resname)
    if expected is None:
        return ()
    present = {
        atom.name.strip()
        for atom in residue.atoms
        if not is_hydrogen(atom)
    }
    return tuple(sorted(expected - present))


def incomplete_residues(
    residues: Sequence[ResidueRecord],
) -> list[IncompleteResidue]:
    """Standard protein residues with heavy atoms absent from the input."""

    found: list[IncompleteResidue] = []
    for residue in residues:
        if not is_standard_protein_residue(residue):
            continue
        missing = missing_heavy_atoms(residue)
        if missing:
            found.append(IncompleteResidue(residue, missing))
    return found


def chain_breaks(
    residues: Sequence[ResidueRecord],
    *,
    also_in_chain: Collection[ResidueId] = (),
) -> list[ChainBreak]:
    """Discontinuities between consecutive protein residues of one chain.

    A gap in residue numbering is reported even when the two ends happen to sit
    close together, because the numbering is what says residues are absent.
    ``also_in_chain`` are heterogens that are residues of a chain -- a
    selenomethionine -- which leave no gap where they are.
    """

    protein = [
        residue
        for residue in residues
        if is_standard_protein_residue(residue) or residue.id in also_in_chain
    ]
    breaks: list[ChainBreak] = []
    for before, after in zip(protein, protein[1:]):
        if before.id.chain_id != after.id.chain_id:
            continue
        gap = after.id.resid - before.id.resid
        if gap > 1:
            breaks.append(
                ChainBreak(before, after, f"{gap - 1} residue(s) absent by numbering")
            )
            continue
        if gap < 1:
            continue
        distance = _peptide_bond_length(before, after)
        if distance is not None and distance > PEPTIDE_BOND_MAX_ANGSTROM:
            breaks.append(
                ChainBreak(before, after, f"C-N distance {distance:.2f} A")
            )
    return breaks


def _peptide_bond_length(
    before: ResidueRecord, after: ResidueRecord
) -> float | None:
    carbon = _atom_named(before.atoms, "C")
    nitrogen = _atom_named(after.atoms, "N")
    if carbon is None or nitrogen is None:
        return None
    return (
        (carbon.x - nitrogen.x) ** 2
        + (carbon.y - nitrogen.y) ** 2
        + (carbon.z - nitrogen.z) ** 2
    ) ** 0.5


def _atom_named(atoms: Iterable[object], name: str) -> object | None:
    for atom in atoms:
        if atom.name.strip() == name:  # type: ignore[attr-defined]
            return atom
    return None
