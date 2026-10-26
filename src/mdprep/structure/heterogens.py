"""Heterogens that are not ligands: modifications, glycans, additives.

A deposition's heterogens are not all molecules to parameterise. Some are
pieces of the protein -- a selenomethionine is a methionine labelled for
phasing, written as a heterogen because its name is not one of the twenty.
Some are bonded to it -- the sugars of a glycoprotein, a covalent
intermediate, a heme held by ester bonds. And many are what the crystal was
grown in: glycerol, sulfate, PEG. Offering each of these as a ligand, as
mdprep used to, asks a student for the SMILES and charge of fifteen
glycerols, and builds a sugar bonded to an asparagine as a free molecule at
bond length from it.

This sorts them, from the structure and from what the file declares about its
bonds (``structure/links.py``). It decides nothing: the planner asks about
each group, and a manifest says what was done with it.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import dist, floor
from typing import Iterable, Sequence

from mdprep.structure.classify import (
    STANDARD_AMINO_ACIDS,
    is_bulk_ion_residue,
    is_metal_ion_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.completeness import BACKBONE_HEAVY_ATOMS, PEPTIDE_BOND_MAX_ANGSTROM
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.links import StructureLink
from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord

ResidueKey = tuple[str, str, int, str | None]

# Standard parents of the modified amino acids a deposition most often holds,
# for files that do not say (a prediction, a file another program rewrote).
# A file's own MODRES or _pdbx_struct_mod_residue takes precedence.
MODIFIED_RESIDUE_PARENTS = {
    "MSE": "MET",  # selenomethionine
    "CSO": "CYS",  # S-hydroxycysteine
    "CSD": "CYS",  # cysteine sulfinic acid
    "OCS": "CYS",  # cysteine sulfonic acid
    "CME": "CYS",  # S,S-(2-hydroxyethyl)thiocysteine
    "CSX": "CYS",  # S-oxycysteine
    "SEP": "SER",  # phosphoserine
    "TPO": "THR",  # phosphothreonine
    "PTR": "TYR",  # phosphotyrosine
    "HYP": "PRO",  # hydroxyproline
    "KCX": "LYS",  # carbamylated lysine
    "LLP": "LYS",  # lysine-pyridoxal-5'-phosphate
    "MLY": "LYS",  # N-dimethyl-lysine
    "M3L": "LYS",  # N-trimethyl-lysine
    "ALY": "LYS",  # N-acetyl-lysine
    "PCA": "GLN",  # pyroglutamic acid
}

# Which of them are artefacts of how the crystal was made rather than of the
# protein: selenium is how a methionine is labelled to solve the phases, and
# the protein being simulated has a methionine there.
LABELLING_MODIFICATIONS = frozenset({"MSE"})

# Monosaccharides of N- and O-linked glycans, by chemical component name.
CARBOHYDRATE_RESNAMES = frozenset(
    {
        "NAG", "NDG", "NGA", "A2G", "MAN", "BMA", "GLC", "BGC", "GAL", "GLA",
        "GLB", "FUC", "FUL", "XYS", "XYP", "SIA", "SLB", "GCU", "BDP", "IDR",
        "ARA", "ARB", "AHR", "RAM", "RHA", "KDO", "GCS", "MMA", "Z9N",
    }
)

# Components of crystallisation and purification buffers, cryoprotectants
# and precipitants. A residue here is not necessarily an artefact -- a sulfate
# can sit where the substrate's phosphate binds -- which is why removing them
# is asked, with every one listed.
CRYSTALLISATION_ADDITIVES = {
    "GOL": "glycerol",
    "EDO": "ethylene glycol",
    "PEG": "di(ethylene glycol)",
    "PGE": "tri(ethylene glycol)",
    "PG4": "tetra(ethylene glycol)",
    "1PE": "penta(ethylene glycol)",
    "P6G": "hexa(ethylene glycol)",
    "2PE": "nona(ethylene glycol)",
    "12P": "dodeca(ethylene glycol)",
    "15P": "polyethylene glycol",
    "PE4": "polyethylene glycol",
    "PE5": "polyethylene glycol",
    "P33": "hepta(ethylene glycol)",
    "MPD": "2-methyl-2,4-pentanediol",
    "MRD": "2-methyl-2,4-pentanediol",
    "MOH": "methanol",
    "EOH": "ethanol",
    "IPA": "isopropanol",
    "DMS": "dimethyl sulfoxide",
    "ACT": "acetate",
    "ACY": "acetic acid",
    "FMT": "formate",
    "SO4": "sulfate",
    "PO4": "phosphate",
    "NO3": "nitrate",
    "NH4": "ammonium",
    "TRS": "Tris",
    "EPE": "HEPES",
    "MES": "MES",
    "BTB": "Bis-Tris",
    "B3P": "Bis-Tris propane",
    "CIT": "citrate",
    "FLC": "citrate",
    "MLI": "malonate",
    "IMD": "imidazole",
    "BME": "2-mercaptoethanol",
    "DTT": "dithiothreitol",
    "CAC": "cacodylate",
}

# A bond is at most this long; a heterogen's atom closer than this to the
# protein's is bonded to it. Used only for carbohydrates, which nothing else
# in a crystal sits this close to; for anything else a clash in a predicted
# complex is as short as a bond, so only a declared link counts.
BOND_MAX_ANGSTROM = 1.75


# Of the twenty amino acids only proline has a side chain bonded to its own
# backbone. A modified residue can close one -- a pyroglutamate's lactam joins
# CD to N -- and then the atoms it shares with its parent by name are not where
# the parent has them: kept, a glutamine's CD would sit 1.5 A from its own
# nitrogen. A side-chain atom this close to N or C is bonded to it; nothing
# unbonded is. Such a side chain is rebuilt, keeping only these atoms; CB stays
# because it carries the residue's chirality.
SIDE_CHAIN_BACKBONE_BOND_MAX_ANGSTROM = 2.0
KEPT_WHEN_SIDE_CHAIN_REBUILT = BACKBONE_HEAVY_ATOMS | {"OXT", "CB"}


def side_chain_closes_onto_backbone(
    atoms: Iterable[tuple[str, Sequence[float]]], parent: str
) -> bool:
    """Whether a side chain is bonded to its own backbone where its parent's is not.

    ``atoms`` are the residue's heavy atoms as (name, position) pairs.
    """

    if parent == "PRO":
        return False
    atoms = list(atoms)
    backbone = [position for name, position in atoms if name in {"N", "C"}]
    side_chain = [
        position for name, position in atoms if name not in KEPT_WHEN_SIDE_CHAIN_REBUILT
    ]
    return any(
        dist(left, right) < SIDE_CHAIN_BACKBONE_BOND_MAX_ANGSTROM
        for left in side_chain
        for right in backbone
    )


@dataclass(frozen=True)
class ModifiedResidue:
    """A heterogen that is a residue of a polymer chain."""

    residue: ResidueRecord
    # The standard amino acid it is a modified form of, if known.
    parent: str | None
    # Where the parent came from: the file, or mdprep's own table.
    parent_source: str | None

    @property
    def side_chain_rebuilt(self) -> bool:
        """Whether replacing it builds its side chain again instead of keeping it."""

        return self.parent is not None and side_chain_closes_onto_backbone(
            (
                (atom.name, (atom.x, atom.y, atom.z))
                for atom in self.residue.atoms
                if not is_hydrogen(atom)
            ),
            self.parent,
        )

    def describe(self) -> str:
        if self.parent is None:
            return f"{self.residue.id.display()}, no standard parent known"
        return f"{self.residue.id.display()} -> {self.parent}"


@dataclass(frozen=True)
class BondedHeterogen:
    """A heterogen bonded to the protein, with the bonds that say so."""

    residue: ResidueRecord
    bonds: tuple[str, ...]


@dataclass(frozen=True)
class HeterogenGroups:
    modified_residues: tuple[ModifiedResidue, ...] = ()
    # Carbohydrates bonded to the protein, and every sugar bonded to those.
    glycans: tuple[BondedHeterogen, ...] = ()
    # Any other heterogen bonded to the protein, and what is bonded to it.
    covalent: tuple[BondedHeterogen, ...] = ()
    additives: tuple[ResidueRecord, ...] = ()
    # Bonds between two protein residues other than a peptide bond or a
    # disulfide: a crosslink tleap does not build.
    crosslinks: tuple[str, ...] = ()
    not_ligands: frozenset[ResidueId] = field(default_factory=frozenset)


def classify_heterogens(
    residues: Sequence[ResidueRecord],
    links: Iterable[StructureLink] = (),
    modified_residue_parents: dict[ResidueKey, str] | None = None,
    *,
    metal_donor_ids: frozenset[ResidueId] | set[ResidueId] = frozenset(),
) -> HeterogenGroups:
    """Sort a structure's heterogens into the groups the planner asks about.

    ``metal_donor_ids`` are residues that coordinate a metal. One of those is
    part of a metal site whatever its name, so it is never offered for
    removal as an additive.
    """

    declared_parents = {
        key: parent
        for key, parent in (modified_residue_parents or {}).items()
        if key[1] != parent
    }
    by_key = {residue_key(residue): residue for residue in residues}
    protein = [residue for residue in residues if is_standard_protein_residue(residue)]
    candidates = [
        residue
        for residue in residues
        if not is_standard_protein_residue(residue)
        and not is_water_residue(residue)
        and not is_metal_ion_residue(residue)
        and not is_bulk_ion_residue(residue)
    ]

    modified = _modified_residues(candidates, protein, links, declared_parents, by_key)
    in_chain = {residue_key(item.residue) for item in modified}
    polymer = {residue_key(residue) for residue in protein} | in_chain
    heterogens = {
        residue_key(residue): residue
        for residue in candidates
        if residue_key(residue) not in in_chain
    }

    attached: dict[ResidueKey, list[str]] = defaultdict(list)
    edges: dict[ResidueKey, set[ResidueKey]] = defaultdict(set)
    crosslinks: list[str] = []
    for link in links:
        if link.kind != "covalent":
            continue
        first, second = link.residues()
        if first in polymer and second in polymer:
            if not _is_peptide_bond(link):
                crosslinks.append(_describe_link(link))
            continue
        if first in heterogens and second in heterogens:
            edges[first].add(second)
            edges[second].add(first)
        elif first in heterogens and second in polymer:
            attached[first].append(_describe_link(link))
        elif second in heterogens and first in polymer:
            attached[second].append(_describe_link(link))

    # Sugars bonded by distance, for a file that declares no links.
    sugars = [
        residue
        for key, residue in heterogens.items()
        if residue.id.resname.strip().upper() in CARBOHYDRATE_RESNAMES
    ]
    if sugars:
        protein_grid = _AtomGrid(
            atom
            for residue in residues
            if residue_key(residue) in polymer
            for atom in residue.atoms
            if _heavy(atom)
        )
        sugar_grid = _AtomGrid(atom for residue in sugars for atom in residue.atoms if _heavy(atom))
        for sugar in sugars:
            key = residue_key(sugar)
            for atom in sugar.atoms:
                if not _heavy(atom):
                    continue
                for other, distance in protein_grid.near(atom, BOND_MAX_ANGSTROM):
                    bond = (
                        f"{_atom_label(atom)} -- {_atom_label(other)} "
                        f"({distance:.2f} A apart)"
                    )
                    if not any(
                        _atom_label(atom) in item and _atom_label(other) in item
                        for item in attached[key]
                    ):
                        attached[key].append(bond)
                for other, _ in sugar_grid.near(atom, BOND_MAX_ANGSTROM):
                    if other.residue_key != key:
                        edges[key].add(other.residue_key)
                        edges[other.residue_key].add(key)

    glycans: list[BondedHeterogen] = []
    covalent: list[BondedHeterogen] = []
    claimed: set[ResidueKey] = set()
    for key in heterogens:
        if key not in attached or key in claimed:
            continue
        component = _component(key, edges)
        claimed |= component
        members = [heterogens[member] for member in heterogens if member in component]
        is_glycan = all(
            member.id.resname.strip().upper() in CARBOHYDRATE_RESNAMES for member in members
        )
        target = glycans if is_glycan else covalent
        for member in members:
            target.append(
                BondedHeterogen(
                    residue=member,
                    bonds=tuple(attached.get(residue_key(member), ())),
                )
            )

    additives = [
        residue
        for key, residue in heterogens.items()
        if key not in claimed
        and residue.id.resname.strip().upper() in CRYSTALLISATION_ADDITIVES
        and residue.id not in metal_donor_ids
    ]
    not_ligands = frozenset(
        [item.residue.id for item in modified]
        + [item.residue.id for item in glycans]
        + [item.residue.id for item in covalent]
    )
    return HeterogenGroups(
        modified_residues=tuple(modified),
        glycans=tuple(glycans),
        covalent=tuple(covalent),
        additives=tuple(additives),
        crosslinks=tuple(crosslinks),
        not_ligands=not_ligands,
    )


def residue_key(residue: ResidueRecord) -> ResidueKey:
    return (residue.id.chain_id, residue.id.resname, residue.id.resid, residue.id.icode)


def _modified_residues(
    candidates: Sequence[ResidueRecord],
    protein: Sequence[ResidueRecord],
    links: Iterable[StructureLink],
    declared_parents: dict[ResidueKey, str],
    by_key: dict[ResidueKey, ResidueRecord],
) -> list[ModifiedResidue]:
    """Heterogens that are residues of a chain: declared so, or peptide-bonded.

    A free amino acid -- a tryptophan substrate in a halogenase -- carries
    the same backbone atoms, so the backbone alone does not make a residue
    part of a chain; a peptide bond to one does.
    """

    peptide_bonded: set[ResidueKey] = set()
    for link in links:
        if link.kind == "covalent" and _is_peptide_bond(link):
            peptide_bonded.update(link.residues())
    backbone = {
        "C": _AtomGrid(_named(residue, "C") for residue in protein),
        "N": _AtomGrid(_named(residue, "N") for residue in protein),
    }
    found: list[ModifiedResidue] = []
    for residue in candidates:
        key = residue_key(residue)
        bonded = key in declared_parents or key in peptide_bonded
        if not bonded:
            nitrogen, carbon = _named(residue, "N"), _named(residue, "C")
            if _named(residue, "CA") is not None and (nitrogen or carbon):
                bonded = bool(
                    (nitrogen is not None and backbone["C"].near(nitrogen, PEPTIDE_BOND_MAX_ANGSTROM + 0.25))
                    or (carbon is not None and backbone["N"].near(carbon, PEPTIDE_BOND_MAX_ANGSTROM + 0.25))
                )
        if not bonded:
            continue
        name = residue.id.resname.strip().upper()
        if name in STANDARD_AMINO_ACIDS:
            # A standard residue the file wrote as a heterogen.
            parent, source = name, "its own name"
        elif key in declared_parents:
            parent, source = declared_parents[key], "the file's MODRES records"
        elif name in MODIFIED_RESIDUE_PARENTS:
            parent, source = MODIFIED_RESIDUE_PARENTS[name], "mdprep's table"
        else:
            parent, source = None, None
        if parent is not None and parent not in STANDARD_AMINO_ACIDS:
            parent, source = None, None
        found.append(ModifiedResidue(residue=residue, parent=parent, parent_source=source))
    return found


def _is_peptide_bond(link: StructureLink) -> bool:
    names = {link.first[4], link.second[4]}
    return names == {"C", "N"}


def _describe_link(link: StructureLink) -> str:
    first = _identity_label(link.first)
    second = _identity_label(link.second)
    return f"{first} -- {second} (declared in the file)"


def _identity_label(identity: tuple[str, str, int, str | None, str]) -> str:
    chain, resname, resid, icode, atom = identity
    return f"{chain or '<blank>'}:{resname}{resid}{icode or ''} {atom}"


def _atom_label(atom: AtomRecord) -> str:
    return _identity_label(atom.atom_identity)


def _component(start: ResidueKey, edges: dict[ResidueKey, set[ResidueKey]]) -> set[ResidueKey]:
    seen = {start}
    stack = [start]
    while stack:
        for neighbour in edges.get(stack.pop(), ()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return seen


def _named(residue: ResidueRecord, name: str) -> AtomRecord | None:
    return next((atom for atom in residue.atoms if atom.name == name), None)


def _heavy(atom: AtomRecord) -> bool:
    return not is_hydrogen(atom)


class _AtomGrid:
    """Atoms binned in cubes, for neighbour searches without numpy.

    The web builder runs this under Pyodide, where there is no numpy, and a
    glycoprotein's sugars against every protein atom is millions of
    distances done naively.
    """

    CELL = 2.0

    def __init__(self, atoms: Iterable[AtomRecord | None]):
        self._cells: dict[tuple[int, int, int], list[AtomRecord]] = defaultdict(list)
        for atom in atoms:
            if atom is not None:
                self._cells[self._cell(atom)].append(atom)

    def _cell(self, atom: AtomRecord) -> tuple[int, int, int]:
        return (floor(atom.x / self.CELL), floor(atom.y / self.CELL), floor(atom.z / self.CELL))

    def near(self, atom: AtomRecord, cutoff: float) -> list[tuple[AtomRecord, float]]:
        cx, cy, cz = self._cell(atom)
        found: list[tuple[AtomRecord, float]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in self._cells.get((cx + dx, cy + dy, cz + dz), ()):
                        if other is atom:
                            continue
                        distance = (
                            (atom.x - other.x) ** 2
                            + (atom.y - other.y) ** 2
                            + (atom.z - other.z) ** 2
                        ) ** 0.5
                        if distance <= cutoff:
                            found.append((other, distance))
        return found
