"""The hydrogens a structure arrives with.

mdprep places every hydrogen of the system itself: tleap builds the protein's
and the waters' from its residue templates, the protonation stage decides the
titratable ones, and a ligand's come from its SMILES. Hydrogens already in the
input were placed by something else -- PyMOL, reduce, PDBFixer, a format
converter, a modelling program, a riding-model refinement -- under that
program's names and that program's idea of the protonation. Each of those has
broken a different stage: a chloride PyMOL protonated into HCl, a lysine NZ the
temporary hydrogenation would not give its third hydrogen back to, an
alpha-ketoglutarate PDBFixer protonated as the neutral acid of a dianion.

So they are removed as the structure is read, before anything looks at it,
and the removal is recorded. A ligand's are the exception: they are kept to
be compared with its SMILES, which the structure stage does atom by atom --
keeping them if they agree, replacing them if not.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Callable

from mdprep.structure.classify import is_standard_protein_residue, is_water_residue
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord, build_residues
from mdprep.structure.selectors import coerce_residue_selector, residue_matches_selector

if TYPE_CHECKING:
    from mdprep.config.models import LigandConfig, ManifestConfig


HYDROGEN_ELEMENTS = frozenset({"H", "D"})


def is_hydrogen(atom: AtomRecord) -> bool:
    """Whether an atom is a hydrogen, deuterium included.

    The element the reader settled on decides. Five stages each had their own
    test, and they disagreed: one read the element column and fell back on
    the name, so a mercury ion written without an element column was a
    hydrogen to it; another counted deuterium and the next did not. The
    reader already weighs the element column against the atom name and the
    one-atom-ion rule, so asking it again is the only answer that cannot
    contradict the element every other stage uses. The name is a fallback
    for an atom built without an element.
    """

    element = (atom.element or "").strip().upper()
    if element:
        return element in HYDROGEN_ELEMENTS
    return atom.name.strip().lstrip("0123456789").upper().startswith("H")


@dataclass(frozen=True)
class InputHydrogenRemoval:
    """What removing a structure's input hydrogens took out, and what it kept."""

    protein: int = 0
    water: int = 0
    other: int = 0
    # Residues other than protein and water, one entry each: the hydrogens a
    # ligand, an ion or a cofactor lost are the ones worth reading by name.
    other_residues: tuple[tuple[str, int], ...] = ()
    kept_residues: tuple[tuple[str, int], ...] = ()

    @property
    def removed(self) -> int:
        return self.protein + self.water + self.other

    @property
    def kept(self) -> int:
        return sum(count for _, count in self.kept_residues)

    def to_dict(self) -> dict[str, object]:
        return {
            "removed": self.removed,
            "protein": self.protein,
            "water": self.water,
            "other": self.other,
            "other_residues": [
                {"residue": residue, "hydrogens": count}
                for residue, count in self.other_residues
            ],
            "kept": self.kept,
            "kept_residues": [
                {"residue": residue, "hydrogens": count}
                for residue, count in self.kept_residues
            ],
        }


def strip_input_hydrogens(
    structure: PdbStructure,
    *,
    keep: Callable[[ResidueRecord], bool] = lambda residue: False,
) -> tuple[PdbStructure, InputHydrogenRemoval]:
    """The structure without its hydrogens, except in residues ``keep`` names.

    A residue made of nothing but hydrogens is left whole: removing its atoms
    would remove the residue, and a residue disappearing is a decision the
    heterogen checks downstream make out loud, not this.
    """

    removed_ids: set[int] = set()
    protein = water = other = 0
    other_residues: list[tuple[str, int]] = []
    kept_residues: list[tuple[str, int]] = []
    for residue in structure.residues:
        hydrogens = [atom for atom in residue.atoms if is_hydrogen(atom)]
        if not hydrogens or len(hydrogens) == len(residue.atoms):
            continue
        if keep(residue):
            kept_residues.append((residue.id.display(), len(hydrogens)))
            continue
        removed_ids.update(id(atom) for atom in hydrogens)
        if is_standard_protein_residue(residue):
            protein += len(hydrogens)
        elif is_water_residue(residue):
            water += len(hydrogens)
        else:
            other += len(hydrogens)
            other_residues.append((residue.id.display(), len(hydrogens)))
    removal = InputHydrogenRemoval(
        protein=protein,
        water=water,
        other=other,
        other_residues=tuple(other_residues),
        kept_residues=tuple(kept_residues),
    )
    if not removed_ids:
        return structure, removal

    atoms = [atom for atom in structure.atoms if id(atom) not in removed_ids]
    removed_atoms = [atom for atom in structure.atoms if id(atom) in removed_ids]
    removed_serials = {atom.serial for atom in removed_atoms if atom.serial is not None}
    removed_identities = {atom.atom_identity for atom in removed_atoms}
    return (
        replace(
            structure,
            atoms=atoms,
            residues=build_residues(atoms),
            conect_bonds={
                bond
                for bond in structure.conect_bonds
                if bond[0] not in removed_serials and bond[1] not in removed_serials
            },
            ter_after_serials=_ter_after_kept_atoms(structure, removed_ids),
            links=tuple(
                link
                for link in structure.links
                if link.first not in removed_identities
                and link.second not in removed_identities
            ),
        ),
        removal,
    )


def _ter_after_kept_atoms(structure: PdbStructure, removed_ids: set[int]) -> set[int]:
    """Chain boundaries, each on the last atom left before it.

    The atom before a TER is as often as not a hydrogen of the chain's last
    residue. Dropping the boundary with it would let the writer join the
    chain to whatever follows.
    """

    boundaries: set[int] = set()
    last_kept: int | None = None
    for atom in structure.atoms:
        if id(atom) not in removed_ids and atom.serial is not None:
            last_kept = atom.serial
        if atom.serial in structure.ter_after_serials and last_kept is not None:
            boundaries.add(last_kept)
    return boundaries


def ligand_keeps_input_hydrogens(ligand: "LigandConfig") -> bool:
    """Whether a ligand's input hydrogens survive the reading of the structure.

    They do unless the ligand says otherwise: the structure stage compares
    them with its SMILES and decides there, which it cannot do with
    hydrogens already thrown away. ``keep_input_hydrogens: false`` replaces
    them without looking.
    """

    return ligand.keep_input_hydrogens is not False


def input_hydrogen_keeper(manifest: "ManifestConfig") -> Callable[[ResidueRecord], bool]:
    """Which residues keep their input hydrogens under this manifest.

    Ligands are matched by their selectors rather than resolved, so the
    answer does not depend on which copy of the structure is asking; a
    selector that matches nothing or too much is the structure stage's to
    report, with the residues it found.
    """

    remove = manifest.structure.remove_input_hydrogens
    decisions = [
        (
            coerce_residue_selector(ligand.selector.model_dump()),
            ligand_keeps_input_hydrogens(ligand),
        )
        for ligand in manifest.ligands
    ]

    def keep(residue: ResidueRecord) -> bool:
        for selector, keeps in decisions:
            if residue_matches_selector(residue, selector):
                return keeps
        return not remove

    return keep


def remove_input_hydrogens(
    structure: PdbStructure, manifest: "ManifestConfig"
) -> tuple[PdbStructure, InputHydrogenRemoval]:
    """The structure as this manifest builds it, hydrogens-wise."""

    return strip_input_hydrogens(structure, keep=input_hydrogen_keeper(manifest))
