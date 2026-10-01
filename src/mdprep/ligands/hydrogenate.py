"""Add explicit hydrogens to a ligand from a user-supplied SMILES.

A crystal structure carries no ligand hydrogens, and PDB coordinates encode
neither bond orders nor protonation, so mdprep refuses to parameterize a
hydrogen-free ligand rather than guess one. That refusal is correct, but it
leaves a deposited structure unusable without an external step.

A SMILES closes the gap without guessing: the user states the protonation they
intend, including the charge, and the hydrogens follow from it. mdprep only
places them.

Two rules govern this module:

* A ligand that already carries hydrogens is never touched. The coordinates
  someone else prepared are not second-guessed.
* Adding hydrogens always warns. Placement is geometric, and a tautomer or a
  rotamer that is wrong for the active site will look perfectly reasonable, so
  the result is something a person has to look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mdprep.structure.models import AtomRecord, ResidueRecord


class LigandHydrogenationError(ValueError):
    """Raised when hydrogens cannot be placed without guessing."""


@dataclass(frozen=True)
class HydrogenationResult:
    residue: ResidueRecord
    added_hydrogens: int
    smiles: str
    warnings: tuple[str, ...]
    skipped: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "residue": self.residue.id.display(),
            "added_hydrogens": self.added_hydrogens,
            "smiles": self.smiles,
            "skipped": self.skipped,
            "warnings": list(self.warnings),
        }


def count_hydrogens(residue: ResidueRecord) -> int:
    return sum(1 for atom in residue.atoms if _is_hydrogen(atom))


def _is_hydrogen(atom: AtomRecord) -> bool:
    element = (atom.element or "").strip().upper()
    if element:
        return element in {"H", "D"}
    return atom.name.strip().startswith(("H", "D"))


def is_already_protonated(residue: ResidueRecord) -> bool:
    """Whether the input already carries explicit ligand hydrogens.

    Any hydrogen at all counts. A partially hydrogenated ligand is a different
    problem, caught downstream by the formula check, and is not something this
    module should quietly finish off.
    """

    return count_hydrogens(residue) > 0


def hydrogen_names_for(parent: AtomRecord, count: int, taken: set[str]) -> list[str]:
    """Names for hydrogens on one heavy atom, unique within the residue.

    Derived from the parent so that the result is readable: C12 gives H12,
    H12A, H12B. PDB atom names are four characters, so a long parent name falls
    back to a plain serial.
    """

    stem = parent.name.strip()
    stem = stem[1:] if stem[:1].isalpha() else stem
    names: list[str] = []
    suffixes = ["", "A", "B", "C", "D", "E", "F"]
    for index in range(count):
        candidate = ""
        for suffix in suffixes[index:] + suffixes:
            trial = f"H{stem}{suffix}"
            if len(trial) <= 4 and trial not in taken:
                candidate = trial
                break
        if not candidate:
            serial = 1
            while f"H{serial}" in taken or len(f"H{serial}") > 4:
                serial += 1
            candidate = f"H{serial}"
        taken.add(candidate)
        names.append(candidate)
    return names


def add_ligand_hydrogens(
    residue: ResidueRecord,
    smiles: str,
    *,
    ligand_id: str,
) -> HydrogenationResult:
    """Place explicit hydrogens on a ligand according to ``smiles``.

    Heavy atoms keep their names, their order and their coordinates; hydrogens
    are appended after them. Anything that would disturb a heavy atom is an
    error rather than a silent correction.
    """

    if is_already_protonated(residue):
        return HydrogenationResult(
            residue=residue,
            added_hydrogens=0,
            smiles=smiles,
            skipped=True,
            warnings=(
                f"Ligand {ligand_id} ({residue.id.display()}) already carries "
                f"{count_hydrogens(residue)} explicit hydrogen(s); mdprep left it "
                "unchanged.",
            ),
        )

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - exercised by the skip guard
        raise LigandHydrogenationError(
            f"Ligand {ligand_id} needs hydrogens added from a SMILES, which "
            "requires RDKit. Install RDKit, or add the hydrogens yourself and "
            "remove ligands[].smiles."
        ) from exc

    heavy_atoms = [atom for atom in residue.atoms if not _is_hydrogen(atom)]
    block = _pdb_block(heavy_atoms)

    molecule = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    if molecule is None:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: RDKit could not read the residue coordinates."
        )
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: ligands[].smiles is not a valid SMILES: {smiles!r}"
        )

    template_heavy = template.GetNumAtoms()
    if template_heavy != len(heavy_atoms):
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the SMILES describes {template_heavy} heavy "
            f"atom(s) but the residue has {len(heavy_atoms)}. The SMILES must "
            "match the molecule in the structure exactly."
        )

    try:
        molecule = AllChem.AssignBondOrdersFromTemplate(template, molecule)
    except Exception as exc:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the SMILES could not be matched onto the "
            f"residue's heavy-atom connectivity ({exc}). Check that it describes "
            "the same molecule, including its charge."
        ) from exc

    molecule = Chem.AddHs(molecule, addCoords=True)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the hydrogenated molecule is not chemically "
            f"valid ({exc})."
        ) from exc

    new_atoms, added = _collect_new_hydrogens(molecule, heavy_atoms, residue)
    if added == 0:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the SMILES {smiles!r} implies no hydrogens, so "
            "there is nothing to add. Declare the ligand hydrogen-free with "
            "expected_formula instead."
        )

    updated = ResidueRecord(
        id=residue.id,
        atoms=list(residue.atoms) + new_atoms,
        record_names=set(residue.record_names),
        original_index=residue.original_index,
    )
    _verify_heavy_atoms_unchanged(residue, updated, ligand_id=ligand_id)

    return HydrogenationResult(
        residue=updated,
        added_hydrogens=added,
        smiles=smiles,
        warnings=(
            f"WARNING [LIGAND_HYDROGENS_ADDED] Ligand {ligand_id} "
            f"({residue.id.display()}) had no explicit hydrogens; mdprep placed "
            f"{added} from the supplied SMILES {smiles!r}. Their positions are "
            "geometric, and the tautomer and charge are whatever the SMILES "
            "says. Inspect this ligand before trusting the charges derived from "
            "it.",
        ),
    )


def _pdb_block(atoms: Sequence[AtomRecord]) -> str:
    from mdprep.structure.writer import format_atom_record

    return "".join(format_atom_record(atom) for atom in atoms) + "END\n"


def _collect_new_hydrogens(
    molecule: object,
    heavy_atoms: Sequence[AtomRecord],
    residue: ResidueRecord,
) -> tuple[list[AtomRecord], int]:
    """Turn RDKit's added hydrogens into atom records attached to their parent."""

    conformer = molecule.GetConformer()  # type: ignore[attr-defined]
    taken = {atom.name.strip() for atom in residue.atoms}
    by_parent: dict[int, list[int]] = {}
    for atom in molecule.GetAtoms():  # type: ignore[attr-defined]
        if atom.GetAtomicNum() != 1 or atom.GetIdx() < len(heavy_atoms):
            continue
        neighbours = atom.GetNeighbors()
        if not neighbours:
            continue
        by_parent.setdefault(neighbours[0].GetIdx(), []).append(atom.GetIdx())

    template = heavy_atoms[0]
    created: list[AtomRecord] = []
    for parent_index in sorted(by_parent):
        parent = heavy_atoms[parent_index]
        indices = by_parent[parent_index]
        names = hydrogen_names_for(parent, len(indices), taken)
        for name, index in zip(names, indices):
            position = conformer.GetAtomPosition(index)
            created.append(
                AtomRecord(
                    serial=None,
                    name=name,
                    altloc=None,
                    resname=residue.id.resname,
                    chain_id=residue.id.chain_id,
                    resid=residue.id.resid,
                    icode=residue.id.icode,
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                    occupancy=1.0,
                    bfactor=0.0,
                    element="H",
                    record_name=template.record_name,
                    original_line="",
                )
            )
    return created, len(created)


def _verify_heavy_atoms_unchanged(
    before: ResidueRecord, after: ResidueRecord, *, ligand_id: str
) -> None:
    """Heavy atom names, order and coordinates must survive untouched."""

    original = [atom for atom in before.atoms if not _is_hydrogen(atom)]
    produced = [atom for atom in after.atoms if not _is_hydrogen(atom)]
    if len(original) != len(produced):
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: hydrogenation changed the heavy-atom count "
            f"from {len(original)} to {len(produced)}."
        )
    for first, second in zip(original, produced):
        if first.name != second.name:
            raise LigandHydrogenationError(
                f"Ligand {ligand_id}: hydrogenation renamed heavy atom "
                f"{first.name!r} to {second.name!r}."
            )
        if (first.x, first.y, first.z) != (second.x, second.y, second.z):
            raise LigandHydrogenationError(
                f"Ligand {ligand_id}: hydrogenation moved heavy atom {first.name}."
            )
