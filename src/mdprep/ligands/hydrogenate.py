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


# Bond lengths for the orders a ligand SMILES can assign, with the spread a
# refined structure is allowed. These are the small-molecule values the
# monomer dictionaries restrain towards, so a 2 A protein structure sits close
# to them even though its density does not resolve them independently.
BOND_LENGTH_ANGSTROM: dict[tuple[str, str, int], float] = {
    ("C", "C", 1): 1.52, ("C", "C", 2): 1.34, ("C", "C", 3): 1.20,
    ("C", "N", 1): 1.47, ("C", "N", 2): 1.28, ("C", "N", 3): 1.16,
    ("C", "O", 1): 1.36, ("C", "O", 2): 1.22,
    ("C", "S", 1): 1.81, ("C", "S", 2): 1.68,
    ("N", "N", 1): 1.45, ("N", "N", 2): 1.25,
    ("N", "O", 1): 1.40, ("N", "O", 2): 1.21,
}

# A single bond between two conjugated centres is shorter than one between
# saturated atoms, because the pi system reaches across it: an amide C-N
# refines near 1.33 A and an adenine's C6-N6 near 1.34, against 1.47 for an
# amine; the C4a-C10a bond inside a flavin near 1.39, against 1.52 for a
# C-C single bond. Measured against the saturated lengths, every amide,
# every amino-heterocycle and every flavin looked like a wrong tautomer.
CONJUGATED_SINGLE_BOND_ANGSTROM: dict[tuple[str, str], float] = {
    ("C", "C"): 1.46,
    ("C", "N"): 1.36,
}

# A bond has to miss its own order by this much before it is questioned, and
# be this much closer to a different one, before the geometry is taken to
# contradict the SMILES rather than merely to be loosely refined.
BOND_LENGTH_TOLERANCE_ANGSTROM = 0.10
BOND_LENGTH_MARGIN_ANGSTROM = 0.06

# Groups whose real geometry is symmetric while any SMILES has to write them
# as alternating single and double bonds. Their formal orders say nothing
# about their lengths, so they are not checked.
DELOCALISED_SMARTS = (
    "[CX3](=[OX1])[OX1-]",        # carboxylate
    "[CX3](=[OX1])[OX2H1]",       # carboxylic acid, often refined symmetric
    "[NX3+](=[OX1])[OX1-]",       # nitro
    "[NX3][CX3](=[NX3+])[NX3]",   # guanidinium
    "[PX4](=[OX1])([OX1-])",      # phosphate
    "[SX4](=[OX1])(=[OX1])",      # sulfonyl
)


def geometry_contradicts_bond_orders(
    molecule: object, residue: ResidueRecord
) -> list[str]:
    """Bonds whose length disagrees with the order the SMILES gave them.

    The coordinates do carry the tautomer: a ketone C=O refines near 1.22 A
    and an enol C-OH near 1.36, so a SMILES naming the wrong one can be caught
    even though formula, charge and connectivity cannot tell them apart.

    Aromatic and delocalised bonds are skipped, because their formal orders
    are a notation rather than a claim about length.
    """

    from rdkit import Chem

    conformer = molecule.GetConformer()
    skip: set[int] = set()
    for pattern in DELOCALISED_SMARTS:
        query = Chem.MolFromSmarts(pattern)
        if query is None:
            continue
        for match in molecule.GetSubstructMatches(query):
            members = set(match)
            for bond in molecule.GetBonds():
                if bond.GetBeginAtomIdx() in members and bond.GetEndAtomIdx() in members:
                    skip.add(bond.GetIdx())

    complaints: list[str] = []
    for bond in molecule.GetBonds():
        if bond.GetIdx() in skip or bond.GetIsAromatic():
            continue
        order = int(bond.GetBondTypeAsDouble())
        if order not in (1, 2, 3):
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        pair = tuple(sorted((begin.GetSymbol(), end.GetSymbol())))
        assigned = BOND_LENGTH_ANGSTROM.get((*pair, order))
        if assigned is None:
            continue
        if order == 1 and _conjugated_single_bond(begin, end):
            assigned = CONJUGATED_SINGLE_BOND_ANGSTROM[pair]
        start = conformer.GetAtomPosition(begin.GetIdx())
        finish = conformer.GetAtomPosition(end.GetIdx())
        observed = (
            (start.x - finish.x) ** 2
            + (start.y - finish.y) ** 2
            + (start.z - finish.z) ** 2
        ) ** 0.5
        if abs(observed - assigned) <= BOND_LENGTH_TOLERANCE_ANGSTROM:
            continue
        better = [
            (abs(observed - length), other)
            for (left, right, other), length in BOND_LENGTH_ANGSTROM.items()
            if (left, right) == pair
            and other != order
            and abs(observed - length) + BOND_LENGTH_MARGIN_ANGSTROM
            < abs(observed - assigned)
        ]
        if not better:
            continue
        _, suggested = min(better)
        complaints.append(
            f"{_atom_label(begin)}-{_atom_label(end)} is {observed:.3f} A, "
            f"which fits bond order {suggested} ({BOND_LENGTH_ANGSTROM[(*pair, suggested)]:.2f} A) "
            f"rather than the {order} the SMILES gives it "
            f"({assigned:.2f} A)"
        )
    return complaints


def _conjugated_single_bond(begin: object, end: object) -> bool:
    """A C-C or C-N single bond that a pi system reaches across.

    C-C between two sp2 carbons; C-N between an sp2 carbon and a neutral
    nitrogen with three neighbours, whose lone pair conjugates -- an amide,
    an amino group on a ring, the ring nitrogen of a dihydropyridine. Not
    C-O: an enol's C-OH against a ketone's C=O is the tautomer error the
    check exists to catch.
    """

    from rdkit import Chem

    def sp2(atom: object) -> bool:
        return atom.GetIsAromatic() or atom.GetHybridization() == Chem.HybridizationType.SP2

    symbols = {begin.GetSymbol(), end.GetSymbol()}
    if symbols == {"C"}:
        return sp2(begin) and sp2(end)
    if symbols == {"C", "N"}:
        carbon, nitrogen = (begin, end) if begin.GetSymbol() == "C" else (end, begin)
        return (
            sp2(carbon)
            and nitrogen.GetFormalCharge() == 0
            and nitrogen.GetDegree() + nitrogen.GetTotalNumHs() == 3
        )
    return False


def _atom_label(atom: object) -> str:
    info = atom.GetPDBResidueInfo()
    if info is not None and info.GetName().strip():
        return info.GetName().strip()
    return f"{atom.GetSymbol()}{atom.GetIdx()}"


def add_ligand_hydrogens(
    residue: ResidueRecord,
    smiles: str,
    *,
    ligand_id: str,
    net_charge: int | None = None,
) -> HydrogenationResult:
    """Place explicit hydrogens on a ligand according to ``smiles``.

    Heavy atoms keep their names, their order and their coordinates; hydrogens
    are appended after them. Anything that would disturb a heavy atom is an
    error rather than a silent correction.

    ``net_charge`` is the charge the manifest declares. The SMILES decides the
    protonation, so the two must agree: a SMILES one proton out from what was
    declared still matches the same heavy atoms and would otherwise be
    accepted, leaving the hydrogens and the charge describing different
    molecules.
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

    if net_charge is not None:
        smiles_charge = Chem.GetFormalCharge(template)
        if smiles_charge != net_charge:
            raise LigandHydrogenationError(
                f"Ligand {ligand_id}: ligands[].smiles has a formal charge of "
                f"{smiles_charge:+d} but net_charge declares {net_charge:+d}. "
                "The SMILES sets the protonation, so one of them is describing "
                "a different molecule; the hydrogens placed from this SMILES "
                "would not match the charge the rest of the run uses."
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

    contradictions = geometry_contradicts_bond_orders(molecule, residue)
    if contradictions:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the bond orders this SMILES implies do not "
            "match the geometry in the structure. The coordinates carry the "
            "tautomer even though the formula and the charge do not, so this "
            "is most likely the wrong tautomer or the wrong protonation site: "
            + "; ".join(contradictions)
            + "."
        )

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
