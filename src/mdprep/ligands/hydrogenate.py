"""Add explicit hydrogens to a ligand from a user-supplied SMILES.

A crystal structure carries no ligand hydrogens, and PDB coordinates encode
neither bond orders nor protonation, so mdprep refuses to parameterize a
hydrogen-free ligand rather than guess one. That refusal is correct, but it
leaves a deposited structure unusable without an external step.

A SMILES closes the gap without guessing: the user states the protonation they
intend, including the charge, and the hydrogens follow from it. mdprep only
places them.

Two rules govern this module:

* The SMILES is the protonation, whether or not the residue already has
  hydrogens. Ones it has are compared with the SMILES heavy atom by heavy
  atom and kept, positions and all, only if they agree; a docking program's
  polar hydrogens or another program's idea of the protonation are replaced,
  and the disagreement is reported. A ligand that already had hydrogens used
  to be left untouched, which is how a docked substrate with two of its
  twenty-five reached antechamber with an odd number of electrons.
* Adding hydrogens always warns. Placement is geometric, and a tautomer or a
  rotamer that is wrong for the active site will look perfectly reasonable, so
  the result is something a person has to look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, ResidueRecord


class LigandHydrogenationError(ValueError):
    """Raised when hydrogens cannot be placed without guessing."""


@dataclass(frozen=True)
class HydrogenationResult:
    residue: ResidueRecord
    added_hydrogens: int
    smiles: str
    warnings: tuple[str, ...]
    # The residue's own hydrogens were the SMILES's and were kept as they are.
    skipped: bool = False
    # Hydrogens the residue arrived with that were not the SMILES's, and so
    # were replaced: how many, and where they disagreed with it.
    replaced_hydrogens: int = 0
    differences: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "residue": self.residue.id.display(),
            "added_hydrogens": self.added_hydrogens,
            "smiles": self.smiles,
            "skipped": self.skipped,
            "replaced_hydrogens": self.replaced_hydrogens,
            "differences": list(self.differences),
            "warnings": list(self.warnings),
        }


def count_hydrogens(residue: ResidueRecord) -> int:
    return sum(1 for atom in residue.atoms if is_hydrogen(atom))


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

# The 24-atom core of a porphyrin, as in heme. A SMILES draws two of its
# pyrroles as pyrrolides, whose beta-beta bond is single, and two as
# pyrrolenines, whose beta-beta bond is double; which two is the writer's
# choice, and the real ring has all four alike. Every carbon is
# three-connected, so a chlorin's saturated pyrrole is still checked.
PORPHYRIN_CORE_SMARTS = (
    "[#6X3]~1~[#6X3]~[#6X3]~2~[#7]~[#6X3]~1~[#6X3]~[#6X3]~3~[#6X3]~[#6X3]"
    "~[#6X3](~[#7]~3)~[#6X3]~[#6X3]~4~[#6X3]~[#6X3]~[#6X3](~[#7]~4)~[#6X3]"
    "~[#6X3]~5~[#6X3]~[#6X3]~[#6X3](~[#6X3]~2)~[#7]~5"
)

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
    PORPHYRIN_CORE_SMARTS,
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
    keep_input_hydrogens: bool | None = None,
) -> HydrogenationResult:
    """Give a ligand the hydrogens ``smiles`` states.

    The SMILES is used whether or not the residue already has hydrogens. Ones
    it has are compared with the SMILES first, heavy atom by heavy atom: if
    every heavy atom has the hydrogens the SMILES gives it, they are kept as
    they are, names and positions included. If not -- a docking program's
    polar hydrogens, PDBFixer's neutral acid -- they are replaced by the
    SMILES's and every disagreement is reported, unless
    ``keep_input_hydrogens`` is true, which makes a disagreement an error.

    Heavy atoms keep their names, their order and their coordinates; placed
    hydrogens come after them. Anything that would disturb a heavy atom is an
    error rather than a silent correction.

    ``net_charge`` is the charge the manifest declares. The SMILES decides the
    protonation, so the two must agree: a SMILES one proton out from what was
    declared still matches the same heavy atoms and would otherwise be
    accepted, leaving the hydrogens and the charge describing different
    molecules.
    """

    heavy_atoms = [atom for atom in residue.atoms if not is_hydrogen(atom)]
    molecule = _match_smiles_onto_residue(
        residue, heavy_atoms, smiles, ligand_id=ligand_id, net_charge=net_charge
    )
    display = residue.id.display()

    input_hydrogens = count_hydrogens(residue)
    differences: list[str] = []
    if input_hydrogens:
        differences = _hydrogen_differences(residue, heavy_atoms, molecule)
        if not differences:
            return HydrogenationResult(
                residue=residue,
                added_hydrogens=0,
                smiles=smiles,
                skipped=True,
                warnings=(
                    f"Ligand {ligand_id} ({display}) kept its {input_hydrogens} "
                    "input hydrogen(s); each heavy atom has the hydrogens the "
                    f"SMILES {smiles!r} gives it.",
                ),
            )
        if keep_input_hydrogens:
            raise LigandHydrogenationError(
                f"Ligand {ligand_id} ({display}) keeps its input hydrogens "
                f"(keep_input_hydrogens: true), but they are not the SMILES's "
                f"{smiles!r}: "
                + "; ".join(differences)
                + ". The SMILES states the protonation, so these hydrogens "
                "describe a different molecule -- often the protonation another "
                "program chose. Let mdprep replace them with the SMILES's (leave "
                "keep_input_hydrogens unset), or correct whichever of the two is "
                "wrong."
            )

    from rdkit import Chem

    molecule = Chem.AddHs(molecule, addCoords=True)
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:
        raise LigandHydrogenationError(
            f"Ligand {ligand_id}: the hydrogenated molecule is not chemically "
            f"valid ({exc})."
        ) from exc

    # Placed on the heavy atoms alone: hydrogens that disagreed with the
    # SMILES go, and their names are free again.
    bare = ResidueRecord(
        id=residue.id,
        atoms=heavy_atoms,
        record_names=set(residue.record_names),
        original_index=residue.original_index,
    )
    new_atoms, added = _collect_new_hydrogens(molecule, heavy_atoms, bare)
    updated = ResidueRecord(
        id=residue.id,
        atoms=heavy_atoms + new_atoms,
        record_names=set(residue.record_names),
        original_index=residue.original_index,
    )
    _verify_heavy_atoms_unchanged(residue, updated, ligand_id=ligand_id)

    if input_hydrogens:
        return HydrogenationResult(
            residue=updated,
            added_hydrogens=added,
            smiles=smiles,
            replaced_hydrogens=input_hydrogens,
            differences=tuple(differences),
            warnings=(
                f"WARNING [LIGAND_HYDROGENS_REPLACED] Ligand {ligand_id} "
                f"({display}) arrived with {input_hydrogens} hydrogen(s) that are "
                f"not the SMILES's {smiles!r}: "
                + "; ".join(differences)
                + f". mdprep replaced them with the SMILES's {added}. A docking "
                "program often writes only the polar hydrogens, and other "
                "programs protonate by their own rules; the SMILES is the "
                "protonation this run uses.",
            ),
        )
    if added == 0:
        return HydrogenationResult(
            residue=updated,
            added_hydrogens=0,
            smiles=smiles,
            warnings=(
                f"Ligand {ligand_id} ({display}) has no hydrogens, and its SMILES "
                f"{smiles!r} gives it none.",
            ),
        )
    return HydrogenationResult(
        residue=updated,
        added_hydrogens=added,
        smiles=smiles,
        warnings=(
            f"WARNING [LIGAND_HYDROGENS_ADDED] Ligand {ligand_id} "
            f"({display}) had no explicit hydrogens; mdprep placed "
            f"{added} from the supplied SMILES {smiles!r}. Their positions are "
            "geometric, and the tautomer and charge are whatever the SMILES "
            "says. Inspect this ligand before trusting the charges derived from "
            "it.",
        ),
    )


# Longest bond a hydrogen makes in a ligand: S-H is 1.34 A, the others are
# shorter. Beyond this a hydrogen is bonded to nothing of its residue.
_HYDROGEN_BOND_LENGTH_LIMIT_ANGSTROM = 1.45


def _hydrogen_differences(
    residue: ResidueRecord,
    heavy_atoms: Sequence[AtomRecord],
    molecule: object,
) -> list[str]:
    """Where the residue's hydrogens are not the SMILES's, heavy atom by heavy atom.

    A formula is not enough -- the keto and enol forms share one -- so each
    heavy atom's hydrogens, assigned by distance, are counted against the
    hydrogens the SMILES gives the atom matched to it. A hydrogen bonded to
    none of the residue's atoms is a difference too.
    """

    from rdkit import Chem

    expected = [
        sum(1 for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() == 1)
        for atom in Chem.AddHs(molecule).GetAtoms()  # type: ignore[call-overload]
        if atom.GetIdx() < len(heavy_atoms)
    ]
    observed = [0] * len(heavy_atoms)
    differences: list[str] = []
    for hydrogen in (atom for atom in residue.atoms if is_hydrogen(atom)):
        distance, index = min(
            (
                (
                    (
                        (hydrogen.x - heavy.x) ** 2
                        + (hydrogen.y - heavy.y) ** 2
                        + (hydrogen.z - heavy.z) ** 2
                    )
                    ** 0.5,
                    position,
                )
                for position, heavy in enumerate(heavy_atoms)
            ),
        )
        if distance > _HYDROGEN_BOND_LENGTH_LIMIT_ANGSTROM:
            differences.append(
                f"{hydrogen.name} is bonded to none of its atoms "
                f"({distance:.2f} A from {heavy_atoms[index].name})"
            )
            continue
        observed[index] += 1
    differences.extend(
        f"{heavy.name} has {have} in the input and {want} in the SMILES"
        for heavy, have, want in zip(heavy_atoms, observed, expected)
        if have != want
    )
    return differences


def _match_smiles_onto_residue(
    residue: ResidueRecord,
    heavy_atoms: Sequence[AtomRecord],
    smiles: str,
    *,
    ligand_id: str,
    net_charge: int | None,
) -> object:
    """The residue's heavy atoms, with the bond orders and charges of ``smiles``.

    Atom order is the residue's, so index i of the result is heavy_atoms[i].
    """

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - exercised by the skip guard
        raise LigandHydrogenationError(
            f"Ligand {ligand_id} has a SMILES, and placing or checking its "
            "hydrogens against it requires RDKit. Install RDKit, or add the "
            "hydrogens yourself and remove ligands[].smiles."
        ) from exc

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
    return molecule


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

    original = [atom for atom in before.atoms if not is_hydrogen(atom)]
    produced = [atom for atom in after.atoms if not is_hydrogen(atom)]
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
