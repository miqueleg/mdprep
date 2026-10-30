"""A residue of a chain, cut out between the caps it is parameterised in.

A residue inside a chain is not a molecule: its nitrogen and its carbonyl
carbon are bonded to its neighbours. To give it hydrogens from a SMILES, and
charges, it is taken out with an acetyl on its nitrogen and a methylamide on
its carbonyl -- the capped residue amino-acid parameterisations have always
used. The caps are built from atoms the structure already has: the previous
residue's CA, C and O are the acetyl's methyl, carbonyl carbon and oxygen, and
the next residue's N and CA the methylamide's. Every bond to the residue is
then at the length and angle the structure gives it.

A SMILES is the residue's protonation, given either capped like this or as
the free amino acid; mdprep caps the free amino acid itself. Only bond orders
and formal charges are taken from it. Its stereochemistry is the structure's,
as a ligand's is.

RDKit is imported lazily: this module is reachable from code the web builder
imports, and Pyodide has no RDKit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import dist
from typing import Sequence

from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, ResidueRecord

# A peptide C-N bond is 1.33 A; the PDB writer starts a new chain past 2 A.
PEPTIDE_BOND_MAX_ANGSTROM = 2.0

BACKBONE_ATOMS = ("N", "CA", "C", "O")

# Names of the cap atoms in the capped model. They are not the residue's and
# never reach the structure, so they are named apart from anything a residue
# can hold.
ACE_ATOMS = (("CAY", "C", "CA"), ("CY", "C", "C"), ("OY", "O", "O"))
NME_ATOMS = (("NT", "N", "N"), ("CAT", "C", "CA"))
CAP_HYDROGEN_NAMES = {"CAY": ("HY1", "HY2", "HY3"), "NT": ("HNT",), "CAT": ("HT1", "HT2", "HT3")}


# What GAFF2 types and AM1-BCC charges: a residue parameterised from its
# SMILES can hold nothing else.
PARAMETERISABLE_ELEMENTS = frozenset({"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})

# Beyond those, the elements that are not metals.
_NONMETALS = frozenset({"B", "Si", "Ge", "As", "Se", "Sb", "Te", "At", "D"})


class ResidueCappingError(ValueError):
    """Raised when a residue cannot be capped or matched to its SMILES."""


def unparameterisable_reason(atoms: Sequence[AtomRecord]) -> str | None:
    """Why a residue holding these atoms cannot be parameterised from a SMILES.

    None when it can. A metal gets its own explanation: mdprep models one as
    a metal site, a one-atom residue bonded or not to what surrounds it, and
    a residue deposited with a metal inside it is read with the metal moved
    out (structure/pdb_fields.py) unless that would break the residue apart.
    """

    foreign = [
        (atom.name.strip(), _element(atom))
        for atom in atoms
        if _element(atom) not in PARAMETERISABLE_ELEMENTS
    ]
    if not foreign:
        return None
    listed = ", ".join(f"{element} ({name})" for name, element in foreign)
    reason = (
        f"it holds {listed}, which neither GAFF2 nor AM1-BCC describes, so mdprep "
        "cannot parameterise it from a SMILES."
    )
    if any(element not in _NONMETALS for _, element in foreign):
        reason += (
            " mdprep models a metal as a metal site of its own, not as an atom of a "
            "residue. This one stayed in the residue because taking it out would have "
            "broken the residue apart -- the metal is bonded to atoms beyond the amino "
            "acid, an organometallic group -- or because it is not a metal mdprep "
            "builds sites for; see docs/noncanonical_residues.md."
        )
    return reason


@dataclass(frozen=True)
class ChainContext:
    """A residue of a chain and the two residues it is bonded to."""

    residue: ResidueRecord
    previous: ResidueRecord
    following: ResidueRecord


@dataclass(frozen=True)
class CappedAtom:
    role: str  # "ACE", "RES" or "NME"
    name: str
    element: str
    x: float
    y: float
    z: float


@dataclass
class CappedResidue:
    """The capped residue with every hydrogen, as RDKit and as plain atoms."""

    molecule: object
    atoms: list[CappedAtom]
    # Zero-based atom indices and Tripos bond orders ("1", "2", "3", "ar").
    bonds: list[tuple[int, int, str]]
    formal_charge: int
    smiles_form: str  # "capped" or "free amino acid"
    warnings: list[str] = field(default_factory=list)

    def residue_atoms(self) -> list[CappedAtom]:
        return [atom for atom in self.atoms if atom.role == "RES"]


def chain_context(residues: Sequence[ResidueRecord], residue: ResidueRecord) -> ChainContext:
    """The residues bonded to ``residue`` on either side, or why there are none.

    Found by the peptide bonds themselves, a C-N distance under 2 A in the
    same chain, not by numbering: an insertion code or a renumbered stretch
    does not move a bond.
    """

    label = residue.id.display()
    present = {atom.name.strip() for atom in residue.atoms if not is_hydrogen(atom)}
    missing = [name for name in BACKBONE_ATOMS if name not in present]
    if missing:
        raise ResidueCappingError(
            f"{label} has no {', '.join(missing)}: a residue kept in a chain needs "
            "the whole backbone, N, CA, C and O."
        )
    nitrogen = _named(residue, "N")
    carbon = _named(residue, "C")
    previous = following = None
    for other in residues:
        if other is residue or other.id.chain_id != residue.id.chain_id:
            continue
        other_carbon = _named(other, "C")
        other_nitrogen = _named(other, "N")
        if other_carbon is not None and _distance(other_carbon, nitrogen) <= PEPTIDE_BOND_MAX_ANGSTROM:
            previous = other
        if other_nitrogen is not None and _distance(carbon, other_nitrogen) <= PEPTIDE_BOND_MAX_ANGSTROM:
            following = other
    if previous is None or following is None:
        end = "N-terminus" if previous is None else "C-terminus"
        raise ResidueCappingError(
            f"{label} is at the {end} of its chain, or next to a gap in it: no residue "
            f"is bonded to its {'N' if previous is None else 'C'}. mdprep builds a kept "
            "noncanonical residue only inside a chain, bonded on both sides; a "
            "terminal one needs terminal templates mdprep does not make."
        )
    for neighbour, needed in ((previous, ("CA", "C", "O")), (following, ("N", "CA"))):
        absent = [name for name in needed if _named(neighbour, name) is None]
        if absent:
            raise ResidueCappingError(
                f"{label}'s neighbour {neighbour.id.display()} has no "
                f"{', '.join(absent)}, which the caps {label} is parameterised "
                "between are built from."
            )
    return ChainContext(residue=residue, previous=previous, following=following)


def capped_template(smiles: str, residue_heavy_atoms: int, *, label: str) -> tuple[object, str]:
    """The capped molecule ``smiles`` describes, and which form it was given in.

    A SMILES with five heavy atoms more than the residue is the residue capped
    -- an acetyl's three and a methylamide's two. One with a single heavy atom
    more is the free amino acid, whose hydroxyl oxygen is the bond to the next
    residue; mdprep caps it, keeping the side chain's protonation and leaving
    the termini's out of it. Anything else describes another molecule.
    """

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ResidueCappingError(f"{label}: {smiles!r} is not a valid SMILES.")
    heavy = molecule.GetNumAtoms()
    if heavy == residue_heavy_atoms + len(ACE_ATOMS) + len(NME_ATOMS):
        return molecule, "capped"
    if heavy == residue_heavy_atoms + 1:
        return _cap_free_amino_acid(molecule, smiles, label=label), "free amino acid"
    raise ResidueCappingError(
        f"{label}: the SMILES {smiles!r} has {heavy} heavy atoms, and the residue "
        f"has {residue_heavy_atoms}. Give the residue between an acetyl and a "
        f"methylamide ({residue_heavy_atoms + 5} heavy atoms, CC(=O)N...C(=O)NC) "
        f"or as the free amino acid ({residue_heavy_atoms + 1}), with every heavy "
        "atom the structure has."
    )


# The backbone of a free alpha-amino acid: an amine (not an amide), its
# alpha carbon, and a carboxylic acid or carboxylate.
_FREE_BACKBONE_SMARTS = "[NX3,NX4+;!$(N[#6]=[#8])][CX4][CX3](=[OX1])[OX2H1,OX1-]"


def _cap_free_amino_acid(molecule: object, smiles: str, *, label: str) -> object:
    from rdkit import Chem

    pattern = Chem.MolFromSmarts(_FREE_BACKBONE_SMARTS)
    matches = molecule.GetSubstructMatches(pattern)  # type: ignore[attr-defined]
    if len(matches) != 1:
        raise ResidueCappingError(
            f"{label}: the SMILES {smiles!r} has the heavy atoms of a free amino acid "
            f"but {len(matches) or 'no'} alpha-amino-acid backbone"
            f"{'s' if len(matches) > 1 else ''} mdprep can tell apart. Give the residue "
            "capped instead, CC(=O)N...C(=O)NC."
        )
    nitrogen, _, carbon, _, hydroxyl = matches[0]
    editable = Chem.RWMol(molecule)
    amine = editable.GetAtomWithIdx(nitrogen)
    # The termini's charges are the free amino acid's, not the residue's.
    amine.SetFormalCharge(0)
    amine.SetNumExplicitHs(0)
    amine.SetNoImplicit(False)
    methyl = editable.AddAtom(Chem.Atom(6))
    carbonyl = editable.AddAtom(Chem.Atom(6))
    oxygen = editable.AddAtom(Chem.Atom(8))
    editable.AddBond(methyl, carbonyl, Chem.BondType.SINGLE)
    editable.AddBond(carbonyl, oxygen, Chem.BondType.DOUBLE)
    editable.AddBond(carbonyl, nitrogen, Chem.BondType.SINGLE)
    amide = editable.AddAtom(Chem.Atom(7))
    amide_methyl = editable.AddAtom(Chem.Atom(6))
    editable.AddBond(carbon, amide, Chem.BondType.SINGLE)
    editable.AddBond(amide, amide_methyl, Chem.BondType.SINGLE)
    editable.RemoveAtom(hydroxyl)
    capped = editable.GetMol()
    try:
        Chem.SanitizeMol(capped)
    except Exception as exc:  # pragma: no cover - RDKit reports the reason
        raise ResidueCappingError(
            f"{label}: capping the free amino acid {smiles!r} gave an invalid molecule ({exc})."
        ) from exc
    return capped


def capped_residue(
    context: ChainContext,
    smiles: str,
    *,
    net_charge: int,
    label: str | None = None,
) -> CappedResidue:
    """The capped residue with the bond orders and hydrogens ``smiles`` states.

    Heavy atoms keep their names and coordinates; the hydrogens are placed by
    RDKit and named after the atom they are on. The SMILES's formal charge
    must be ``net_charge``, and the residue's backbone must be an amide
    nitrogen with one hydrogen and a carbonyl carbon with none: a backbone
    nitrogen that carries no hydrogen (a proline, an N-methyl amino acid)
    takes backbone charges no table in mdprep holds.
    """

    residue = context.residue
    label = label or residue.id.display()
    heavy = [atom for atom in residue.atoms if not is_hydrogen(atom)]
    foreign = unparameterisable_reason(heavy)
    if foreign is not None:
        raise ResidueCappingError(f"{label}: {foreign}")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # pragma: no cover - exercised by the skip guard
        raise ResidueCappingError(
            "Parameterising a noncanonical residue from its SMILES requires RDKit."
        ) from exc
    from mdprep.ligands.hydrogenate import geometry_contradicts_bond_orders

    template, form = capped_template(smiles, len(heavy), label=label)
    formal = Chem.GetFormalCharge(template)
    if formal != net_charge:
        raise ResidueCappingError(
            f"{label}: the SMILES {smiles!r} has a formal charge of {formal:+d} "
            f"{'once capped ' if form != 'capped' else ''}but net_charge is "
            f"{net_charge:+d}. The SMILES is the residue's protonation, so one of "
            "the two describes another molecule."
        )

    atoms: list[CappedAtom] = []
    for name, element, source in ACE_ATOMS:
        atoms.append(_cap_atom("ACE", name, element, _named(context.previous, source)))
    for atom in heavy:
        atoms.append(
            CappedAtom(
                role="RES",
                name=atom.name.strip(),
                element=_element(atom),
                x=atom.x,
                y=atom.y,
                z=atom.z,
            )
        )
    for name, element, source in NME_ATOMS:
        atoms.append(_cap_atom("NME", name, element, _named(context.following, source)))

    model = Chem.MolFromPDBBlock(_pdb_block(atoms), removeHs=False, sanitize=False)
    if model is None:
        raise ResidueCappingError(f"{label}: RDKit could not read the capped residue.")
    try:
        model = AllChem.AssignBondOrdersFromTemplate(template, model)
    except Exception as exc:
        raise ResidueCappingError(
            f"{label}: the SMILES {smiles!r} could not be matched onto the residue "
            f"between its caps ({exc}). Check that it has the residue's atoms, "
            "connected as they are in the structure."
        ) from exc
    # The bonds to the caps are amides by construction, not something the
    # SMILES claims; a deposited peptide bond can be short (1H4X has one at
    # 1.23 A) without saying anything about the residue's chemistry.
    cap_names = {name for name, _, _ in ACE_ATOMS + NME_ATOMS}
    contradictions = [
        complaint
        for complaint in geometry_contradicts_bond_orders(model, residue)
        if not cap_names & set(complaint.split(" is ", 1)[0].split("-"))
    ]
    if contradictions:
        raise ResidueCappingError(
            f"{label}: the bond orders the SMILES {smiles!r} implies do not match the "
            "structure, most likely the wrong tautomer or protonation site: "
            + "; ".join(contradictions)
            + "."
        )
    molecule = Chem.AddHs(model, addCoords=True)
    Chem.SanitizeMol(molecule)

    names = [atom.name for atom in atoms]
    roles = {index: atom.role for index, atom in enumerate(atoms)}
    hydrogens: dict[int, list[int]] = {}
    for atom in molecule.GetAtoms():
        if atom.GetIdx() >= len(atoms):
            hydrogens.setdefault(atom.GetNeighbors()[0].GetIdx(), []).append(atom.GetIdx())
    taken = {atom.name for atom in atoms if atom.role == "RES"}
    labels: dict[int, str] = {}
    for parent in sorted(hydrogens):
        indices = hydrogens[parent]
        if roles[parent] == "RES":
            chosen = residue_hydrogen_names(names[parent], len(indices), taken)
        else:
            chosen = list(CAP_HYDROGEN_NAMES.get(names[parent], ()))
            if len(chosen) != len(indices):
                raise ResidueCappingError(
                    f"{label}: the SMILES {smiles!r} gives the cap atom {names[parent]} "
                    f"{len(indices)} hydrogen(s); an acetyl and a methylamide have "
                    "neutral, saturated caps."
                )
        for index, name in zip(indices, chosen):
            labels[index] = name
            roles[index] = roles[parent]

    backbone = {name: index for index, name in enumerate(names) if roles[index] == "RES"}
    on_nitrogen = len(hydrogens.get(backbone["N"], []))
    on_carbon = len(hydrogens.get(backbone["C"], []))
    if on_nitrogen != 1 or on_carbon != 0:
        raise ResidueCappingError(
            f"{label}: with this SMILES its backbone N carries {on_nitrogen} "
            f"hydrogen(s) and its C {on_carbon}. mdprep parameterises a residue whose "
            "backbone is an amide N-H and a carbonyl, as the backbone charges it "
            "fixes are those of one; a proline or an N-methyl amino acid is not "
            "supported."
        )

    conformer = molecule.GetConformer()
    result_atoms = list(atoms)
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if index < len(atoms):
            continue
        position = conformer.GetAtomPosition(index)
        result_atoms.append(
            CappedAtom(
                role=roles[index],
                name=labels[index],
                element="H",
                x=float(position.x),
                y=float(position.y),
                z=float(position.z),
            )
        )
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), _tripos_order(bond))
        for bond in molecule.GetBonds()
    ]
    return CappedResidue(
        molecule=molecule,
        atoms=result_atoms,
        bonds=bonds,
        formal_charge=formal,
        smiles_form=form,
    )


def residue_hydrogen_names(parent: str, count: int, taken: set[str]) -> list[str]:
    """PDB-style names for the hydrogens on one heavy atom of a residue.

    H on the backbone N, HA on CA; then by the parent's remoteness letter, as
    the PDB names the twenty: one hydrogen is HB, two are HB2 and HB3, three
    are HZ1 to HZ3, and CG2's three are HG21 to HG23. Names stay within four
    characters and unique in the residue, falling back to the ligand scheme
    when they would not.
    """

    from mdprep.ligands.hydrogenate import hydrogen_names_for

    stem = parent[1:] if len(parent) > 1 and parent[0].isalpha() else parent
    if parent == "N":
        candidates = ["H"] if count == 1 else [f"H{index}" for index in range(1, count + 1)]
    elif count == 1:
        candidates = [f"H{stem}"]
    elif count == 2:
        candidates = [f"H{stem}2", f"H{stem}3"]
    else:
        candidates = [f"H{stem}{index}" for index in range(1, count + 1)]
    if all(len(name) <= 4 and name not in taken for name in candidates) and len(
        set(candidates)
    ) == len(candidates):
        taken.update(candidates)
        return candidates
    template = AtomRecord(
        serial=None, name=parent, altloc=None, resname="UNL", chain_id="", resid=1,
        icode=None, x=0.0, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0, element=None,
        record_name="ATOM", original_line="",
    )
    return hydrogen_names_for(template, count, taken)


def residue_hydrogens(capped: CappedResidue, residue: ResidueRecord) -> list[AtomRecord]:
    """The capped model's residue hydrogens, as atoms of ``residue``."""

    return [
        AtomRecord(
            serial=None,
            name=atom.name,
            altloc=None,
            resname=residue.id.resname,
            chain_id=residue.id.chain_id,
            resid=residue.id.resid,
            icode=residue.id.icode,
            x=atom.x,
            y=atom.y,
            z=atom.z,
            occupancy=1.0,
            bfactor=0.0,
            element="H",
            record_name="ATOM",
            original_line="",
        )
        for atom in capped.atoms
        if atom.role == "RES" and atom.element == "H"
    ]


def _cap_atom(role: str, name: str, element: str, source: AtomRecord | None) -> CappedAtom:
    assert source is not None  # chain_context has checked it is there
    return CappedAtom(role=role, name=name, element=element, x=source.x, y=source.y, z=source.z)


def _pdb_block(atoms: Sequence[CappedAtom]) -> str:
    # One residue of a name RDKit has no template for, so its bonds come from
    # the coordinates alone, as a ligand's do.
    lines = []
    for serial, atom in enumerate(atoms, start=1):
        name = atom.name if len(atom.name) == 4 else f" {atom.name:<3s}"
        lines.append(
            f"HETATM{serial:5d} {name} UNL A   1    "
            f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}  1.00  0.00          "
            f"{atom.element.upper():>2s}\n"
        )
    return "".join(lines) + "END\n"


def _tripos_order(bond: object) -> str:
    if bond.GetIsAromatic():  # type: ignore[attr-defined]
        return "ar"
    order = bond.GetBondTypeAsDouble()  # type: ignore[attr-defined]
    return {1.0: "1", 2.0: "2", 3.0: "3"}.get(order, "1")


def _named(residue: ResidueRecord, name: str) -> AtomRecord | None:
    return next(
        (atom for atom in residue.atoms if atom.name.strip() == name and not is_hydrogen(atom)),
        None,
    )


def _element(atom: AtomRecord) -> str:
    element = (atom.element or "").strip()
    if element:
        return element.capitalize()
    name = atom.name.strip()
    return name[:1].upper()


def _distance(first: AtomRecord | None, second: AtomRecord | None) -> float:
    if first is None or second is None:
        return float("inf")
    return dist((first.x, first.y, first.z), (second.x, second.y, second.z))
