"""1-4 interactions through a metal that have nothing to stop them collapsing.

Amber computes a scaled nonbonded interaction between the first and last
atoms of every dihedral. Through a bonded metal site that pairs atoms of
different ligands: in SyrB2 (PDB 2FCT) the chloride and the water both bind
the iron, so Cl-Fe-O-H makes the chloride and the water's hydrogen a 1-4
pair. MCPB.py sets the torsions through the metal to zero, the O...Cl
repulsion is excluded because the two are 1-3 through the iron, and a TIP3P
hydrogen has no Lennard-Jones radius at all. What is left is a Coulomb
attraction between -0.95 and +0.45 with no repulsive wall under it: the
first minimisation drove that hydrogen 0.14 A from the chloride and pulled
the water onto the iron, and the heating step went to NaN.

A pair like that has no lower bound, and giving the hydrogen a small radius
only moves where it collapses to -- GAFF2's hydroxyl radius still let it
reach about 1.4 A. So the 1-4 term is switched off for exactly those pairs:
through a metal, between different residues, where one atom has no
repulsion. The coordination geometry is still held by the angles MCPB.py
fitted around the metal, which is what describes it; every other interaction
is left as tleap wrote it.

The prmtop is edited with ParmEd's raw format reader, which changes the flags
and nothing else. Loading it as a ParmEd Structure would reorder the atoms of
a bonded site -- the MCPB bonds make the protein a non-contiguous molecule --
and leave the topology out of step with the PDB written beside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdprep.structure.classify import COORDINATING_METAL_ELEMENTS

_ATOMIC_NUMBER = {
    "MG": 12, "CA": 20, "V": 23, "MN": 25, "FE": 26, "CO": 27, "NI": 28,
    "CU": 29, "ZN": 30, "MO": 42, "CD": 48, "W": 74, "HG": 80,
}
METAL_ATOMIC_NUMBERS = frozenset(
    _ATOMIC_NUMBER[element] for element in COORDINATING_METAL_ELEMENTS
)


class ThroughMetalError(RuntimeError):
    """Raised when a topology cannot be inspected for through-metal pairs."""


@dataclass(frozen=True)
class SwitchedOffPair:
    first: str
    metal: str
    last: str

    def to_dict(self) -> dict[str, str]:
        return {"first": self.first, "metal": self.metal, "last": self.last}


def switch_off_unshielded_through_metal_14(prmtop: str | Path) -> list[SwitchedOffPair]:
    """Ignore the 1-4 ends of through-metal dihedrals with an unshielded atom.

    Edits ``prmtop`` in place and returns the pairs it switched off, one per
    pair. Amber already lists every 1-4 pair among the excluded atoms and adds
    it back through the dihedral list; flagging the dihedral's end group as
    ignored -- a negative third index -- removes it from that second step,
    which is the convention every Amber engine and OpenMM read.
    """

    try:
        from parmed.amber import AmberFormat
    except ImportError as exc:  # pragma: no cover - parmed is always installed
        raise ThroughMetalError("ParmEd is required to edit the topology") from exc

    path = Path(prmtop)
    parm = AmberFormat(str(path))
    data = parm.parm_data
    if "ATOMIC_NUMBER" not in data:
        raise ThroughMetalError(
            f"{path.name} has no ATOMIC_NUMBER section, so its metals cannot be found"
        )
    natom = data["POINTERS"][0]
    ntypes = data["POINTERS"][1]
    atomic_number = data["ATOMIC_NUMBER"]
    metals = {i for i in range(natom) if atomic_number[i] in METAL_ATOMIC_NUMBERS}
    if not metals:
        return []

    def unshielded(atom: int) -> bool:
        lj_type = data["ATOM_TYPE_INDEX"][atom]
        index = data["NONBONDED_PARM_INDEX"][ntypes * (lj_type - 1) + lj_type - 1]
        # A negative index points at the 10-12 table, which no Amber force
        # field in use writes; treat it as shielded rather than guess.
        return index > 0 and data["LENNARD_JONES_ACOEF"][index - 1] == 0.0

    pointers = list(data["RESIDUE_POINTER"]) + [natom + 1]
    residue_of = [0] * natom
    for residue in range(len(pointers) - 1):
        for atom in range(pointers[residue] - 1, pointers[residue + 1] - 1):
            residue_of[atom] = residue

    def label(atom: int) -> str:
        residue = residue_of[atom]
        # A space between name and number: MCPB.py's residue names end in
        # digits (WT1, HD1), and "WT13" would not say which residue it was.
        return f"{data['RESIDUE_LABEL'][residue].strip()} {residue + 1}:{data['ATOM_NAME'][atom].strip()}"

    switched: dict[frozenset[int], SwitchedOffPair] = {}
    for flag in ("DIHEDRALS_INC_HYDROGEN", "DIHEDRALS_WITHOUT_HYDROGEN"):
        entries = data[flag]
        for start in range(0, len(entries), 5):
            first = entries[start] // 3
            second = entries[start + 1] // 3
            third_raw = entries[start + 2]
            third = abs(third_raw) // 3
            last = abs(entries[start + 3]) // 3
            if third_raw <= 0:
                # Already ignored (negative) -- or atom 1, which cannot carry
                # the flag and is never a metal's neighbour in practice.
                continue
            metal = second if second in metals else third if third in metals else None
            if metal is None:
                continue
            if residue_of[first] == residue_of[last]:
                continue
            if not (unshielded(first) or unshielded(last)):
                continue
            entries[start + 2] = -third_raw
            pair = frozenset((first, last))
            switched.setdefault(
                pair, SwitchedOffPair(label(first), label(metal), label(last))
            )
    if switched:
        temporary = path.with_suffix(path.suffix + ".tmp")
        parm.write_parm(str(temporary))
        temporary.replace(path)
    return sorted(switched.values(), key=lambda pair: (pair.first, pair.last))
