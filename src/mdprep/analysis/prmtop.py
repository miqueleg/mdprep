"""Reading an Amber topology without ParmEd.

A prmtop is plain text: named sections (``%FLAG CHARGE``), each with a
Fortran format (``%FORMAT(5E16.8)``) and fixed-width fields. The few sections
needed to show charges and draw a trajectory are read here directly.

Molecules are worked out from the bonds rather than taken from
``ATOMS_PER_MOLECULE``: that section assumes a molecule's atoms are
contiguous, and a bonded metal site joins residues that are not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# tleap stores a charge q as q * 18.2223, which makes the Coulomb constant 1
# in its units.
AMBER_CHARGE_UNIT = 18.2223

_ELEMENTS = {
    1: "H", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 12: "Mg",
    14: "Si", 15: "P", 16: "S", 17: "Cl", 19: "K", 20: "Ca", 23: "V", 25: "Mn",
    26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 34: "Se", 35: "Br",
    37: "Rb", 42: "Mo", 48: "Cd", 53: "I", 55: "Cs", 74: "W", 80: "Hg",
}
METAL_ATOMIC_NUMBERS = frozenset({12, 20, 23, 25, 26, 27, 28, 29, 30, 42, 48, 74, 80})

_FORMAT = re.compile(r"\((\d+)([aAiIeEfF])(\d+)")


class PrmtopError(ValueError):
    """Raised when a file is not a readable Amber topology."""


@dataclass(frozen=True)
class AmberAtom:
    index: int
    name: str
    type: str
    charge: float
    mass: float
    atomic_number: int
    residue_index: int

    @property
    def element(self) -> str:
        return _ELEMENTS.get(self.atomic_number, "X")


@dataclass(frozen=True)
class AmberResidue:
    index: int
    name: str
    start: int
    stop: int  # one past the last atom

    @property
    def atom_indices(self) -> range:
        return range(self.start, self.stop)


@dataclass
class AmberTopology:
    atoms: list[AmberAtom]
    residues: list[AmberResidue]
    bonds: list[tuple[int, int]]
    periodic: bool

    def residue_of(self, atom: int) -> AmberResidue:
        return self.residues[self.atoms[atom].residue_index]

    def is_metal(self, atom: int) -> bool:
        return self.atoms[atom].atomic_number in METAL_ATOMIC_NUMBERS

    def molecules(self) -> list[list[int]]:
        """Connected components of the bond graph, in atom order."""

        parent = list(range(len(self.atoms)))

        def root(atom: int) -> int:
            while parent[atom] != atom:
                parent[atom] = parent[parent[atom]]
                atom = parent[atom]
            return atom

        for first, second in self.bonds:
            a, b = root(first), root(second)
            if a != b:
                parent[max(a, b)] = min(a, b)
        groups: dict[int, list[int]] = {}
        for atom in range(len(self.atoms)):
            groups.setdefault(root(atom), []).append(atom)
        return sorted(groups.values(), key=lambda group: group[0])

    def metal_bonds(self) -> list[tuple[int, int]]:
        """Bonds to a metal, metal first: a bonded (MCPB) site's coordination."""

        bonds = []
        for first, second in self.bonds:
            if self.is_metal(first) and not self.is_metal(second):
                bonds.append((first, second))
            elif self.is_metal(second) and not self.is_metal(first):
                bonds.append((second, first))
        return sorted(bonds)

    def label(self, atom: int) -> str:
        # A space between name and number: MCPB.py's residue names end in a
        # digit, and HD1110 would not say which residue it was.
        residue = self.residue_of(atom)
        return f"{residue.name} {residue.index + 1}:{self.atoms[atom].name}"


def _sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("%FLAG"):
            index += 1
            continue
        flag = line[5:].strip()
        index += 1
        while index < len(lines) and lines[index].startswith("%COMMENT"):
            index += 1
        if index >= len(lines) or not lines[index].startswith("%FORMAT"):
            raise PrmtopError(f"Section {flag} has no %FORMAT line")
        match = _FORMAT.search(lines[index])
        if match is None:
            raise PrmtopError(f"Cannot read the format of section {flag}: {lines[index]}")
        width = int(match.group(3))
        kind = match.group(2).lower()
        index += 1
        values: list[str] = []
        while index < len(lines) and not lines[index].startswith("%"):
            row = lines[index].rstrip("\n")
            values.extend(
                row[start : start + width] for start in range(0, len(row), width)
            )
            index += 1
        if kind != "a":
            values = [value for value in values if value.strip()]
        sections[flag] = values if kind == "a" else [value.strip() for value in values]
    return sections


def read_prmtop(path: str | Path) -> AmberTopology:
    """Atoms, residues and bonds of an Amber ``.prmtop``."""

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    sections = _sections(text)
    required = ("POINTERS", "ATOM_NAME", "CHARGE", "MASS", "RESIDUE_LABEL", "RESIDUE_POINTER")
    missing = [flag for flag in required if flag not in sections]
    if missing:
        raise PrmtopError(f"{Path(path).name} is not an Amber topology: no {', '.join(missing)}")

    pointers = [int(value) for value in sections["POINTERS"]]
    natom, nres = pointers[0], pointers[11]
    periodic = len(pointers) > 27 and pointers[27] > 0
    names = [name.strip() for name in sections["ATOM_NAME"][:natom]]
    types = [name.strip() for name in sections.get("AMBER_ATOM_TYPE", [""] * natom)[:natom]]
    charges = [float(value) / AMBER_CHARGE_UNIT for value in sections["CHARGE"][:natom]]
    masses = [float(value) for value in sections["MASS"][:natom]]
    if "ATOMIC_NUMBER" in sections:
        numbers = [int(value) for value in sections["ATOMIC_NUMBER"][:natom]]
    else:
        numbers = [_element_from_mass(mass) for mass in masses]
    labels = [label.strip() for label in sections["RESIDUE_LABEL"][:nres]]
    starts = [int(value) - 1 for value in sections["RESIDUE_POINTER"][:nres]] + [natom]
    if len(names) != natom or len(charges) != natom or len(labels) != nres:
        raise PrmtopError(f"{Path(path).name} is truncated: sections are shorter than POINTERS says")

    residues = [
        AmberResidue(index=index, name=labels[index], start=starts[index], stop=starts[index + 1])
        for index in range(nres)
    ]
    residue_of = [0] * natom
    for residue in residues:
        for atom in residue.atom_indices:
            residue_of[atom] = residue.index
    atoms = [
        AmberAtom(
            index=index,
            name=names[index],
            type=types[index] if index < len(types) else "",
            charge=charges[index],
            mass=masses[index],
            atomic_number=numbers[index],
            residue_index=residue_of[index],
        )
        for index in range(natom)
    ]
    bonds: list[tuple[int, int]] = []
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        entries = [int(value) for value in sections.get(flag, [])]
        for start in range(0, len(entries) - 2, 3):
            bonds.append((entries[start] // 3, entries[start + 1] // 3))
    return AmberTopology(atoms=atoms, residues=residues, bonds=bonds, periodic=periodic)


def _element_from_mass(mass: float) -> int:
    """Atomic number from a mass, for topologies older than ATOMIC_NUMBER."""

    table = {
        1.008: 1, 12.01: 6, 14.01: 7, 16.0: 8, 19.0: 9, 22.99: 11, 24.305: 12,
        30.97: 15, 32.06: 16, 35.45: 17, 39.1: 19, 40.08: 20, 54.94: 25,
        55.85: 26, 58.93: 27, 58.69: 28, 63.55: 29, 65.4: 30, 79.9: 35, 126.9: 53,
    }
    if mass <= 0.0:
        return 0
    nearest = min(table, key=lambda value: abs(value - mass))
    return table[nearest] if abs(nearest - mass) < 0.6 else 0


def read_inpcrd(path: str | Path) -> tuple[list[tuple[float, float, float]], tuple[float, ...] | None]:
    """Coordinates (angstrom) and box (a, b, c, alpha, beta, gamma) of an inpcrd.

    The format is a title, the atom count, then fixed 12-column numbers six to
    a line: coordinates, velocities if the file is a restart, and the box last.
    """

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise PrmtopError(f"{Path(path).name} is not an Amber coordinate file")
    natom = int(lines[1].split()[0])
    numbers: list[float] = []
    for line in lines[2:]:
        numbers.extend(float(line[start : start + 12]) for start in range(0, len(line.rstrip()), 12))
    if len(numbers) < 3 * natom:
        raise PrmtopError(f"{Path(path).name} has fewer coordinates than its {natom} atoms")
    coordinates = [tuple(numbers[3 * i : 3 * i + 3]) for i in range(natom)]
    rest = numbers[3 * natom :]
    box = tuple(rest[-6:]) if len(rest) in (6, 3 * natom + 6) else None
    return coordinates, box  # type: ignore[return-value]
