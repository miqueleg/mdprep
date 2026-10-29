"""Reading a unit out of a tleap OFF library.

Two things need a library's contents without running tleap: checking that
the library mdprep wrote gives every atom its element, and the charges of an
AmberTools template for the MM embedding of the histidine tautomer
calculations, which run before tleap does.
"""

from __future__ import annotations

from dataclasses import dataclass


class OffLibraryError(ValueError):
    """Raised when a library does not hold the unit asked for."""


@dataclass(frozen=True)
class OffAtom:
    name: str
    type: str
    atomic_number: int
    charge: float


@dataclass(frozen=True)
class OffUnit:
    name: str
    atoms: tuple[OffAtom, ...]
    # Zero-based atom indices.
    bonds: tuple[tuple[int, int], ...]


def read_off_unit(text: str, unit: str) -> OffUnit:
    lines = text.splitlines()
    atoms = [
        OffAtom(
            name=fields[0].strip('"'),
            type=fields[1].strip('"'),
            atomic_number=int(fields[6]),
            charge=float(fields[7]),
        )
        for fields in (line.split() for line in _table(lines, unit, "atoms"))
    ]
    bonds = tuple(
        (int(fields[0]) - 1, int(fields[1]) - 1)
        for fields in (line.split() for line in _table(lines, unit, "connectivity"))
    )
    return OffUnit(name=unit, atoms=tuple(atoms), bonds=bonds)


def _table(lines: list[str], unit: str, table: str) -> list[str]:
    marker = f"!entry.{unit}.unit.{table} table"
    for index, line in enumerate(lines):
        if line.startswith(marker):
            rows: list[str] = []
            for row in lines[index + 1 :]:
                if row.startswith("!"):
                    break
                rows.append(row)
            return rows
    raise OffLibraryError(f"The library has no {table} table for unit {unit}.")
