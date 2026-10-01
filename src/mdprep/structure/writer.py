"""PDB writing helpers."""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.models import AtomRecord, PdbStructure
from mdprep.structure.pdb import (
    HYBRID36_DIGITS_LOWER,
    HYBRID36_DIGITS_UPPER,
    SERIAL_FIELD_WIDTH,
    HYBRID36_LOWER_OFFSET,
    HYBRID36_UPPER_OFFSET,
)


def write_pdb(structure: PdbStructure, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    previous_atom: AtomRecord | None = None
    ter_written_after: set[int] = set()

    for atom in structure.atoms:
        if previous_atom is not None and _needs_implicit_ter(previous_atom, atom):
            previous_serial = previous_atom.serial
            if previous_serial is None or previous_serial not in ter_written_after:
                lines.append("TER\n")
        lines.append(format_atom_record(atom))
        if atom.serial is not None and atom.serial in structure.ter_after_serials:
            lines.append("TER\n")
            ter_written_after.add(atom.serial)
        previous_atom = atom

    lines.extend(_format_conect_records(structure))
    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def _needs_implicit_ter(previous: AtomRecord, current: AtomRecord) -> bool:
    if previous.chain_id != current.chain_id:
        return True
    # A same-chain HETATM must not be interpreted as the next peptide residue.
    return previous.record_name == "ATOM" and current.record_name == "HETATM"


def _format_conect_records(structure: PdbStructure) -> list[str]:
    serials = {atom.serial for atom in structure.atoms if atom.serial is not None}
    neighbors: dict[int, set[int]] = {}
    for left, right in structure.conect_bonds:
        if left not in serials or right not in serials:
            continue
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    lines: list[str] = []
    for source in sorted(neighbors):
        targets = sorted(neighbors[source])
        for offset in range(0, len(targets), 4):
            group = targets[offset : offset + 4]
            lines.append(
                "CONECT" + f"{source:5d}" + "".join(f"{target:5d}" for target in group) + "\n"
            )
    return lines


def _encode_base36(value: int, digits: str) -> str:
    characters: list[str] = []
    for _ in range(SERIAL_FIELD_WIDTH):
        value, remainder = divmod(value, 36)
        characters.append(digits[remainder])
    return "".join(reversed(characters))


def format_serial_field(serial: int) -> str:
    """Render columns 7-12 of an ATOM record: the serial plus its separator.

    Systems above 99,999 atoms do not fit the five-column serial field, so the
    sixth digit is written into column 12 exactly as tleap and ambpdb do, which
    keeps the atom-name field at columns 13-16. Beyond 999,999 atoms the field
    switches to hybrid-36, which read_pdb decodes on the way back in.
    """

    if 0 <= serial <= 99999:
        return f"{serial:5d} "
    if 100000 <= serial <= 999999:
        return f"{serial:6d}"
    if 10 ** 6 <= serial < HYBRID36_LOWER_OFFSET:
        return _encode_base36(serial - HYBRID36_UPPER_OFFSET, HYBRID36_DIGITS_UPPER) + " "
    if HYBRID36_LOWER_OFFSET <= serial < HYBRID36_LOWER_OFFSET + 26 * 36 ** 4:
        return _encode_base36(serial - HYBRID36_LOWER_OFFSET, HYBRID36_DIGITS_LOWER) + " "
    raise ValueError(f"Atom serial {serial} cannot be represented in a PDB file")


def format_atom_record(atom: AtomRecord) -> str:
    serial = atom.serial if atom.serial is not None else 0
    atom_name = _atom_name_field(atom)
    altloc = atom.altloc or " "
    chain_id = atom.chain_id if atom.chain_id else " "
    icode = atom.icode or " "
    occupancy = atom.occupancy if atom.occupancy is not None else 1.0
    bfactor = atom.bfactor if atom.bfactor is not None else 0.0
    element = (atom.element or "").rjust(2)
    return (
        f"{atom.record_name:<6}{format_serial_field(serial)}{atom_name}{altloc}{atom.resname:>3} "
        f"{chain_id}{atom.resid:4d}{icode}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element}\n"
    )


def _atom_name_field(atom: AtomRecord) -> str:
    if len(atom.original_line) >= 16:
        original = atom.original_line[12:16]
        if original.strip() == atom.name:
            return original
    if atom.element and len(atom.element.strip()) == 1 and len(atom.name) < 4:
        return f" {atom.name:<3}"[:4]
    return f"{atom.name:>4}"[-4:]
