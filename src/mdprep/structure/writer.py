"""PDB writing helpers."""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.models import AtomRecord, PdbStructure
from mdprep.structure.pdb import (
    HYBRID36_DIGITS_LOWER,
    HYBRID36_DIGITS_UPPER,
    HYBRID36_RESID_LOWER_OFFSET,
    HYBRID36_RESID_UPPER_OFFSET,
    RESID_FIELD_WIDTH,
    SERIAL_FIELD_WIDTH,
    HYBRID36_LOWER_OFFSET,
    HYBRID36_UPPER_OFFSET,
)


def write_pdb(structure: PdbStructure, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_pdb(structure), encoding="utf-8")


def render_pdb(structure: PdbStructure) -> str:
    """The PDB text for this structure, without writing it anywhere.

    The web builder draws the structure it is asking questions about. Giving
    the viewer the uploaded file instead means the picture is of the names
    the file came with, while the questions are about the names mdprep
    chose -- so a renamed ligand or metal highlights nothing, on the one
    question that asks the student to confirm what they can see.
    """

    lines: list[str] = []
    previous_atom: AtomRecord | None = None
    ter_written_after: set[int] = set()
    # An atom mdprep created -- a ligand hydrogen it placed -- carries no
    # serial, and writing them all as 0 made every one of them the same
    # atom. With CONECT records alongside, mdprep's own reader then refused
    # the file it had just written, which is how the intermediate structures
    # became unreadable. Existing serials are left alone so the CONECT
    # records keep pointing at what they pointed at.
    taken = {atom.serial for atom in structure.atoms if atom.serial is not None}
    assigned: dict[int, int] = {}
    next_serial = 1
    for atom in structure.atoms:
        if atom.serial is None:
            while next_serial in taken:
                next_serial += 1
            taken.add(next_serial)
            assigned[id(atom)] = next_serial

    for atom in structure.atoms:
        if previous_atom is not None and _needs_implicit_ter(previous_atom, atom):
            previous_serial = previous_atom.serial
            if previous_serial is None or previous_serial not in ter_written_after:
                lines.append("TER\n")
        lines.append(format_atom_record(atom, serial=assigned.get(id(atom))))
        if atom.serial is not None and atom.serial in structure.ter_after_serials:
            lines.append("TER\n")
            ter_written_after.add(atom.serial)
        previous_atom = atom

    lines.extend(_format_conect_records(structure))
    lines.append("END\n")
    return "".join(lines)


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


def _encode_base36(value: int, digits: str, *, width: int = SERIAL_FIELD_WIDTH) -> str:
    characters: list[str] = []
    for _ in range(width):
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


def format_resid_field(resid: int) -> str:
    """Render columns 23-26, continuing in hybrid-36 past 9,999.

    Four columns do not hold the residue count of a solvated box, and this is
    what tleap and ambpdb do when they run out: "A000" follows "9999".
    read_pdb decodes it on the way back in.
    """

    if resid <= 9999:
        return f"{resid:4d}"
    if resid < HYBRID36_RESID_LOWER_OFFSET:
        return _encode_base36(
            resid - HYBRID36_RESID_UPPER_OFFSET,
            HYBRID36_DIGITS_UPPER,
            width=RESID_FIELD_WIDTH,
        )
    if resid < HYBRID36_RESID_LOWER_OFFSET + 26 * 36 ** (RESID_FIELD_WIDTH - 1):
        return _encode_base36(
            resid - HYBRID36_RESID_LOWER_OFFSET,
            HYBRID36_DIGITS_LOWER,
            width=RESID_FIELD_WIDTH,
        )
    raise ValueError(f"Residue number {resid} cannot be represented in a PDB file")


def format_atom_record(atom: AtomRecord, *, serial: int | None = None) -> str:
    _check_fields_fit(atom)
    if serial is None:
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
        f"{chain_id}{format_resid_field(atom.resid)}{icode}   "
        f"{atom.x:8.3f}{atom.y:8.3f}{atom.z:8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element}\n"
    )


def _check_fields_fit(atom: AtomRecord) -> None:
    """Refuse to write a record whose fields would not line up.

    Every field below is a minimum width, not a maximum, so a value one
    character too wide shifts the whole rest of the line right rather than
    being truncated or rejected. The result still looks like a PDB file and is
    read back as nonsense -- a chain id inside the residue number, coordinates
    missing their first digit -- somewhere far from here. Readers are expected
    to have fitted their input to these widths already (see
    mdprep.structure.pdb_fields), so reaching this is a bug rather than bad
    user input, and it should say so where it happens.
    """

    from mdprep.structure.pdb_fields import (
        CHAIN_ID_WIDTH,
        COORDINATE_MAX,
        COORDINATE_MIN,
        RESID_MAX,
        RESID_MIN,
        RESNAME_WIDTH,
        PdbFieldError,
    )

    if len(atom.resname) > RESNAME_WIDTH:
        raise PdbFieldError(
            f"Residue name {atom.resname!r} needs {len(atom.resname)} columns but "
            f"a PDB record has {RESNAME_WIDTH} for it, so writing it would shift "
            "every later field out of place."
        )
    if len(atom.chain_id) > CHAIN_ID_WIDTH:
        raise PdbFieldError(
            f"Chain id {atom.chain_id!r} needs {len(atom.chain_id)} columns but a "
            f"PDB record has {CHAIN_ID_WIDTH} for it, so writing it would shift "
            "every later field out of place."
        )
    for axis in ("x", "y", "z"):
        value = getattr(atom, axis)
        if not COORDINATE_MIN <= value <= COORDINATE_MAX:
            raise PdbFieldError(
                f"Coordinate {axis}={value} does not fit the eight columns a "
                f"PDB record gives it ({COORDINATE_MIN} to {COORDINATE_MAX}); "
                "writing it would shift the occupancy and everything after it."
            )
    if atom.resid < RESID_MIN:
        raise PdbFieldError(
            f"Residue number {atom.resid} is below the smallest a PDB record "
            f"can hold ({RESID_MIN}), so writing it would shift every later "
            "field out of place."
        )


def _atom_name_field(atom: AtomRecord) -> str:
    if len(atom.original_line) >= 16:
        original = atom.original_line[12:16]
        if original.strip() == atom.name:
            return original
    if atom.element and len(atom.element.strip()) == 1 and len(atom.name) < 4:
        return f" {atom.name:<3}"[:4]
    return f"{atom.name:>4}"[-4:]
