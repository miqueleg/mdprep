"""PDB writing helpers."""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.models import AtomRecord, PdbStructure


def write_pdb(structure: PdbStructure, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    previous_chain: str | None = None
    previous_was_atom = False

    for atom in structure.atoms:
        if previous_chain is not None and atom.chain_id != previous_chain and previous_was_atom:
            lines.append("TER\n")
        lines.append(format_atom_record(atom))
        previous_chain = atom.chain_id
        previous_was_atom = atom.record_name == "ATOM"

    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="utf-8")


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
        f"{atom.record_name:<6}{serial:5d} {atom_name}{altloc}{atom.resname:>3} "
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

