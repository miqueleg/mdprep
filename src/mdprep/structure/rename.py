"""Changing one residue's name in a structure, in the input, not behind it.

mdprep refuses a ligand whose residue name an Amber force field already
defines, and tells the user to rename it. For a deposition fetched by ID that
is otherwise a text-editor job on a megabyte of PDB, so the GUI offers to make
the change -- and the change is made here, to the structure itself, so that
what the user sees, what the bundle carries and what mdprep reads are one
file and not three descriptions of one.

Renaming is by residue identity and never by name. 6NIE's substrate is a free
lysine called LYS 301 and the protein it sits in has twenty-odd lysines of its
own, also called LYS. Renaming every residue that shares the name would
dissolve the protein.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping

from mdprep.structure.models import AtomRecord, PdbStructure, build_residues


class ResidueRenameError(ValueError):
    """Raised when a requested rename does not match the structure."""


def _key(chain_id: str, resname: str, resid: int, icode: str | None):
    return (
        (chain_id or "").strip(),
        (resname or "").strip().upper(),
        int(resid),
        (icode or "").strip() or None,
    )


def rename_residues(
    structure: PdbStructure, renames: Iterable[Mapping[str, object]]
) -> PdbStructure:
    """Return ``structure`` with the named residues renamed.

    Each entry gives ``chain``, ``resname``, ``resid`` and optionally
    ``icode`` -- enough to pick out one residue -- plus the ``new_resname``
    it should carry.
    """

    wanted: dict[tuple, str] = {}
    for entry in renames:
        key = _key(
            str(entry.get("chain", "")),
            str(entry.get("resname", "")),
            int(entry["resid"]),  # type: ignore[arg-type]
            entry.get("icode"),  # type: ignore[arg-type]
        )
        new = str(entry["new_resname"]).strip().upper()
        if not new or len(new) > 4:
            raise ResidueRenameError(
                f"{new!r} is not a usable residue name; PDB allows up to four "
                "characters"
            )
        wanted[key] = new
    if not wanted:
        return structure

    present = {
        _key(atom.chain_id, atom.resname, atom.resid, atom.icode)
        for atom in structure.atoms
    }
    missing = sorted(set(wanted) - present)
    if missing:
        listed = ", ".join(
            f"{chain}:{resname}{resid}" for chain, resname, resid, _ in missing
        )
        raise ResidueRenameError(f"Not in this structure: {listed}")

    # A new name that something else already uses would merge two different
    # residues under one template, which is the failure being fixed.
    existing_names = {
        atom.resname.strip().upper()
        for atom in structure.atoms
        if _key(atom.chain_id, atom.resname, atom.resid, atom.icode) not in wanted
    }
    clashing = sorted(set(wanted.values()) & existing_names)
    if clashing:
        raise ResidueRenameError(
            f"{', '.join(clashing)} is already used by another residue here; "
            "choose a name nothing else uses"
        )

    def renamed(atom: AtomRecord) -> AtomRecord:
        new = wanted.get(_key(atom.chain_id, atom.resname, atom.resid, atom.icode))
        return atom if new is None else replace(atom, resname=new)

    atoms = [renamed(atom) for atom in structure.atoms]
    # Residues are rebuilt from the atoms rather than renamed in parallel:
    # two statements of one fact is how a resname and its residue record
    # drift apart.
    return PdbStructure(
        path=structure.path,
        atoms=atoms,
        residues=build_residues(atoms),
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
    )
