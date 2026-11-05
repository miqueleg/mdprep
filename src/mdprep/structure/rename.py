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
from typing import Iterable, Mapping, Sequence

from mdprep.structure.classify import LEAP_RESERVED_RESIDUE_NAMES
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
from mdprep.structure.pdb_fields import ion_element, looks_like_ion_resname


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


def apply_residue_renames(
    structure: PdbStructure, renames: Sequence[object]
) -> tuple[PdbStructure, list[dict[str, object]]]:
    """Apply a manifest's ``structure.rename_residues`` to the input as read.

    Unlike :func:`rename_residues`, a new name may be one another residue
    already has: that is the point of building two copies of one molecule
    under one name. It may only be one that a residue with the same heavy
    atoms has, named alike and in the same order, since those two then
    share one template; anything else would build a molecule from another
    one's parameters, and is refused. A metal keeps a name that says which
    element it is, and nothing else takes a name a force field defines or
    that reads as a metal.

    Returns the renamed structure and one record per rename, for the
    structure report.
    """

    wanted: dict[tuple, tuple[str, str]] = {}
    for entry in renames:
        selector = entry.selector  # type: ignore[attr-defined]
        key = _key(selector.chain, selector.resname, selector.resid, selector.icode)
        wanted[key] = (str(entry.resname).strip(), str(entry.reason))  # type: ignore[attr-defined]
    if not wanted:
        return structure, []

    residues = {
        _key(*residue.id.to_dict().values()): residue for residue in structure.residues
    }
    missing = sorted(set(wanted) - set(residues), key=str)
    if missing:
        listed = ", ".join(
            f"{chain}:{resname}{resid}{icode or ''}" for chain, resname, resid, icode in missing
        )
        raise ResidueRenameError(
            f"structure.rename_residues names residues that are not in the input: {listed}"
        )

    for key, (new, _) in wanted.items():
        residue = residues[key]
        if len(residue.atoms) == 1:
            element = (residue.atoms[0].element or "").strip().upper()
            if ion_element(residue.atoms[0].resname) and ion_element(new) != element:
                raise ResidueRenameError(
                    f"{residue.id.display()} is a lone {element} atom; renamed {new} "
                    "it would no longer be read as one. Give it a name that starts "
                    "with its element."
                )
            continue
        if new.upper() in LEAP_RESERVED_RESIDUE_NAMES or looks_like_ion_resname(new):
            raise ResidueRenameError(
                f"{residue.id.display()} cannot be built as {new}: the force fields "
                "already define that name, or it reads as a metal ion."
            )

    def heavy(residue) -> list[tuple[str, str]]:
        return [
            (atom.name, (atom.element or "").strip().upper())
            for atom in residue.atoms
            if not is_hydrogen(atom)
        ]

    final: dict[tuple, str] = {
        key: wanted[key][0] if key in wanted else residue.id.resname
        for key, residue in residues.items()
    }
    for new in {name for name, _ in wanted.values()}:
        sharing = [key for key, name in final.items() if name == new]
        first = residues[sharing[0]]
        for key in sharing[1:]:
            other = residues[key]
            if heavy(other) != heavy(first):
                raise ResidueRenameError(
                    f"structure.rename_residues would build {other.id.display()} and "
                    f"{first.id.display()} under one name, {new}, but they are not the "
                    "same atoms named alike, and tleap builds every residue of one "
                    "name from one template."
                )

    def renamed(atom: AtomRecord) -> AtomRecord:
        change = wanted.get(_key(atom.chain_id, atom.resname, atom.resid, atom.icode))
        # The line it was read from has the old name; writers fall back to
        # the fields when there is none.
        return atom if change is None else replace(atom, resname=change[0], original_line="")

    def moved(identity: tuple) -> tuple:
        change = wanted.get(_key(*identity[:4]))
        return identity if change is None else (identity[0], change[0], *identity[2:])

    atoms = [renamed(atom) for atom in structure.atoms]
    records = [
        {
            "chain": key[0],
            "resname": key[1],
            "resid": key[2],
            "icode": key[3],
            "display": residues[key].id.display(),
            "new_resname": new,
            "reason": reason,
        }
        for key, (new, reason) in wanted.items()
    ]
    return (
        replace(
            structure,
            atoms=atoms,
            residues=build_residues(atoms),
            warnings=list(structure.warnings),
            links=tuple(
                replace(link, first=moved(link.first), second=moved(link.second))
                for link in structure.links
            ),
            modified_residue_parents={
                moved(key): parent for key, parent in structure.modified_residue_parents.items()
            },
        ),
        records,
    )
