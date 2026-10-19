"""Internal structure data models."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


RecordName = Literal["ATOM", "HETATM"]


@dataclass(frozen=True)
class AtomRecord:
    serial: int | None
    name: str
    altloc: str | None
    resname: str
    chain_id: str
    resid: int
    icode: str | None
    x: float
    y: float
    z: float
    occupancy: float | None
    bfactor: float | None
    element: str | None
    record_name: RecordName
    original_line: str

    @property
    def residue_key(self) -> tuple[str, str, int, str | None]:
        return (self.chain_id, self.resname, self.resid, self.icode)

    @property
    def atom_identity(self) -> tuple[str, str, int, str | None, str]:
        return (self.chain_id, self.resname, self.resid, self.icode, self.name)


@dataclass(frozen=True)
class ResidueId:
    chain_id: str
    resname: str
    resid: int
    icode: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "chain_id": self.chain_id,
            "resname": self.resname,
            "resid": self.resid,
            "icode": self.icode,
        }

    def display(self) -> str:
        chain = self.chain_id if self.chain_id else "<blank>"
        icode = self.icode or ""
        return f"{chain}:{self.resname}{self.resid}{icode}"


@dataclass
class ResidueRecord:
    id: ResidueId
    atoms: list[AtomRecord]
    record_names: set[str]
    original_index: int

    def atom_names(self) -> list[str]:
        return [atom.name for atom in self.atoms]


@dataclass
class PdbStructure:
    path: Path
    atoms: list[AtomRecord]
    residues: list[ResidueRecord]
    model_count: int
    used_model: int = 1
    warnings: list[str] = field(default_factory=list)
    # Connectivity is stored by PDB atom serial so it survives residue
    # renaming and filtering.  Each pair is canonicalized as (min, max).
    conect_bonds: set[tuple[int, int]] = field(default_factory=set)
    # TER is a polymer/chain boundary, not decoration.  Store the serial of
    # the last atom before each explicit input TER so writers cannot
    # accidentally reconnect an unresolved chain break or a heterogen.
    ter_after_serials: set[int] = field(default_factory=set)
    # Residue names mdprep assigned rather than read, because the input used
    # one that could not be written to a PDB column. The name is as good as
    # any for building with, but it carries none of the meaning a deposited
    # name would: nothing may be inferred from it that was not already known
    # from the atoms.
    assigned_resnames: frozenset[str] = frozenset()


def build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    """Group atoms into residues, in the order they first appear.

    Seven modules had a private copy of this, in three spellings and all
    doing the same thing: group by ``atom.residue_key`` preserving order,
    carry the record names, number them from zero. Every stage that rewrites
    a structure's atoms has to rebuild its residues afterwards, so this is
    the single statement of what a residue list is -- and ``original_index``
    in particular, which downstream code orders by, now cannot mean one
    thing in the reader and another in the metal site builder.
    """

    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = (
        OrderedDict()
    )
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    return [
        ResidueRecord(
            id=ResidueId(chain_id=chain_id, resname=resname, resid=resid, icode=icode),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, ((chain_id, resname, resid, icode), residue_atoms) in enumerate(
            grouped.items()
        )
    ]
