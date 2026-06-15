"""Internal structure data models."""

from __future__ import annotations

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

