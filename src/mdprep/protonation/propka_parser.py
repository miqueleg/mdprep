"""PropKa output parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from mdprep.structure.models import PdbStructure, ResidueRecord


class PropkaParseError(ValueError):
    """Raised when parsed PropKa records cannot be mapped safely."""


_PKA_LINE = re.compile(
    r"^\s*(?P<resname>ASP|GLU|HIS|LYS|CYS|ARG)\s+"
    r"(?P<resid>-?\d+)\s+"
    r"(?P<chain>\S)\s+"
    r"(?P<pka>-?\d+(?:\.\d+)?)\b"
)


@dataclass(frozen=True)
class PropkaRecord:
    resname: str
    resid: int
    chain_id: str
    pka: float
    raw_line: str

    def to_dict(self) -> dict[str, object]:
        return {
            "resname": self.resname,
            "resid": self.resid,
            "chain_id": self.chain_id,
            "pka": self.pka,
            "raw_line": self.raw_line,
        }


def parse_propka_file(path: str | Path) -> list[PropkaRecord]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    summary_records = _parse_summary_records(lines)
    if summary_records:
        return summary_records
    records: list[PropkaRecord] = []
    for line in lines:
        record = parse_propka_line(line)
        if record is not None:
            records.append(record)
    return records


def parse_propka_line(line: str) -> PropkaRecord | None:
    match = _PKA_LINE.match(line)
    if match is None:
        return None
    chain_id = match.group("chain")
    if chain_id in {"-", "_"}:
        chain_id = ""
    return PropkaRecord(
        resname=match.group("resname"),
        resid=int(match.group("resid")),
        chain_id=chain_id,
        pka=float(match.group("pka")),
        raw_line=line,
    )


def _parse_summary_records(lines: list[str]) -> list[PropkaRecord]:
    in_summary = False
    records: list[PropkaRecord] = []
    for line in lines:
        if "SUMMARY OF THIS PREDICTION" in line:
            in_summary = True
            continue
        if in_summary and line.startswith("----") and records:
            break
        if not in_summary:
            continue
        record = parse_propka_line(line)
        if record is not None:
            records.append(record)
    return records


def map_propka_records(
    structure: PdbStructure,
    records: list[PropkaRecord],
) -> dict[int, PropkaRecord]:
    mapped: dict[int, PropkaRecord] = {}
    for record in records:
        matches = [
            residue
            for residue in structure.residues
            if residue.id.chain_id == record.chain_id
            and residue.id.resid == record.resid
            and _propka_family_name(residue.id.resname) == record.resname
        ]
        if len(matches) > 1:
            formatted = ", ".join(_format_residue(residue) for residue in matches)
            raise PropkaParseError(
                f"PropKa record maps ambiguously to mdprep residues: {record.raw_line!r}. "
                f"Matches: {formatted}. Use a manual override for residues with insertion codes."
            )
        if len(matches) == 0:
            continue
        residue = matches[0]
        key = id(residue)
        if key in mapped:
            raise PropkaParseError(
                f"Multiple PropKa records map to residue {_format_residue(residue)}."
            )
        mapped[key] = record
    return mapped


def _propka_family_name(resname: str) -> str | None:
    return {
        "ASP": "ASP",
        "ASH": "ASP",
        "GLU": "GLU",
        "GLH": "GLU",
        "HIS": "HIS",
        "HID": "HIS",
        "HIE": "HIS",
        "HIP": "HIS",
        "LYS": "LYS",
        "LYN": "LYS",
        "CYS": "CYS",
        "CYM": "CYS",
        "CYX": "CYS",
        "ARG": "ARG",
    }.get(resname)


def _format_residue(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    return f"{chain}:{residue.id.resname}{residue.id.resid}{residue.id.icode or ''}"
