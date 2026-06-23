"""Residue and atom selector parsing and resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord


class SelectorError(ValueError):
    """Raised when a selector cannot be resolved exactly."""


@dataclass(frozen=True)
class ResidueSelector:
    chain_id: str | None
    resid: int | None
    resname: str | None = None
    icode: str | None = None

    def display(self) -> str:
        chain = "*" if self.chain_id is None else self.chain_id
        residue_number = "*" if self.resid is None else str(self.resid)
        residue = f"{self.resname or ''}{residue_number}{self.icode or ''}"
        return f"{chain}:{residue}"


@dataclass(frozen=True)
class AtomSelector:
    residue: ResidueSelector
    atom_name: str

    def display(self) -> str:
        return f"{self.residue.display()}@{self.atom_name}"


_RESID_ONLY_BODY_RE = re.compile(r"^(?P<resid>-?\d+)(?P<icode>[A-Za-z]?)$")
_RESNAME_RESID_BODY_RE = re.compile(r"^(?P<resname>[A-Za-z0-9]{1,3})(?P<resid>-?\d+)(?P<icode>[A-Za-z]?)$")


def parse_residue_selector(text: str) -> ResidueSelector:
    if ":" not in text:
        raise SelectorError(f"Residue selector {text!r} must contain ':'")
    if "@" in text:
        raise SelectorError(f"Residue selector {text!r} must not contain '@'")
    chain_id, body = text.split(":", 1)
    if not body:
        raise SelectorError(f"Residue selector {text!r} is missing a residue number")
    stripped_body = body.strip()
    match = _RESID_ONLY_BODY_RE.match(stripped_body)
    if match is None:
        match = _RESNAME_RESID_BODY_RE.match(stripped_body)
    if match is None:
        raise SelectorError(f"Could not parse residue selector {text!r}")
    resname = match.groupdict().get("resname")
    return ResidueSelector(
        chain_id=chain_id,
        resname=resname,
        resid=int(match.group("resid")),
        icode=match.group("icode") or None,
    )


def parse_atom_selector(text: str) -> AtomSelector:
    if "@" not in text:
        raise SelectorError(f"Atom selector {text!r} must contain '@'")
    residue_text, atom_name = text.split("@", 1)
    atom_name = atom_name.strip()
    if not atom_name:
        raise SelectorError(f"Atom selector {text!r} is missing an atom name")
    return AtomSelector(residue=parse_residue_selector(residue_text), atom_name=atom_name)


def resolve_all_residue_selector(
    structure: PdbStructure,
    selector: ResidueSelector | dict[str, Any] | str,
) -> list[ResidueRecord]:
    resolved_selector = coerce_residue_selector(selector)
    matches = [
        residue
        for residue in structure.residues
        if _matches_residue_selector(residue, resolved_selector)
    ]
    if not matches:
        nearby = _nearby_residues(structure, resolved_selector)
        suffix = f" Nearby residues: {', '.join(_format_residue(residue) for residue in nearby)}" if nearby else ""
        raise SelectorError(f"No residue matched selector {resolved_selector.display()}.{suffix}")
    return matches


def resolve_residue_selector(
    structure: PdbStructure,
    selector: ResidueSelector | dict[str, Any] | str,
) -> ResidueRecord:
    resolved_selector = coerce_residue_selector(selector)
    matches = resolve_all_residue_selector(structure, resolved_selector)
    if len(matches) > 1:
        formatted = ", ".join(_format_residue(residue) for residue in matches)
        raise SelectorError(
            f"Selector {resolved_selector.display()} matched {len(matches)} residues: {formatted}"
        )
    return matches[0]


def resolve_atom_selector(
    structure: PdbStructure,
    selector: AtomSelector | dict[str, Any] | str,
) -> AtomRecord:
    resolved_selector = coerce_atom_selector(selector)
    residue = resolve_residue_selector(structure, resolved_selector.residue)
    matches = [atom for atom in residue.atoms if atom.name == resolved_selector.atom_name]
    if not matches:
        available = ", ".join(residue.atom_names())
        raise SelectorError(
            f"No atom matched selector {resolved_selector.display()}. Available atoms: {available}"
        )
    if len(matches) > 1:
        raise SelectorError(f"Selector {resolved_selector.display()} matched {len(matches)} atoms")
    return matches[0]


def coerce_residue_selector(selector: ResidueSelector | dict[str, Any] | str) -> ResidueSelector:
    if isinstance(selector, ResidueSelector):
        return selector
    if isinstance(selector, str):
        return parse_residue_selector(selector)
    if isinstance(selector, dict):
        chain_value = selector.get("chain_id", selector.get("chain"))
        if chain_value is not None:
            chain_value = str(chain_value)
        has_resid = "resid" in selector and selector.get("resid") is not None
        resname = selector.get("resname")
        if not has_resid and not resname:
            raise SelectorError(f"Structured selector must include at least 'resname' or 'resid': {selector}")
        return ResidueSelector(
            chain_id=chain_value,
            resname=str(resname) if resname is not None else None,
            resid=int(selector["resid"]) if has_resid else None,
            icode=selector.get("icode"),
        )
    raise SelectorError(f"Unsupported residue selector type: {type(selector).__name__}")


def coerce_atom_selector(selector: AtomSelector | dict[str, Any] | str) -> AtomSelector:
    if isinstance(selector, AtomSelector):
        return selector
    if isinstance(selector, str):
        return parse_atom_selector(selector)
    if isinstance(selector, dict):
        atom_name = selector.get("atom_name", selector.get("atom"))
        if not atom_name:
            raise SelectorError(f"Structured atom selector is missing atom_name: {selector}")
        residue_data = selector.get("residue", selector)
        return AtomSelector(residue=coerce_residue_selector(residue_data), atom_name=str(atom_name).strip())
    raise SelectorError(f"Unsupported atom selector type: {type(selector).__name__}")


def _nearby_residues(structure: PdbStructure, selector: ResidueSelector) -> list[ResidueRecord]:
    candidates: list[ResidueRecord] = []
    if selector.resid is not None:
        candidates = [
            residue
            for residue in structure.residues
            if (selector.chain_id is None or residue.id.chain_id == selector.chain_id)
            and abs(residue.id.resid - selector.resid) <= 2
        ]
    if not candidates and selector.resname is not None:
        candidates = [
            residue
            for residue in structure.residues
            if (selector.chain_id is None or residue.id.chain_id == selector.chain_id)
            and residue.id.resname == selector.resname
        ]
    if not candidates:
        candidates = [
            residue
            for residue in structure.residues
            if selector.resid is not None and residue.id.resid == selector.resid
        ]
    if not candidates:
        candidates = structure.residues
    return candidates[:5]


def _matches_residue_selector(residue: ResidueRecord, selector: ResidueSelector) -> bool:
    if selector.chain_id is not None and residue.id.chain_id != selector.chain_id:
        return False
    if selector.resid is not None and residue.id.resid != selector.resid:
        return False
    if selector.resname is not None and residue.id.resname != selector.resname:
        return False
    if selector.icode is not None and residue.id.icode != selector.icode:
        return False
    if selector.resid is not None and selector.icode is None and residue.id.icode is not None:
        return False
    return True


def _format_residue(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    icode = residue.id.icode or ""
    return f"chain={chain} resname={residue.id.resname} resid={residue.id.resid}{icode}"
