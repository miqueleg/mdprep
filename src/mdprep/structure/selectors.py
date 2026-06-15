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
    chain_id: str
    resid: int
    resname: str | None = None
    icode: str | None = None

    def display(self) -> str:
        chain = self.chain_id
        residue = f"{self.resname or ''}{self.resid}{self.icode or ''}"
        return f"{chain}:{residue}"


@dataclass(frozen=True)
class AtomSelector:
    residue: ResidueSelector
    atom_name: str

    def display(self) -> str:
        return f"{self.residue.display()}@{self.atom_name}"


_RESIDUE_BODY_RE = re.compile(r"^(?:(?P<resname>[A-Za-z]+))?(?P<resid>-?\d+)(?P<icode>[A-Za-z]?)$")


def parse_residue_selector(text: str) -> ResidueSelector:
    if ":" not in text:
        raise SelectorError(f"Residue selector {text!r} must contain ':'")
    if "@" in text:
        raise SelectorError(f"Residue selector {text!r} must not contain '@'")
    chain_id, body = text.split(":", 1)
    if not body:
        raise SelectorError(f"Residue selector {text!r} is missing a residue number")
    match = _RESIDUE_BODY_RE.match(body.strip())
    if match is None:
        raise SelectorError(f"Could not parse residue selector {text!r}")
    return ResidueSelector(
        chain_id=chain_id,
        resname=match.group("resname"),
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
        if residue.id.chain_id == resolved_selector.chain_id
        and residue.id.resid == resolved_selector.resid
        and residue.id.icode == resolved_selector.icode
        and (resolved_selector.resname is None or residue.id.resname == resolved_selector.resname)
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
        chain_value = selector.get("chain_id", selector.get("chain", ""))
        if chain_value is None:
            chain_value = ""
        if "resid" not in selector:
            raise SelectorError(f"Structured selector is missing 'resid': {selector}")
        return ResidueSelector(
            chain_id=str(chain_value),
            resname=selector.get("resname"),
            resid=int(selector["resid"]),
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
    candidates = [
        residue
        for residue in structure.residues
        if residue.id.chain_id == selector.chain_id
        and abs(residue.id.resid - selector.resid) <= 2
    ]
    if not candidates:
        candidates = [residue for residue in structure.residues if residue.id.resid == selector.resid]
    return candidates[:5]


def _format_residue(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    icode = residue.id.icode or ""
    return f"chain={chain} resname={residue.id.resname} resid={residue.id.resid}{icode}"
