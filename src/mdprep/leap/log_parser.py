"""Parse tleap logs into validation-friendly summaries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


class LeapLogError(ValueError):
    """Raised when a tleap log indicates an unsafe build."""


@dataclass(frozen=True)
class LeapLogSummary:
    returncode: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unknown_residues: list[str] = field(default_factory=list)
    missing_atom_types: list[str] = field(default_factory=list)
    missing_parameters: list[str] = field(default_factory=list)
    atoms_created: list[str] = field(default_factory=list)
    total_charge: float | None = None
    check_output: list[str] = field(default_factory=list)
    log_tail: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "returncode": self.returncode,
            "warnings": self.warnings,
            "errors": self.errors,
            "unknown_residues": self.unknown_residues,
            "missing_atom_types": self.missing_atom_types,
            "missing_parameters": self.missing_parameters,
            "atoms_created": self.atoms_created,
            "total_charge": self.total_charge,
            "check_output": self.check_output,
            "log_tail": self.log_tail,
        }


_CHARGE_PATTERNS = [
    re.compile(r"Total unperturbed charge:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"unperturbed charge of the unit:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE),
]


def parse_tleap_log(path: str | Path, *, returncode: int = 0) -> LeapLogSummary:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return parse_tleap_log_text(text, returncode=returncode)


def parse_tleap_log_text(text: str, *, returncode: int = 0) -> LeapLogSummary:
    warnings: list[str] = []
    errors: list[str] = []
    unknown_residues: list[str] = []
    missing_atom_types: list[str] = []
    missing_parameters: list[str] = []
    atoms_created: list[str] = []
    check_output: list[str] = []
    total_charge: float | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if "warning:" in lower:
            warnings.append(line)
        if "fatal error" in lower or lower.startswith("error:") or " error!" in lower:
            errors.append(line)
        if "unknown residue" in lower:
            unknown_residues.append(line)
            errors.append(line)
        if "does not have a type" in lower or "could not find type" in lower:
            missing_atom_types.append(line)
            errors.append(line)
        if (
            "could not find bond parameter" in lower
            or "could not find angle parameter" in lower
            or "could not find torsion parameter" in lower
            or "missing parameters" in lower
        ):
            missing_parameters.append(line)
            errors.append(line)
        if "created a new atom named" in lower:
            atoms_created.append(line)
        if lower.startswith("checking") or "unit is ok" in lower:
            check_output.append(line)
        for pattern in _CHARGE_PATTERNS:
            match = pattern.search(line)
            if match:
                total_charge = float(match.group(1))

    return LeapLogSummary(
        returncode=returncode,
        warnings=warnings,
        errors=_dedupe(errors),
        unknown_residues=_dedupe(unknown_residues),
        missing_atom_types=_dedupe(missing_atom_types),
        missing_parameters=_dedupe(missing_parameters),
        atoms_created=atoms_created,
        total_charge=total_charge,
        check_output=check_output,
        log_tail=_tail_lines(text),
    )


def assert_tleap_success(
    summary: LeapLogSummary,
    *,
    fail_on_warnings: bool,
    context: str,
) -> None:
    if summary.returncode != 0:
        raise LeapLogError(
            "\n".join(
                [
                    f"{context} tleap exited with code {summary.returncode}.",
                    *_summary_details(summary),
                ]
            )
        )
    if summary.unknown_residues:
        raise LeapLogError(f"{context} tleap reported unknown residues: {summary.unknown_residues}")
    if summary.missing_atom_types:
        raise LeapLogError(f"{context} tleap reported missing atom types: {summary.missing_atom_types}")
    if summary.missing_parameters:
        raise LeapLogError(f"{context} tleap reported missing parameters: {summary.missing_parameters}")
    if summary.errors:
        raise LeapLogError(f"{context} tleap reported errors: {summary.errors}")
    if fail_on_warnings and summary.warnings:
        raise LeapLogError(f"{context} tleap warnings are fatal by validation config: {summary.warnings}")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _summary_details(summary: LeapLogSummary) -> list[str]:
    details: list[str] = []
    if summary.errors:
        details.append(f"errors: {summary.errors}")
    if summary.unknown_residues:
        details.append(f"unknown residues: {summary.unknown_residues}")
    if summary.missing_atom_types:
        details.append(f"missing atom types: {summary.missing_atom_types}")
    if summary.missing_parameters:
        details.append(f"missing parameters: {summary.missing_parameters}")
    if summary.warnings:
        details.append(f"warnings: {summary.warnings}")
    if summary.log_tail:
        details.append("tleap log tail:")
        details.extend(summary.log_tail)
    return details


def _tail_lines(text: str, *, lines: int = 30) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return stripped.splitlines()[-lines:]
