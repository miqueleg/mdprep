"""xTB output energy parsing and tautomer selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


HARTREE_TO_KCAL_MOL = 627.509474


class XtbParseError(ValueError):
    """Raised when xTB output does not contain a parseable energy."""


ENERGY_PATTERNS = [
    re.compile(r"TOTAL\s+ENERGY\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"total\s+energy\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"::\s*total\s+energy\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"total\s+E\s*=\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE),
]


@dataclass(frozen=True)
class XtbEnergyComparison:
    hid_energy_hartree: float
    hie_energy_hartree: float
    delta_kcal_mol: float
    selected_state: str
    close_call: bool
    warnings: list[str]


def parse_xtb_energy_text(text: str, *, source: str = "xTB stdout") -> float:
    energies: list[float] = []
    for line in text.splitlines():
        for pattern in ENERGY_PATTERNS:
            match = pattern.search(line)
            if match:
                energies.append(float(match.group(1)))
                break
    if not energies:
        raise XtbParseError(f"Could not parse xTB total energy from {source}")
    return energies[-1]


def parse_xtb_energy_file(path: str | Path) -> float:
    path_obj = Path(path)
    return parse_xtb_energy_text(path_obj.read_text(encoding="utf-8"), source=str(path_obj))


def compare_hid_hie_energies(
    *,
    hid_energy_hartree: float,
    hie_energy_hartree: float,
    close_call_kcal_mol: float,
) -> XtbEnergyComparison:
    delta = (hid_energy_hartree - hie_energy_hartree) * HARTREE_TO_KCAL_MOL
    warnings: list[str] = []
    if hid_energy_hartree < hie_energy_hartree:
        selected = "HID"
    elif hie_energy_hartree < hid_energy_hartree:
        selected = "HIE"
    else:
        selected = "HIE"
        warnings.append("HID and HIE xTB energies are equal; selected HIE by deterministic tie-break.")
    close_call = abs(delta) <= close_call_kcal_mol
    if close_call:
        warnings.append("HID/HIE xTB energy difference is within the configured close-call threshold.")
    return XtbEnergyComparison(
        hid_energy_hartree=hid_energy_hartree,
        hie_energy_hartree=hie_energy_hartree,
        delta_kcal_mol=delta,
        selected_state=selected,
        close_call=close_call,
        warnings=warnings,
    )

