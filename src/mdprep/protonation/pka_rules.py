"""pH/pKa residue-state assignment rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.structure.models import ResidueRecord


class PkaRuleError(ValueError):
    """Raised when pKa rules require unsupported automatic chemistry."""


AutoSource = Literal["propka", "propka_xtb_his", "input_state"]


AMBER_INPUT_STATES = {"ASH", "GLH", "HID", "HIE", "HIP", "LYN", "CYM", "CYX"}


@dataclass(frozen=True)
class PkaDecision:
    residue: ResidueRecord
    final_state: str | None
    source: AutoSource
    reason: str
    pka: float | None = None
    needs_xtb: bool = False
    warnings: list[str] = field(default_factory=list)


def decide_residue_state(
    residue: ResidueRecord,
    *,
    record: PropkaRecord | None,
    ph: float,
    method: Literal["propka", "propka_xtb_his"],
) -> PkaDecision | None:
    resname = residue.id.resname
    if resname in AMBER_INPUT_STATES:
        return PkaDecision(
            residue=residue,
            final_state=resname,
            source="input_state",
            reason="Amber-specific input residue state preserved",
            pka=record.pka if record else None,
            warnings=_close_to_ph_warning(residue, ph, record),
        )

    if resname not in {"ASP", "GLU", "CYS", "LYS", "ARG", "HIS"}:
        return None

    warnings = _close_to_ph_warning(residue, ph, record)
    if record is None:
        warnings.append(f"Missing pKa for {_format_residue(residue)}; using conservative default.")
        if resname == "HIS":
            return _neutral_histidine_without_pka(residue, ph, method, warnings)
        return PkaDecision(
            residue=residue,
            final_state=_conservative_default(resname),
            source="propka",
            reason="Missing PropKa pKa; conservative default used",
            pka=None,
            warnings=warnings,
        )

    protonated = ph < record.pka
    if resname == "ASP":
        return _decision(residue, "ASH" if protonated else "ASP", record, ph, warnings)
    if resname == "GLU":
        return _decision(residue, "GLH" if protonated else "GLU", record, ph, warnings)
    if resname == "CYS":
        return _decision(residue, "CYS" if protonated else "CYM", record, ph, warnings)
    if resname == "LYS":
        return _decision(residue, "LYS" if protonated else "LYN", record, ph, warnings)
    if resname == "ARG":
        if not protonated:
            warnings.append(
                "ARG deprotonation is not automatically represented in mdprep v0.1; "
                "use a manual override or custom parameters if needed."
            )
        return _decision(residue, "ARG", record, ph, warnings)
    if resname == "HIS":
        if protonated:
            return _decision(residue, "HIP", record, ph, warnings)
        if method == "propka":
            raise PkaRuleError(
                f"Neutral HIS {_format_residue(residue)} requires HID/HIE assignment; "
                "use method: propka_xtb_his or add a manual HIS override to HID/HIE/HIP."
            )
        return PkaDecision(
            residue=residue,
            final_state=None,
            source="propka_xtb_his",
            reason="Neutral HIS requires HID/HIE xTB comparison",
            pka=record.pka,
            needs_xtb=True,
            warnings=warnings,
        )
    return None


def _neutral_histidine_without_pka(
    residue: ResidueRecord,
    ph: float,
    method: Literal["propka", "propka_xtb_his"],
    warnings: list[str],
) -> PkaDecision:
    if method == "propka":
        raise PkaRuleError(
            f"Neutral HIS {_format_residue(residue)} has no pKa record and requires HID/HIE assignment; "
            "use method: propka_xtb_his or add a manual HIS override to HID/HIE/HIP."
        )
    return PkaDecision(
        residue=residue,
        final_state=None,
        source="propka_xtb_his",
        reason="Missing HIS pKa; conservative neutral HIS sent to xTB comparison",
        pka=None,
        needs_xtb=True,
        warnings=warnings,
    )


def _decision(
    residue: ResidueRecord,
    state: str,
    record: PropkaRecord,
    ph: float,
    warnings: list[str],
) -> PkaDecision:
    return PkaDecision(
        residue=residue,
        final_state=state,
        source="propka",
        reason=f"pH {ph:g} {'<' if ph < record.pka else '>='} pKa {record.pka:g}",
        pka=record.pka,
        warnings=warnings,
    )


def _close_to_ph_warning(
    residue: ResidueRecord,
    ph: float,
    record: PropkaRecord | None,
) -> list[str]:
    if record is not None and abs(ph - record.pka) <= 0.5:
        return [f"pKa for {_format_residue(residue)} is within 0.5 pH units of target pH."]
    return []


def _conservative_default(resname: str) -> str:
    return {
        "ASP": "ASP",
        "GLU": "GLU",
        "CYS": "CYS",
        "LYS": "LYS",
        "ARG": "ARG",
    }[resname]


def _format_residue(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    return f"{chain}:{residue.id.resname}{residue.id.resid}{residue.id.icode or ''}"

