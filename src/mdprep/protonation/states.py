"""Residue protonation-state family validation."""

from __future__ import annotations


class ProtonationStateError(ValueError):
    """Raised for incompatible residue protonation state assignments."""


STATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "ASP": ("ASP", "ASH"),
    "ASH": ("ASP", "ASH"),
    "GLU": ("GLU", "GLH"),
    "GLH": ("GLU", "GLH"),
    "LYS": ("LYS", "LYN"),
    "LYN": ("LYS", "LYN"),
    "ARG": ("ARG",),
    "HIS": ("HIS", "HID", "HIE", "HIP"),
    "HID": ("HIS", "HID", "HIE", "HIP"),
    "HIE": ("HIS", "HID", "HIE", "HIP"),
    "HIP": ("HIS", "HID", "HIE", "HIP"),
    "CYS": ("CYS", "CYM", "CYX"),
    "CYM": ("CYS", "CYM", "CYX"),
    "CYX": ("CYS", "CYM", "CYX"),
}

KNOWN_STATES = frozenset(STATE_FAMILIES)


def allowed_states_for_resname(resname: str) -> tuple[str, ...]:
    return STATE_FAMILIES.get(resname, ())


def validate_state_transition(
    *,
    current_resname: str,
    requested_state: str,
    selector: str,
) -> None:
    if requested_state not in KNOWN_STATES:
        raise ProtonationStateError(
            f"Unknown protonation state for selector {selector}: requested {requested_state}."
        )
    allowed = allowed_states_for_resname(current_resname)
    if not allowed or requested_state not in allowed:
        allowed_text = ", ".join(allowed) if allowed else "none"
        raise ProtonationStateError(
            f"Incompatible protonation override for selector {selector}: "
            f"current residue {current_resname}, requested {requested_state}, "
            f"allowed states: {allowed_text}."
        )

