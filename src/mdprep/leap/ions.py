"""Ion and salt command helpers for tleap scripts."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sqrt


class IonPlanError(ValueError):
    """Raised when ion placement cannot be planned safely."""


@dataclass(frozen=True)
class IonPlan:
    neutralizing_ion: str | None
    neutralizing_count: int
    salt_pairs: int
    commands: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "neutralizing_ion": self.neutralizing_ion,
            "neutralizing_count": self.neutralizing_count,
            "salt_pairs": self.salt_pairs,
            "commands": self.commands,
        }


def neutralizing_ion_count(
    total_charge: float,
    *,
    positive_ion: str,
    negative_ion: str,
    tolerance: float = 0.05,
) -> tuple[str | None, int]:
    rounded = round(total_charge)
    if abs(total_charge - rounded) > tolerance:
        raise IonPlanError(
            f"System charge {total_charge:.6f} is not within {tolerance:.2f} e of an integer."
        )
    if rounded < 0:
        return positive_ion, abs(rounded)
    if rounded > 0:
        return negative_ion, rounded
    return None, 0


def salt_pair_count(molarity: float, volume_a3: float) -> int:
    if molarity < 0:
        raise IonPlanError("Salt concentration cannot be negative.")
    if volume_a3 <= 0:
        raise IonPlanError("Periodic box volume must be positive to add salt pairs.")
    return round(molarity * volume_a3 * 0.000602214076)


def build_ion_plan(
    *,
    total_charge: float,
    neutralize: bool,
    positive_ion: str,
    negative_ion: str,
    salt_concentration_molar: float,
    volume_a3: float | None,
) -> IonPlan:
    commands: list[str] = []
    neutralizing_ion: str | None = None
    neutralizing_count = 0
    if neutralize:
        neutralizing_ion, neutralizing_count = neutralizing_ion_count(
            total_charge,
            positive_ion=positive_ion,
            negative_ion=negative_ion,
        )
        if neutralizing_ion is not None and neutralizing_count > 0:
            commands.append(f"addionsrand system {neutralizing_ion} {neutralizing_count}")

    salt_pairs = 0
    if salt_concentration_molar > 0:
        if volume_a3 is None:
            raise IonPlanError("Cannot add salt pairs because solvated box volume was not determined.")
        salt_pairs = salt_pair_count(salt_concentration_molar, volume_a3)
        if salt_pairs > 0:
            commands.append(f"addionsrand system {positive_ion} {salt_pairs}")
            commands.append(f"addionsrand system {negative_ion} {salt_pairs}")
    return IonPlan(
        neutralizing_ion=neutralizing_ion,
        neutralizing_count=neutralizing_count,
        salt_pairs=salt_pairs,
        commands=commands,
    )


def amber_inpcrd_box_volume(path: str) -> float | None:
    """Return periodic box volume from an Amber restart/inpcrd file, if present."""

    lines = [line.strip() for line in open(path, encoding="utf-8").read().splitlines() if line.strip()]
    if len(lines) < 3:
        return None
    fields = lines[-1].split()
    if len(fields) < 6:
        return None
    try:
        a, b, c, alpha, beta, gamma = (float(value) for value in fields[:6])
    except ValueError:
        return None
    cos_a = cos(radians(alpha))
    cos_b = cos(radians(beta))
    cos_g = cos(radians(gamma))
    factor = 1.0 + 2.0 * cos_a * cos_b * cos_g - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g
    if factor <= 0:
        return None
    return a * b * c * sqrt(factor)
