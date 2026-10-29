"""Charges for a residue parameterised between caps, on the protein's backbone.

The charges of a capped residue are fitted for the whole capped molecule, so
the caps carry a little of the residue's charge and the backbone atoms come
out unlike every other residue's. Amber's own protocol for a new residue
(Cornell et al. 1995, and the phosphorylated residues AmberTools ships) gives
its backbone the charges the force field gives the backbone of a residue of
the same sign: ff94 fitted one set for neutral residues, one for the anionic
ASP and GLU and one for the cationic LYS, ARG and HIP, and ff14SB and ff19SB
kept them. So the capped residue's backbone N, H, C and O take that set, the
caps are dropped, and whatever charge is then missing from the residue's
integer is spread evenly over its other atoms -- the alpha carbon and the side
chain, whose charges are the residue's own.

Pure Python: the arithmetic is the part worth testing without AmberTools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ff14SB and ff19SB, amino12.lib and amino19.lib: SER, ASP and LYS.
BACKBONE_CHARGES: dict[str, dict[str, float]] = {
    "neutral": {"N": -0.4157, "H": 0.2719, "C": 0.5973, "O": -0.5679},
    "anionic": {"N": -0.5163, "H": 0.2936, "C": 0.5366, "O": -0.5819},
    "cationic": {"N": -0.3479, "H": 0.2747, "C": 0.7341, "O": -0.5894},
}

# A shift this large means the caps and the backbone took a good part of the
# charge the residue's own atoms carried in the fit, which is worth a look.
LARGE_SHIFT_E = 0.05

_MICRO = 1_000_000


class ResidueChargeError(ValueError):
    """Raised when a residue's charges cannot be put on the backbone."""


def backbone_set(net_charge: int) -> str:
    if net_charge < 0:
        return "anionic"
    if net_charge > 0:
        return "cationic"
    return "neutral"


@dataclass(frozen=True)
class ResidueCharges:
    # Residue atom name to charge, in the capped molecule's order.
    charges: dict[str, float]
    net_charge: int
    backbone_set: str
    fitted_residue_sum: float
    cap_sums: dict[str, float]
    shift_per_atom: float
    shifted_atoms: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "charges": dict(self.charges),
            "net_charge": self.net_charge,
            "backbone_set": self.backbone_set,
            "backbone_charges": dict(BACKBONE_CHARGES[self.backbone_set]),
            "fitted_residue_sum": round(self.fitted_residue_sum, 6),
            "cap_sums": {key: round(value, 6) for key, value in self.cap_sums.items()},
            "shift_per_atom": round(self.shift_per_atom, 6),
            "shifted_atoms": list(self.shifted_atoms),
            "warnings": list(self.warnings),
        }


def constrain_to_backbone(
    fitted: Sequence[tuple[str, str, float]], net_charge: int
) -> ResidueCharges:
    """The residue's charges from the capped molecule's fitted ones.

    ``fitted`` is (role, atom name, charge) for every atom of the capped
    molecule, role ``ACE``, ``RES`` or ``NME``. The result sums to
    ``net_charge`` exactly at the six decimals a mol2 file holds; the last
    microelectron of rounding goes to CA, so the arithmetic is the same on
    every machine.
    """

    residue = [(name, charge) for role, name, charge in fitted if role == "RES"]
    names = [name for name, _ in residue]
    if len(set(names)) != len(names):
        raise ResidueChargeError(f"the residue's atom names are not unique: {names}")
    which = backbone_set(net_charge)
    backbone = BACKBONE_CHARGES[which]
    absent = [name for name in backbone if name not in names]
    if absent:
        raise ResidueChargeError(
            f"the residue has no {', '.join(absent)}, whose charges are fixed to the backbone's"
        )
    if "CA" not in names:
        raise ResidueChargeError("the residue has no CA")
    free = [name for name in names if name not in backbone]
    fitted_sum = sum(charge for _, charge in residue)
    fixed_sum = sum(backbone.values())
    free_sum = sum(charge for name, charge in residue if name not in backbone)
    shift = (net_charge - fixed_sum - free_sum) / len(free)

    micro = {
        name: round((backbone[name] if name in backbone else charge + shift) * _MICRO)
        for name, charge in residue
    }
    micro["CA"] += net_charge * _MICRO - sum(micro.values())
    charges = {name: micro[name] / _MICRO for name in names}

    cap_sums = {
        cap: sum(charge for role, _, charge in fitted if role == cap) for cap in ("ACE", "NME")
    }
    warnings: list[str] = []
    if abs(shift) > LARGE_SHIFT_E:
        warnings.append(
            f"Fixing the backbone to the force field's {which} charges and dropping the "
            f"caps left {net_charge - fixed_sum - free_sum:+.3f} e to spread over the "
            f"residue's {len(free)} other atoms, {shift:+.3f} e each. The caps held "
            f"{cap_sums['ACE']:+.3f} and {cap_sums['NME']:+.3f} e in the fit."
        )
    return ResidueCharges(
        charges=charges,
        net_charge=net_charge,
        backbone_set=which,
        fitted_residue_sum=fitted_sum,
        cap_sums=cap_sums,
        shift_per_atom=shift,
        shifted_atoms=tuple(free),
        warnings=tuple(warnings),
    )
