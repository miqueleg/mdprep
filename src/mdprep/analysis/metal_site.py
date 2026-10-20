"""The bonded metal site's force constants, named by the atoms they hold.

MCPB.py gives the metal and each atom bound to it a new atom type (M1, Y1,
Y2...) and writes the Seminario force constants for them to a frcmod. Read on
its own the frcmod says ``M1-Y6 20.8 2.2403``; read against the topology it
says the iron-water bond is the softest in the site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mdprep.analysis.prmtop import AmberTopology


@dataclass(frozen=True)
class FrcmodTerm:
    types: tuple[str, ...]
    force_constant: float  # kcal/mol/A^2 for bonds, kcal/mol/rad^2 for angles
    equilibrium: float  # A for bonds, degrees for angles
    comment: str


def read_frcmod_terms(path: str | Path) -> dict[str, list[FrcmodTerm]]:
    """BOND and ANGL entries of a frcmod. Atom types are two columns wide."""

    terms: dict[str, list[FrcmodTerm]] = {"BOND": [], "ANGL": []}
    section = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        keyword = line.strip().upper()
        if keyword in {"MASS", "BOND", "ANGL", "ANGLE", "DIHE", "IMPR", "NONB", "NONBON"}:
            section = "ANGL" if keyword in {"ANGL", "ANGLE"} else keyword
            continue
        if section not in terms or not line.strip():
            continue
        count = 2 if section == "BOND" else 3
        width = 3 * count - 1
        types = tuple(part.strip() for part in line[:width].split("-"))
        fields = line[width:].split()
        if len(types) != count or len(fields) < 2:
            continue
        try:
            constant, value = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        terms[section].append(FrcmodTerm(types, constant, value, " ".join(fields[2:])))
    return terms


def site_bonds(frcmod: str | Path, topology: AmberTopology) -> list[dict[str, object]]:
    """Each fitted metal-donor bond, with the atoms whose types it names."""

    by_type: dict[str, list[int]] = {}
    for atom in topology.atoms:
        by_type.setdefault(atom.type, []).append(atom.index)
    bonded = {frozenset(pair) for pair in topology.metal_bonds()}
    rows = []
    for term in read_frcmod_terms(frcmod)["BOND"]:
        first, second = term.types
        pairs = [
            (a, b)
            for a in by_type.get(first, [])
            for b in by_type.get(second, [])
            if frozenset((a, b)) in bonded
        ]
        for a, b in pairs:
            metal, donor = (a, b) if topology.is_metal(a) else (b, a)
            rows.append(
                {
                    "metal": topology.label(metal),
                    "donor": topology.label(donor),
                    "types": f"{topology.atoms[metal].type}-{topology.atoms[donor].type}",
                    "k_kcal_mol_A2": term.force_constant,
                    "r0_A": term.equilibrium,
                }
            )
    return sorted(rows, key=lambda row: -row["k_kcal_mol_A2"])
