"""The charges on every atom of a prepared system, and what they add up to.

Most residues carry the integer charge of their force-field template. A
bonded metal site does not: MCPB.py fits one set of RESP charges to the
metal and everything bound to it, so the iron of an Fe(II) site carries far
less than +2 and its histidines are no longer neutral -- only the site as a
whole has the integer charge of its parts. That is the thing worth seeing,
and ``metal_site`` lays it out.
"""

from __future__ import annotations

from dataclasses import dataclass

from mdprep.analysis.prmtop import AmberTopology
from mdprep.structure.classify import WATER_RESIDUES

# Formal charges of the amino-acid templates, for comparison with what a
# residue actually carries.
TEMPLATE_CHARGES = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIP": 1, "CYM": -1,
    "ASH": 0, "GLH": 0, "LYN": 0, "HID": 0, "HIE": 0, "CYX": 0,
}


@dataclass(frozen=True)
class ResidueCharge:
    index: int
    name: str
    kind: str  # protein, water, ion, metal_site, ligand
    charge: float
    atoms: int

    @property
    def label(self) -> str:
        return f"{self.name} {self.index + 1}"


def _site_residues(topology: AmberTopology) -> set[int]:
    site: set[int] = set()
    for metal, partner in topology.metal_bonds():
        site.add(topology.atoms[metal].residue_index)
        site.add(topology.atoms[partner].residue_index)
    for atom in topology.atoms:
        residue = topology.residues[atom.residue_index]
        if topology.is_metal(atom.index) and residue.stop - residue.start == 1:
            # A metal on its own -- a nonbonded ion mdprep was asked to keep.
            site.add(atom.residue_index)
    return site


def residue_charges(topology: AmberTopology) -> list[ResidueCharge]:
    site = _site_residues(topology)
    bonded = {atom for bond in topology.bonds for atom in bond}
    result = []
    for residue in topology.residues:
        indices = list(residue.atom_indices)
        names = {topology.atoms[index].name for index in indices}
        if residue.index in site:
            kind = "metal_site"
        elif residue.name in WATER_RESIDUES:
            kind = "water"
        elif len(indices) == 1 and indices[0] not in bonded:
            kind = "ion"
        elif {"N", "CA", "C"} <= names:
            kind = "protein"
        else:
            kind = "ligand"
        result.append(
            ResidueCharge(
                index=residue.index,
                name=residue.name,
                kind=kind,
                charge=sum(topology.atoms[index].charge for index in indices),
                atoms=len(indices),
            )
        )
    return result


def summary(topology: AmberTopology) -> dict[str, object]:
    """Totals by kind, ion counts, and residues whose charge is not an integer."""

    residues = residue_charges(topology)
    by_kind: dict[str, float] = {}
    for residue in residues:
        by_kind[residue.kind] = by_kind.get(residue.kind, 0.0) + residue.charge
    ions: dict[str, int] = {}
    for residue in residues:
        if residue.kind == "ion":
            ions[residue.name] = ions.get(residue.name, 0) + 1
    fractional = [
        {"residue": residue.label, "kind": residue.kind, "charge": round(residue.charge, 4)}
        for residue in residues
        if abs(residue.charge - round(residue.charge)) > 1e-3
    ]
    return {
        "total": round(sum(residue.charge for residue in residues), 4),
        "by_kind": {kind: round(value, 4) for kind, value in sorted(by_kind.items())},
        "residues_by_kind": {
            kind: sum(1 for residue in residues if residue.kind == kind)
            for kind in sorted({residue.kind for residue in residues})
        },
        "ions": dict(sorted(ions.items())),
        "fractional_residues": fractional,
    }


def metal_sites(topology: AmberTopology) -> list[dict[str, object]]:
    """Each metal site on its own, as ``metal_site`` lays one out.

    A dimer has one site per copy, and each is a separate integer charge:
    added together, two -1 sites read as one site of -2. A metal joins the
    residues bound to it, and two metals sharing a residue are one site.
    """

    residues = {residue.index: residue for residue in residue_charges(topology)}
    parent = {index: index for index in _site_residues(topology)}

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for metal, partner in topology.metal_bonds():
        first = root(topology.atoms[metal].residue_index)
        second = root(topology.atoms[partner].residue_index)
        if first != second:
            parent[max(first, second)] = min(first, second)
    groups: dict[int, set[int]] = {}
    for index in parent:
        groups.setdefault(root(index), set()).add(index)
    sites = []
    for members in sorted(groups.values(), key=min):
        bonds = [
            (metal, donor)
            for metal, donor in topology.metal_bonds()
            if topology.atoms[metal].residue_index in members
        ]
        sites.append(_site_table(topology, sorted(members), bonds, residues))
    return sites


def metal_site(topology: AmberTopology) -> dict[str, object]:
    """Every metal site together: each residue's charge, each metal bond.

    For one site that is the site; with several, see ``metal_sites``.
    """

    site = sorted(_site_residues(topology))
    bonds = topology.metal_bonds()
    residues = {residue.index: residue for residue in residue_charges(topology)}
    return _site_table(topology, site, bonds, residues)


def _site_table(
    topology: AmberTopology,
    site: list[int],
    bonds: list[tuple[int, int]],
    residues: dict[int, ResidueCharge],
) -> dict[str, object]:
    return {
        "metals": [
            topology.label(atom)
            for index in site
            for atom in topology.residues[index].atom_indices
            if topology.is_metal(atom)
        ],
        "residues": [
            {
                "residue": residues[index].label,
                "charge": round(residues[index].charge, 4),
                "template_charge": TEMPLATE_CHARGES.get(residues[index].name),
            }
            for index in site
        ],
        "total": round(sum(residues[index].charge for index in site), 4),
        "bonds": [
            {
                "metal": topology.label(metal),
                "metal_charge": round(topology.atoms[metal].charge, 4),
                "donor": topology.label(donor),
                "donor_type": topology.atoms[donor].type,
                "donor_charge": round(topology.atoms[donor].charge, 4),
            }
            for metal, donor in bonds
        ],
    }


def atom_charges(topology: AmberTopology, residue_indices: list[int]) -> list[dict[str, object]]:
    """One row per atom of the given residues: name, type, element, charge."""

    rows = []
    for index in residue_indices:
        residue = topology.residues[index]
        for atom_index in residue.atom_indices:
            atom = topology.atoms[atom_index]
            rows.append(
                {
                    "residue": f"{residue.name} {residue.index + 1}",
                    "atom": atom.name,
                    "type": atom.type,
                    "element": atom.element,
                    "charge": round(atom.charge, 4),
                }
            )
    return rows
