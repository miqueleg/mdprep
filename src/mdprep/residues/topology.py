"""Checks that a built topology holds each kept residue as it was meant to be.

tleap reports nothing when a residue it was told about is built wrong in a
way it considers legal: a residue whose head was never connected is a valid
residue with a dangling nitrogen. So the topology itself is read back
(``mdprep.analysis.prmtop``, no ParmEd needed), and every kept residue must
be there under its name, carry the charge it was built with, and be bonded
by peptide bonds to the residues on either side.
"""

from __future__ import annotations

from pathlib import Path

from mdprep.analysis.prmtop import AmberTopology, read_prmtop
from mdprep.config.models import ManifestConfig
from mdprep.residues.workflow import expected_charge, final_resname

CHARGE_TOLERANCE_E = 1.0e-3


def kept_residue_checks(
    prmtop: str | Path, manifest: ManifestConfig
) -> tuple[list[dict[str, object]], list[str]]:
    """One check per kept residue name, and the problems found."""

    if not manifest.noncanonical_residues:
        return [], []
    topology = read_prmtop(prmtop)
    checks: list[dict[str, object]] = []
    problems: list[str] = []
    for block in manifest.noncanonical_residues:
        name = final_resname(block)
        charge = expected_charge(block)
        found = [residue for residue in topology.residues if residue.name == name]
        charges = [
            round(sum(topology.atoms[index].charge for index in residue.atom_indices), 6)
            for residue in found
        ]
        unbonded = [
            f"{name}{residue.index + 1}"
            for residue in found
            if not _peptide_bonded(topology, residue.index)
        ]
        wrong_charge = [value for value in charges if abs(value - charge) > CHARGE_TOLERANCE_E]
        ok = len(found) == len(block.residues) and not unbonded and not wrong_charge
        checks.append(
            {
                "resname": block.resname,
                "built_as": name,
                "expected_count": len(block.residues),
                "count": len(found),
                "expected_charge": charge,
                "charges": charges,
                "peptide_bonded": not unbonded,
                "ok": ok,
            }
        )
        if len(found) != len(block.residues):
            problems.append(
                f"the topology holds {len(found)} residue(s) named {name} where "
                f"noncanonical_residues {block.resname} keeps {len(block.residues)}"
            )
        if wrong_charge:
            problems.append(
                f"{name} carries {', '.join(f'{value:+.4f}' for value in wrong_charge)} e "
                f"in the topology, not {charge:+d}"
            )
        if unbonded:
            problems.append(
                f"{', '.join(unbonded)} is not bonded by a peptide bond to the residue "
                "on each side of it"
            )
    return checks, problems


def _peptide_bonded(topology: AmberTopology, index: int) -> bool:
    """Whether residue ``index``'s N is bonded to a C before it, and its C to an N after."""

    residue = topology.residues[index]
    own = {topology.atoms[atom].name: atom for atom in residue.atom_indices}
    if "N" not in own or "C" not in own:
        return False
    partners: dict[int, set[int]] = {}
    for first, second in topology.bonds:
        partners.setdefault(first, set()).add(second)
        partners.setdefault(second, set()).add(first)

    def bonded_to(atom: int, name: str, offset: int) -> bool:
        return any(
            topology.atoms[other].name == name
            and topology.atoms[other].residue_index == index + offset
            for other in partners.get(atom, ())
        )

    return bonded_to(own["N"], "C", -1) and bonded_to(own["C"], "N", 1)
