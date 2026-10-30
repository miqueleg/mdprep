"""Checks that a built topology holds each kept residue as it was meant to be.

tleap reports nothing when a residue it was told about is built wrong in a
way it considers legal: a residue whose head was never connected is a valid
residue with a dangling nitrogen. So the topology itself is read back
(``mdprep.analysis.prmtop``, no ParmEd needed), and every kept residue must
be there under its name, carry the charge it was built with, and be bonded
by peptide bonds to the residues on either side.

A kept residue on a bonded metal is the exception to the first two: MCPB.py
gave it a template of its own, under a new name, and refitted its charges
with the rest of the site, which leaves no residue of the site an integer.
It is counted under the name MCPB.py gave it, and its charge is reported
rather than checked; the peptide bonds still have to be there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from mdprep.analysis.prmtop import AmberTopology, read_prmtop
from mdprep.config.models import ManifestConfig
from mdprep.residues.workflow import expected_charge, final_resname

CHARGE_TOLERANCE_E = 1.0e-3


def kept_residue_checks(
    prmtop: str | Path,
    manifest: ManifestConfig,
    mcpb_renames: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], list[str]]:
    """One check per kept residue name, and the problems found.

    ``mcpb_renames`` are MCPB.py's renames of the residues of a bonded site,
    as its result reports them.
    """

    if not manifest.noncanonical_residues:
        return [], []
    topology = read_prmtop(prmtop)
    checks: list[dict[str, object]] = []
    problems: list[str] = []
    for block in manifest.noncanonical_residues:
        name = final_resname(block)
        charge = expected_charge(block)
        refitted_names = {
            str(rename["final_resname"])
            for rename in mcpb_renames
            if rename.get("original_resname") == name
        }
        found = [residue for residue in topology.residues if residue.name == name]
        refitted = [residue for residue in topology.residues if residue.name in refitted_names]
        charges = [
            round(sum(topology.atoms[index].charge for index in residue.atom_indices), 6)
            for residue in found
        ]
        refitted_charges = {
            f"{residue.name}{residue.index + 1}": round(
                sum(topology.atoms[index].charge for index in residue.atom_indices), 6
            )
            for residue in refitted
        }
        unbonded = [
            f"{residue.name}{residue.index + 1}"
            for residue in (*found, *refitted)
            if not _peptide_bonded(topology, residue.index)
        ]
        wrong_charge = [value for value in charges if abs(value - charge) > CHARGE_TOLERANCE_E]
        count = len(found) + len(refitted)
        ok = count == len(block.residues) and not unbonded and not wrong_charge
        checks.append(
            {
                "resname": block.resname,
                "built_as": name,
                "expected_count": len(block.residues),
                "count": count,
                "expected_charge": charge,
                "charges": charges,
                "refitted_by_mcpb": refitted_charges,
                "peptide_bonded": not unbonded,
                "ok": ok,
            }
        )
        if count != len(block.residues):
            problems.append(
                f"the topology holds {count} residue(s) named {name}"
                + (
                    f" or MCPB.py's {', '.join(sorted(refitted_names))}"
                    if refitted_names
                    else ""
                )
                + f" where noncanonical_residues {block.resname} keeps {len(block.residues)}"
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
