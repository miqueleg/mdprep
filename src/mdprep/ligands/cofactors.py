"""Cofactors whose metal mdprep reads out into a residue of its own.

A heme is deposited with its iron inside it. mdprep moves the iron into a
residue of its own as the structure is read (``structure/pdb_fields.py``), so
it can be modelled as a metal site, and what remains is a porphyrin: a ligand
that coordinates the iron through four nitrogens, and that needs a SMILES like
any other ligand whose hydrogens the crystal did not resolve.

Typing a porphyrin SMILES correctly is not something to ask of anyone in the
middle of a form. For a cofactor whose chemistry is fixed by its name it is
known, so the planner proposes it -- filled in, explained, and changeable. A
proposal is not a default: nothing reaches a manifest that nobody accepted.
The one choice it makes, the protonation of the propionates, is stated with it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mdprep.structure.models import ResidueRecord


@dataclass(frozen=True)
class MetalFreeCofactor:
    """A cofactor without its metal, as it binds the metal."""

    resname: str
    smiles: str
    net_charge: int
    # The heavy atoms the residue has once its metal is out, so a residue
    # that only shares the name -- or still holds its metal -- gets nothing.
    heavy_atoms: tuple[tuple[str, int], ...]
    explanation: str


KNOWN_METAL_FREE_COFACTORS: dict[str, MetalFreeCofactor] = {
    "HEM": MetalFreeCofactor(
        resname="HEM",
        # Protoporphyrin IX, from the PDB's PP9, with the two pyrrole
        # nitrogens that carry a proton in the free base deprotonated, as
        # they are on the iron, and both propionates ionised.
        smiles=(
            "Cc1c2[n-]c(cc3nc(cc4[n-]c(cc5nc(c2)c(C)c5C=C)c(C)c4C=C)"
            "c(C)c3CCC([O-])=O)c1CCC([O-])=O"
        ),
        net_charge=-4,
        heavy_atoms=(("C", 34), ("N", 4), ("O", 4)),
        explanation=(
            "HEM is heme b, protoporphyrin IX around an iron that mdprep has "
            "moved into a residue of its own. Proposed as the porphyrin binds "
            "the iron, with both pyrrole nitrogens deprotonated, and with both "
            "propionates ionised as they are at pH 7: net charge -4. Change it "
            "if a propionate is protonated in your structure."
        ),
    ),
}


def known_metal_free_cofactor(residue: ResidueRecord) -> MetalFreeCofactor | None:
    """The cofactor this residue is, without its metal, if mdprep knows it."""

    entry = KNOWN_METAL_FREE_COFACTORS.get(residue.id.resname.strip().upper())
    if entry is None:
        return None
    heavy = Counter(
        (atom.element or "").strip().upper()
        for atom in residue.atoms
        if (atom.element or "").strip().upper() not in {"H", "D"}
    )
    if heavy != Counter(dict(entry.heavy_atoms)):
        return None
    return entry
