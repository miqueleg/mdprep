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

The same holds for the organic cofactors a crystal holds without a metal --
the flavins of flavin-dependent halogenases, SAM in a fluorinase, the
nicotinamides: their names fix their chemistry up to a protonation state,
and the state at pH 7 is proposed with the reason for it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mdprep.structure.models import ResidueRecord


@dataclass(frozen=True)
class KnownCofactor:
    """A cofactor whose chemistry its name fixes, in its usual state."""

    resname: str
    smiles: str
    net_charge: int
    # The heavy atoms the residue must have -- a metal cofactor's once its
    # metal is out -- so a residue that only shares the name, lost atoms to
    # disorder, or still holds its metal gets nothing.
    heavy_atoms: tuple[tuple[str, int], ...]
    explanation: str


# A cofactor without its metal, as it binds the metal.
MetalFreeCofactor = KnownCofactor

KNOWN_METAL_FREE_COFACTORS: dict[str, KnownCofactor] = {
    "HEM": KnownCofactor(
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


_ADENOSINE = "CC1OC(n2cnc3c(N)ncnc32)C(O)C1O"
_DIPHOSPHATE_TO_ADENOSINE = "COP(=O)([O-])OP(=O)([O-])OCC2OC(n3cnc4c(N)ncnc43)C({phosphate})C2O"
_FLAVIN = "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(CC(O)C(O)C(O){tail})c2cc1C"
_NICOTINAMIDE_RIBOSE = {
    "oxidised": "NC(=O)c1ccc[n+](c1)C1OC({tail})C(O)C1O",
    "reduced": "NC(=O)C1=CN(C=CC1)C1OC({tail})C(O)C1O",
}

# Cofactors without a metal. Stereochemistry is left out of each SMILES on
# purpose: it is the structure's to state, and mdprep takes only bond orders
# and protonation from a SMILES.
KNOWN_ORGANIC_COFACTORS: dict[str, KnownCofactor] = {
    "FAD": KnownCofactor(
        resname="FAD",
        smiles=_FLAVIN.format(tail="COP(=O)([O-])OP(=O)([O-])OCC3OC(n4cnc5c(N)ncnc54)C(O)C3O"),
        net_charge=-2,
        heavy_atoms=(("C", 27), ("N", 9), ("O", 15), ("P", 2)),
        explanation=(
            "FAD is flavin adenine dinucleotide, oxidised: the form a "
            "deposition of a flavin-dependent halogenase holds. Proposed with "
            "the diphosphate fully ionised, as at pH 7, and the flavin's N3 "
            "protonated: net charge -2. The reduced FADH- that reacts with O2 "
            "is a different molecule, with two more hydrogens; change the "
            "SMILES and the charge if that is what you are simulating."
        ),
    ),
    "FMN": KnownCofactor(
        resname="FMN",
        smiles=_FLAVIN.format(tail="COP(=O)([O-])[O-]"),
        net_charge=-2,
        heavy_atoms=(("C", 17), ("N", 4), ("O", 9), ("P", 1)),
        explanation=(
            "FMN is flavin mononucleotide, oxidised. Proposed with the "
            "phosphate as its dianion, as at pH 7 (its second pKa is near 6), "
            "and the flavin's N3 protonated: net charge -2. Change both if your "
            "flavin is reduced or its phosphate protonated."
        ),
    ),
    "SAM": KnownCofactor(
        resname="SAM",
        smiles="C[S+](CCC([NH3+])C(=O)[O-])" + _ADENOSINE,
        net_charge=1,
        heavy_atoms=(("C", 15), ("N", 6), ("O", 5), ("S", 1)),
        explanation=(
            "SAM is S-adenosylmethionine. Proposed as it is at pH 7: the "
            "sulfonium positive and the amino acid a zwitterion, net charge +1."
        ),
    ),
    "SAH": KnownCofactor(
        resname="SAH",
        smiles="[NH3+]C(CCS" + _ADENOSINE + ")C(=O)[O-]",
        net_charge=0,
        heavy_atoms=(("C", 14), ("N", 6), ("O", 5), ("S", 1)),
        explanation=(
            "SAH is S-adenosylhomocysteine, SAM once its methyl has gone. "
            "Proposed with the amino acid a zwitterion, as at pH 7: net "
            "charge 0."
        ),
    ),
    "NAD": KnownCofactor(
        resname="NAD",
        smiles=_NICOTINAMIDE_RIBOSE["oxidised"].format(
            tail=_DIPHOSPHATE_TO_ADENOSINE.format(phosphate="O")
        ),
        net_charge=-1,
        heavy_atoms=(("C", 21), ("N", 7), ("O", 14), ("P", 2)),
        explanation=(
            "NAD is NAD+, oxidised. Proposed with the diphosphate fully "
            "ionised, as at pH 7, and the nicotinamide ring carrying its "
            "positive charge: net charge -1."
        ),
    ),
    "NAI": KnownCofactor(
        resname="NAI",
        smiles=_NICOTINAMIDE_RIBOSE["reduced"].format(
            tail=_DIPHOSPHATE_TO_ADENOSINE.format(phosphate="O")
        ),
        net_charge=-2,
        heavy_atoms=(("C", 21), ("N", 7), ("O", 14), ("P", 2)),
        explanation=(
            "NAI is NADH, reduced. Proposed with the diphosphate fully "
            "ionised, as at pH 7: net charge -2. Its nicotinamide ring is not "
            "aromatic; if mdprep reports that the ring's bond lengths "
            "contradict that, the deposition refined it flat, as NAD+, and "
            "you have to decide which one you are simulating."
        ),
    ),
    "NAP": KnownCofactor(
        resname="NAP",
        smiles=_NICOTINAMIDE_RIBOSE["oxidised"].format(
            tail=_DIPHOSPHATE_TO_ADENOSINE.format(phosphate="OP(=O)([O-])[O-]")
        ),
        net_charge=-3,
        heavy_atoms=(("C", 21), ("N", 7), ("O", 17), ("P", 3)),
        explanation=(
            "NAP is NADP+, oxidised. Proposed with the diphosphate and the "
            "2'-phosphate fully ionised, as at pH 7, and the nicotinamide ring "
            "carrying its positive charge: net charge -3."
        ),
    ),
    "NDP": KnownCofactor(
        resname="NDP",
        smiles=_NICOTINAMIDE_RIBOSE["reduced"].format(
            tail=_DIPHOSPHATE_TO_ADENOSINE.format(phosphate="OP(=O)([O-])[O-]")
        ),
        net_charge=-4,
        heavy_atoms=(("C", 21), ("N", 7), ("O", 17), ("P", 3)),
        explanation=(
            "NDP is NADPH, reduced. Proposed with the diphosphate and the "
            "2'-phosphate fully ionised, as at pH 7: net charge -4. Its "
            "nicotinamide ring is not aromatic; if mdprep reports that the "
            "ring's bond lengths contradict that, the deposition refined it "
            "flat, as NADP+, and you have to decide which one you are "
            "simulating."
        ),
    ),
}


def known_metal_free_cofactor(residue: ResidueRecord) -> KnownCofactor | None:
    """The cofactor this residue is, without its metal, if mdprep knows it."""

    return _matching(KNOWN_METAL_FREE_COFACTORS, residue)


def known_organic_cofactor(residue: ResidueRecord) -> KnownCofactor | None:
    """The metal-free cofactor this residue is, if mdprep knows it."""

    return _matching(KNOWN_ORGANIC_COFACTORS, residue)


def _matching(table: dict[str, KnownCofactor], residue: ResidueRecord) -> KnownCofactor | None:
    # Only a residue with exactly the heavy atoms of the entry: one that
    # shares the name but lost atoms to disorder -- or still holds its
    # metal -- gets nothing proposed.
    entry = table.get(residue.id.resname.strip().upper())
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
