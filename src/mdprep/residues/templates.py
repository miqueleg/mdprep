"""What can build a noncanonical residue, and what mdprep can propose for one.

AmberTools ships residue templates fitted to the protein force fields for a
few modified amino acids: the phosphorylated residues in
``phosaa14SB``/``phosaa19SB``, and five modified residues in
``ff14SB_modAA``/``ff19SB_modAA``. Where one exists it is
the better choice -- its charges were fitted with the protein force field's
own protocol, and it carries the backbone corrections ff19SB applies by
residue name -- so the planner offers it first.

A template is chosen by name, and the name fixes the protonation: SEP is a
phosphoserine with its phosphate dianionic, S1P the monoanion. Each entry
lists the heavy atoms the template has, so a residue whose atoms tleap would
not recognise is refused rather than built with atoms missing.

The second table is the chemistry mdprep can propose for the common
modifications when there is no template, as the SMILES of the residue capped
the way it is parameterised. A proposal is shown, explained and changeable;
nothing reaches a manifest that nobody accepted. Stereochemistry is left out
of each SMILES on purpose, as it is for cofactors: the structure states it,
and mdprep takes only bond orders and protonation from a SMILES.

Nothing here imports beyond the standard library: the web builder plans with
it under Pyodide.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

# Where each family of templates is loaded from, by protein force field.
# "protein" is the force field itself: ff14SB and ff19SB both define HYP.
LIBRARY_LEAPRC: dict[str, dict[str, str | None]] = {
    "phosaa": {"ff14SB": "leaprc.phosaa14SB", "ff19SB": "leaprc.phosaa19SB"},
    "modAA": {
        "ff14SB": "leaprc.protein.ff14SB_modAA",
        "ff19SB": "leaprc.protein.ff19SB_modAA",
    },
    "protein": {"ff14SB": None, "ff19SB": None},
}

# The OFF library each family's units are in, under $AMBERHOME/dat/leap/lib.
LIBRARY_FILES: dict[str, dict[str, str]] = {
    "phosaa": {"ff14SB": "phosaa14SB.lib", "ff19SB": "phosaa19SB.lib"},
    "modAA": {"ff14SB": "mod_amino.lib", "ff19SB": "mod_amino19.lib"},
    "protein": {"ff14SB": "amino12.lib", "ff19SB": "amino19.lib"},
}

# The frcmod with each family's own atom types and terms, under
# $AMBERHOME/dat/leap/parm, which its leaprc loads; the force field needs none.
LIBRARY_FRCMODS: dict[str, dict[str, str | None]] = {
    "phosaa": {"ff14SB": "frcmod.phosaa14SB", "ff19SB": "frcmod.phosaa19SB"},
    "modAA": {"ff14SB": "frcmod.ff14SBmodAA", "ff19SB": "frcmod.ff19SBmodAA"},
    "protein": {"ff14SB": None, "ff19SB": None},
}

LIBRARY_NAMES: dict[str, dict[str, str]] = {
    "phosaa": {"ff14SB": "phosaa14SB", "ff19SB": "phosaa19SB"},
    "modAA": {"ff14SB": "ff14SB_modAA", "ff19SB": "ff19SB_modAA"},
    "protein": {"ff14SB": "ff14SB", "ff19SB": "ff19SB"},
}


@dataclass(frozen=True)
class AmberResidueTemplate:
    """A residue template AmberTools ships for a protein force field."""

    # The unit's name, which is the residue name tleap builds it under.
    name: str
    net_charge: int
    library: str
    # Chemical component codes a deposition writes this residue as.
    deposited_as: frozenset[str]
    heavy_atoms: frozenset[str]
    description: str

    def library_name(self, forcefield: str) -> str:
        return LIBRARY_NAMES[self.library][forcefield]

    def leaprc(self, forcefield: str) -> str | None:
        return LIBRARY_LEAPRC[self.library][forcefield]

    def library_file(self, forcefield: str) -> str:
        return LIBRARY_FILES[self.library][forcefield]

    def frcmod_file(self, forcefield: str) -> str | None:
        return LIBRARY_FRCMODS[self.library][forcefield]


_SERINE_PHOSPHATE = frozenset(
    {"N", "CA", "CB", "OG", "P", "O1P", "O2P", "O3P", "C", "O"}
)
_THREONINE_PHOSPHATE = frozenset(
    {"N", "CA", "CB", "CG2", "OG1", "P", "O1P", "O2P", "O3P", "C", "O"}
)
_PHENYL = frozenset({"N", "CA", "CB", "CG", "CD1", "CE1", "CZ", "CE2", "CD2", "C", "O"})

# The monoanionic phosphotyrosine of phosaa (Y1P) is left out: it names the
# phenolic oxygen OG where a deposition's PTR names it OH, and tleap matches
# atoms by name. The SMILES route builds that protonation instead.
AMBER_RESIDUE_TEMPLATES: dict[str, AmberResidueTemplate] = {
    template.name: template
    for template in (
        AmberResidueTemplate(
            name="SEP",
            net_charge=-2,
            library="phosaa",
            deposited_as=frozenset({"SEP"}),
            heavy_atoms=_SERINE_PHOSPHATE,
            description="phosphoserine, phosphate dianionic",
        ),
        AmberResidueTemplate(
            name="S1P",
            net_charge=-1,
            library="phosaa",
            deposited_as=frozenset({"SEP"}),
            heavy_atoms=_SERINE_PHOSPHATE,
            description="phosphoserine, phosphate monoanionic",
        ),
        AmberResidueTemplate(
            name="TPO",
            net_charge=-2,
            library="phosaa",
            deposited_as=frozenset({"TPO"}),
            heavy_atoms=_THREONINE_PHOSPHATE,
            description="phosphothreonine, phosphate dianionic",
        ),
        AmberResidueTemplate(
            name="T1P",
            net_charge=-1,
            library="phosaa",
            deposited_as=frozenset({"TPO"}),
            heavy_atoms=_THREONINE_PHOSPHATE,
            description="phosphothreonine, phosphate monoanionic",
        ),
        AmberResidueTemplate(
            name="PTR",
            net_charge=-2,
            library="phosaa",
            deposited_as=frozenset({"PTR"}),
            heavy_atoms=_PHENYL | {"OH", "P", "O1P", "O2P", "O3P"},
            description="phosphotyrosine, phosphate dianionic",
        ),
        AmberResidueTemplate(
            name="ALY",
            net_charge=0,
            library="modAA",
            deposited_as=frozenset({"ALY"}),
            heavy_atoms=frozenset(
                {"N", "CA", "CB", "CG", "CD", "CE", "NZ", "CH", "OH", "CH3", "C", "O"}
            ),
            description="N6-acetyl-lysine",
        ),
        AmberResidueTemplate(
            name="MSE",
            net_charge=0,
            library="modAA",
            deposited_as=frozenset({"MSE"}),
            heavy_atoms=frozenset({"N", "CA", "CB", "CG", "SE", "CE", "C", "O"}),
            description="selenomethionine",
        ),
        AmberResidueTemplate(
            name="AZF",
            net_charge=0,
            library="modAA",
            deposited_as=frozenset({"4II"}),
            heavy_atoms=_PHENYL | {"N1", "N2", "N3"},
            description="p-azido-L-phenylalanine",
        ),
        AmberResidueTemplate(
            name="CYF",
            net_charge=0,
            library="modAA",
            deposited_as=frozenset({"4CF"}),
            heavy_atoms=_PHENYL | {"C1", "N1"},
            description="p-cyano-L-phenylalanine",
        ),
        AmberResidueTemplate(
            name="HYP",
            net_charge=0,
            library="protein",
            deposited_as=frozenset({"HYP"}),
            heavy_atoms=frozenset({"N", "CD", "CG", "OD1", "CB", "CA", "C", "O"}),
            description="trans-4-hydroxyproline",
        ),
    )
}


def templates_for(resname: str) -> tuple[AmberResidueTemplate, ...]:
    """The templates that can build a residue deposited under ``resname``.

    In the table's order: the state a phosphate takes at pH 7 first.
    """

    name = resname.strip().upper()
    return tuple(
        template
        for template in AMBER_RESIDUE_TEMPLATES.values()
        if name in template.deposited_as
    )


def missing_template_atoms(
    template: AmberResidueTemplate, heavy_atom_names: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Heavy atoms the residue has and the template lacks, and the reverse.

    The first would be dropped by tleap and the second built from nothing,
    so either makes the template the wrong one for this residue.
    """

    present = {name.strip().upper() for name in heavy_atom_names} - {"OXT"}
    return sorted(present - template.heavy_atoms), sorted(template.heavy_atoms - present)


@dataclass(frozen=True)
class KnownResidueChemistry:
    """A modification whose chemistry its name fixes, in its usual state."""

    resname: str
    # The residue between an acetyl and a methylamide, as it is parameterised.
    smiles: str
    net_charge: int
    # Heavy atoms of the residue itself, caps excluded, by element.
    heavy_atoms: tuple[tuple[str, int], ...]
    explanation: str

    def matches(self, elements: Iterable[str]) -> bool:
        counts = Counter(element.strip().upper() for element in elements)
        counts.pop("H", None)
        return counts == Counter({element.upper(): count for element, count in self.heavy_atoms})


# Why a phosphate on a residue is proposed dianionic, for its SMILES and for
# the template that has its charge.
PHOSPHATE_STATE_REASON = (
    "The second pKa of a phosphate on a serine, threonine or tyrosine is about "
    "6, so the dianion predominates at pH 7. A phosphate buried, or held by "
    "arginines or a metal, may keep a proton"
)

_PHOSPHATE_EXPLANATION = (
    "{name}: the residue between an acetyl and a methylamide, with its "
    "phosphate dianionic, net charge -2. "
    + PHOSPHATE_STATE_REASON
    + ": make one of its oxygens OH and the charge -1 then."
)

KNOWN_RESIDUE_CHEMISTRY: dict[str, KnownResidueChemistry] = {
    entry.resname: entry
    for entry in (
        KnownResidueChemistry(
            resname="SEP",
            smiles="CC(=O)NC(COP(=O)([O-])[O-])C(=O)NC",
            net_charge=-2,
            heavy_atoms=(("C", 3), ("N", 1), ("O", 5), ("P", 1)),
            explanation=_PHOSPHATE_EXPLANATION.format(name="SEP is phosphoserine"),
        ),
        KnownResidueChemistry(
            resname="TPO",
            smiles="CC(=O)NC(C(C)OP(=O)([O-])[O-])C(=O)NC",
            net_charge=-2,
            heavy_atoms=(("C", 4), ("N", 1), ("O", 5), ("P", 1)),
            explanation=_PHOSPHATE_EXPLANATION.format(name="TPO is phosphothreonine"),
        ),
        KnownResidueChemistry(
            resname="PTR",
            smiles="CC(=O)NC(Cc1ccc(OP(=O)([O-])[O-])cc1)C(=O)NC",
            net_charge=-2,
            heavy_atoms=(("C", 9), ("N", 1), ("O", 5), ("P", 1)),
            explanation=_PHOSPHATE_EXPLANATION.format(name="PTR is phosphotyrosine"),
        ),
        KnownResidueChemistry(
            resname="MLZ",
            smiles="CC(=O)NC(CCCC[NH2+]C)C(=O)NC",
            net_charge=1,
            heavy_atoms=(("C", 7), ("N", 2), ("O", 1)),
            explanation=(
                "MLZ is N6-methyl-lysine. Proposed with its amine protonated, "
                "net charge +1: methylation leaves the amine's pKa near "
                "lysine's, above 10."
            ),
        ),
        KnownResidueChemistry(
            resname="MLY",
            smiles="CC(=O)NC(CCCC[NH+](C)C)C(=O)NC",
            net_charge=1,
            heavy_atoms=(("C", 8), ("N", 2), ("O", 1)),
            explanation=(
                "MLY is N6,N6-dimethyl-lysine. Proposed with its amine "
                "protonated, net charge +1: its pKa stays above 10."
            ),
        ),
        KnownResidueChemistry(
            resname="M3L",
            smiles="CC(=O)NC(CCCC[N+](C)(C)C)C(=O)NC",
            net_charge=1,
            heavy_atoms=(("C", 9), ("N", 2), ("O", 1)),
            explanation=(
                "M3L is N6,N6,N6-trimethyl-lysine, a quaternary ammonium: "
                "net charge +1 whatever the pH."
            ),
        ),
        KnownResidueChemistry(
            resname="ALY",
            smiles="CC(=O)NC(CCCCNC(C)=O)C(=O)NC",
            net_charge=0,
            heavy_atoms=(("C", 8), ("N", 2), ("O", 2)),
            explanation=(
                "ALY is N6-acetyl-lysine, an amide: neutral, net charge 0."
            ),
        ),
        KnownResidueChemistry(
            resname="KCX",
            smiles="CC(=O)NC(CCCCNC(=O)[O-])C(=O)NC",
            net_charge=-1,
            heavy_atoms=(("C", 7), ("N", 2), ("O", 3)),
            explanation=(
                "KCX is N6-carboxy-lysine, the carbamate of a lysine. Proposed "
                "ionised, net charge -1, as it binds the metal ions of the "
                "enzymes that carry it."
            ),
        ),
        KnownResidueChemistry(
            resname="CGU",
            smiles="CC(=O)NC(CC(C(=O)[O-])C(=O)[O-])C(=O)NC",
            net_charge=-2,
            heavy_atoms=(("C", 6), ("N", 1), ("O", 5)),
            explanation=(
                "CGU is gamma-carboxyglutamate (Gla), the malonate the vitamin "
                "K-dependent carboxylase makes of a glutamate, which binds calcium "
                "in osteocalcin and the Gla domains of the coagulation factors. "
                "Proposed with both carboxylates ionised, net charge -2: their pKa "
                "values are well below 7."
            ),
        ),
        KnownResidueChemistry(
            resname="CSO",
            smiles="CC(=O)NC(CSO)C(=O)NC",
            net_charge=0,
            heavy_atoms=(("C", 3), ("N", 1), ("O", 2), ("S", 1)),
            explanation=(
                "CSO is S-hydroxycysteine, a cysteine sulfenic acid. Proposed "
                "neutral, net charge 0; its pKa is near 7 in some proteins, so "
                "check yours."
            ),
        ),
        KnownResidueChemistry(
            resname="CSD",
            smiles="CC(=O)NC(CS(=O)[O-])C(=O)NC",
            net_charge=-1,
            heavy_atoms=(("C", 3), ("N", 1), ("O", 3), ("S", 1)),
            explanation=(
                "CSD is 3-sulfinoalanine, a cysteine sulfinic acid. Its pKa is "
                "about 2, so it is proposed as the sulfinate, net charge -1."
            ),
        ),
        KnownResidueChemistry(
            resname="OCS",
            smiles="CC(=O)NC(CS(=O)(=O)[O-])C(=O)NC",
            net_charge=-1,
            heavy_atoms=(("C", 3), ("N", 1), ("O", 4), ("S", 1)),
            explanation=(
                "OCS is cysteine sulfonic acid, a strong acid: proposed as the "
                "sulfonate, net charge -1."
            ),
        ),
        KnownResidueChemistry(
            resname="CME",
            smiles="CC(=O)NC(CSSCCO)C(=O)NC",
            net_charge=0,
            heavy_atoms=(("C", 5), ("N", 1), ("O", 2), ("S", 2)),
            explanation=(
                "CME is S,S-(2-hydroxyethyl)thiocysteine, a mixed disulfide "
                "with mercaptoethanol: neutral, net charge 0."
            ),
        ),
    )
}


def known_residue_chemistry(
    resname: str, elements: Iterable[str]
) -> KnownResidueChemistry | None:
    """The chemistry to propose for a residue, if its name and atoms fix it."""

    entry = KNOWN_RESIDUE_CHEMISTRY.get(resname.strip().upper())
    if entry is None or not entry.matches(elements):
        return None
    return entry


def proposed_template(
    resname: str, templates: Iterable[str]
) -> tuple[str, str] | None:
    """The template to propose for a phosphorylated residue, and why.

    The dianion, for the reason its SMILES would be proposed dianionic; the
    monoanion's template is the alternative named.
    """

    names = list(templates)
    chemistry = KNOWN_RESIDUE_CHEMISTRY.get(resname.strip().upper())
    if chemistry is None or PHOSPHATE_STATE_REASON not in chemistry.explanation:
        return None
    dianion = [name for name in names if AMBER_RESIDUE_TEMPLATES[name].net_charge == -2]
    others = [name for name in names if name not in dianion]
    if not dianion:
        return None
    chosen = AMBER_RESIDUE_TEMPLATES[dianion[0]]
    return chosen.name, (
        f"{chosen.name}: {chosen.description}, net charge {chosen.net_charge:+d}. "
        + PHOSPHATE_STATE_REASON
        + (f": choose {' or '.join(others)} then." if others else ".")
    )
