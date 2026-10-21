"""Structure-aware manifest decision planning.

A manifest exposes several hundred settings, but a given structure only needs a
few dozen of them, and only a handful are decisions a researcher must actually
make. This module turns an input structure into that short list, so that every
front end -- the interactive CLI wizard, a generated web form, a notebook --
asks the same questions, in the same order, with the same defaults.

The list is data, not a user interface. Rendering it is the front end's job.

Chemistry-sensitive decisions carry ``default=None`` and
``requires_user_input=True``. They have no suggested value on purpose: ligand
net charge, metal oxidation state and the fate of unknown heterogens cannot be
guessed from a PDB file, and a pre-filled form field is an invitation to accept
a guess without noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from mdprep.ligands.cofactors import (
    KNOWN_METAL_FREE_COFACTORS,
    KNOWN_ORGANIC_COFACTORS,
    known_metal_free_cofactor,
    known_organic_cofactor,
)
from mdprep.structure.classify import (
    COORDINATING_METAL_ELEMENTS,
    LEAP_RESERVED_RESIDUE_NAMES,
    describe_reserved_residue_names,
    free_residue_name,
    is_metal_ion_residue,
    is_water_residue,
)
from mdprep.structure.completeness import chain_breaks, incomplete_residues
from mdprep.structure.coordination import (
    COORDINATION_CUTOFF_ANGSTROM,
    metal_bound_waters,
    metal_donor_atoms,
)
from mdprep.structure.heterogens import (
    CRYSTALLISATION_ADDITIVES,
    LABELLING_MODIFICATIONS,
    HeterogenGroups,
    classify_heterogens,
)
from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.models import ResidueId, ResidueRecord
from mdprep.structure.oligomer import OligomerAnalysis, analyse_oligomer
from mdprep.structure.pdb import AltlocPolicy


DecisionKind = Literal["choice", "integer", "number", "text", "boolean"]

# What a 3D viewer should show while a question is on screen.
#   none      -- the question has no structural aspect (a project name, a step count)
#   targets   -- emphasise and frame `targets` / `target_resnames`
#   structure -- show the whole system (force field, solvation, box)
#   chains    -- the whole system with each chain in its own colour, and the
#                targets emphasised: which chains to keep, which copies match
FocusMode = Literal["none", "targets", "structure", "chains"]


class DecisionError(ValueError):
    """Raised when a decision plan cannot be built or answered.

    ``decision_id`` names the question whose answer would resolve it, when
    there is one, so a front end can take the user back to it rather than
    leave them to find it.
    """

    def __init__(self, message: str, *, decision_id: str | None = None):
        super().__init__(message)
        self.decision_id = decision_id


class ReservedResidueNameError(DecisionError):
    """A ligand carries a residue name an Amber force field already defines.

    Carries the offending residues and a free name for each, so a front end
    can offer the rename as an action on the input rather than leaving the
    user to edit a megabyte of PDB by hand. The rename is the user's to make:
    mdprep will not make it silently, which is what it used to do.
    """

    def __init__(self, message: str, residues: list[dict[str, object]]):
        super().__init__(message)
        self.residues = residues

    def to_dict(self) -> dict[str, object]:
        return {"message": str(self), "residues": self.residues}



@dataclass(frozen=True)
class Choice:
    """One allowed value of a ``choice`` decision."""

    value: str
    label: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"value": self.value, "label": self.label, "detail": self.detail}


@dataclass(frozen=True)
class Condition:
    """Ask a decision only when an earlier answer is one of ``values``."""

    decision_id: str
    values: tuple[str, ...]

    def is_met(self, answers: Mapping[str, Any]) -> bool:
        if self.decision_id not in answers:
            return False
        return _as_answer_token(answers[self.decision_id]) in self.values

    def to_dict(self) -> dict[str, object]:
        return {"decision_id": self.decision_id, "values": list(self.values)}


@dataclass(frozen=True)
class ResidueTarget:
    """The residue a decision is about, carried structurally.

    Keeping this on the decision means a front end can build a manifest from
    the plan plus the answers alone, without re-reading the structure.
    """

    chain: str
    resname: str
    resid: int
    icode: str | None = None
    element: str | None = None
    atom_name: str | None = None

    def to_selector(self) -> dict[str, Any]:
        """Residue-level selector, as a ligand block expects."""

        return {
            "chain": self.chain,
            "resname": self.resname,
            "resid": self.resid,
            "icode": self.icode,
        }

    def to_atom_selector(self) -> dict[str, Any]:
        """Atom-level selector, as a metal ion block expects."""

        if self.atom_name is None:
            raise DecisionError(
                f"No atom name recorded for {self.resname}{self.resid}"
            )
        return {**self.to_selector(), "atom_name": self.atom_name}

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_selector(),
            "element": self.element,
            "atom_name": self.atom_name,
        }


@dataclass(frozen=True)
class Decision:
    """One question a front end must put to the user."""

    id: str
    section: str
    question: str
    kind: DecisionKind
    why: str
    default: str | None = None
    choices: tuple[Choice, ...] = ()
    when: tuple[Condition, ...] = ()
    # Asked only if at least one of these holds as well as every `when`.
    # Conditions are otherwise all-of, and some questions are needed for one
    # reason or another: the spin of the QM region whichever metal turns out
    # bonded, a copy's ligand whether its chain was kept alone or its
    # parameters were not to be shared.
    when_any: tuple[Condition, ...] = ()
    requires_user_input: bool = False
    evidence: tuple[str, ...] = ()
    # targets[0] is the subject of the decision -- the ligand being charged, the
    # ion being modelled -- and any further entries are surrounding context the
    # viewer should also show. The manifest builders rely on that ordering.
    targets: tuple[ResidueTarget, ...] = ()
    # The same question about the counterpart of this residue in the copy
    # whose parameters this one shares. Skipped when parameters are shared,
    # and given that question's answer instead.
    copy_of: str | None = None
    # Residue names standing in for a set too large to enumerate usefully, such
    # as every crystallographic water. A viewer selects on these directly.
    target_resnames: tuple[str, ...] = ()
    focus: FocusMode = "none"
    minimum: int | float | None = None
    maximum: int | float | None = None
    # A question that only means something where the answer will be used.
    # A browser has no filesystem the user can usefully name, so asking for
    # an output directory there is a box to fill in for no reason: the page
    # hands back a bundle and whoever runs it chooses where it lands.
    needs_local_filesystem: bool = False

    @property
    def subject(self) -> "ResidueTarget":
        """The residue this decision is about."""

        if not self.targets:
            raise DecisionError(f"{self.id}: no residue recorded")
        return self.targets[0]

    def is_applicable(self, answers: Mapping[str, Any]) -> bool:
        return all(condition.is_met(answers) for condition in self.when) and (
            not self.when_any
            or any(condition.is_met(answers) for condition in self.when_any)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "section": self.section,
            "question": self.question,
            "kind": self.kind,
            "why": self.why,
            "default": self.default,
            "requires_user_input": self.requires_user_input,
            "choices": [choice.to_dict() for choice in self.choices],
            "when": [condition.to_dict() for condition in self.when],
            "when_any": [condition.to_dict() for condition in self.when_any],
            "copy_of": self.copy_of,
            "evidence": list(self.evidence),
            "targets": [target.to_dict() for target in self.targets],
            "target_resnames": list(self.target_resnames),
            "focus": self.focus,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "needs_local_filesystem": self.needs_local_filesystem,
        }


@dataclass(frozen=True)
class DecisionPlan:
    """Every decision implied by one input structure."""

    structure_path: Path
    decisions: tuple[Decision, ...]
    findings: dict[str, Any] = field(default_factory=dict)
    # Which chains are copies of one protein and how their residues pair.
    # Kept on the plan because the manifest built from it has to name every
    # copy's residues, and the front end holds the plan, not the structure.
    oligomer: OligomerAnalysis | None = None

    def by_id(self, decision_id: str) -> Decision:
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        raise DecisionError(f"Unknown decision id: {decision_id}")

    def applicable(self, answers: Mapping[str, Any]) -> list[Decision]:
        """Decisions still reachable given the answers collected so far.

        An answer only counts while its own question is reachable, the way a
        front end walks the plan: a confirmed coordination shell left over
        from when a site was bonded says nothing once it is nonbonded.
        """

        reachable: dict[str, Any] = {}
        found: list[Decision] = []
        for decision in self.decisions:
            if not decision.is_applicable(reachable):
                continue
            found.append(decision)
            if decision.id in answers:
                reachable[decision.id] = answers[decision.id]
        return found

    def blocking(self, answers: Mapping[str, Any] | None = None) -> list[Decision]:
        """Decisions with no safe default, which a user must answer.

        Without ``answers`` this lists every decision that could become
        blocking; with them, only those still reachable.
        """

        if answers is None:
            return [
                decision for decision in self.decisions if decision.requires_user_input
            ]
        return [
            decision
            for decision in self.applicable(answers)
            if decision.requires_user_input
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "structure_path": str(self.structure_path),
            "findings": self.findings,
            "decision_count": len(self.decisions),
            "blocking_count": len(self.blocking()),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def _as_answer_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _id_stem(residue: ResidueRecord) -> str:
    """Identifier parts for one residue, including its chain.

    The chain is required: a crystal structure with several copies of the same
    complex in the asymmetric unit repeats every residue number, and manifest
    ligand and metal-site ids have to stay unique across them.
    """

    return _id_stem_of(residue.id)


def _id_stem_of(residue_id: ResidueId) -> str:
    parts = [part for part in (residue_id.chain_id.strip().lower(),) if part]
    parts.append(residue_id.resname.lower())
    parts.append(str(residue_id.resid))
    if residue_id.icode:
        parts.append(residue_id.icode.strip().lower())
    return "_".join(parts)


def _residue_id_of(target: "ResidueTarget") -> ResidueId:
    return ResidueId(target.chain, target.resname, target.resid, target.icode)


def ligand_id_for(residue: ResidueRecord) -> str:
    """Stable manifest id for a detected ligand residue."""

    return _id_stem(residue)


def metal_site_id_for(residue: ResidueRecord) -> str:
    return f"{_id_stem(residue)}_site"


def _target_for(
    residue: ResidueRecord,
    *,
    element: str | None = None,
    atom_name: str | None = None,
) -> "ResidueTarget":
    if atom_name is None and len(residue.atoms) == 1:
        atom_name = residue.atoms[0].name
    return ResidueTarget(
        chain=residue.id.chain_id,
        resname=residue.id.resname,
        resid=residue.id.resid,
        icode=residue.id.icode,
        element=element,
        atom_name=atom_name,
    )


# Generous enough to catch first-shell metal-nitrogen and metal-oxygen
# distances. This frames the 3D view; it is NOT a determination of coordination,
# which mdprep requires the user to declare explicitly.
CONTACT_CUTOFF_ANGSTROM = 3.0

def contact_residues(
    residues: Sequence[ResidueRecord],
    centre: ResidueRecord,
    *,
    cutoff_angstrom: float = CONTACT_CUTOFF_ANGSTROM,
) -> list[ResidueRecord]:
    """Residues with any atom within ``cutoff_angstrom`` of ``centre``.

    Used to show a metal ion together with whatever surrounds it, so that the
    person answering "what is this site?" can see what they are answering about.
    """

    origin = centre.atoms[0]
    found: list[ResidueRecord] = []
    for residue in residues:
        if residue.id == centre.id:
            continue
        for atom in residue.atoms:
            dx = atom.x - origin.x
            dy = atom.y - origin.y
            dz = atom.z - origin.z
            if dx * dx + dy * dy + dz * dz <= cutoff_angstrom * cutoff_angstrom:
                found.append(residue)
                break
    return found


def _target_from_id(residue_id: Any) -> "ResidueTarget":
    return ResidueTarget(
        chain=residue_id.chain_id,
        resname=residue_id.resname,
        resid=residue_id.resid,
        icode=residue_id.icode,
    )


def _residue_element(residue: ResidueRecord) -> str:
    element = (residue.atoms[0].element or residue.id.resname).strip()
    return element.capitalize() if len(element) == 2 else element.upper()


def plan_decisions(
    structure_path: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
    disulfide_cutoff_angstrom: float = 2.2,
) -> DecisionPlan:
    """Inspect a structure and derive the decisions its manifest needs."""

    path = Path(structure_path)
    summary = inspect_pdb_structure(
        path,
        altloc_policy=altloc_policy,
        disulfide_cutoff_angstrom=disulfide_cutoff_angstrom,
    )
    # A metal ion is a heterogen, but it is parameterised as a metal site, not
    # as a GAFF ligand, so it must not also be offered for antechamber.
    # A metal ion is parameterised as a metal site, and a single-atom donor --
    # a coordinating halide, say -- belongs to the site it coordinates:
    # antechamber's semi-empirical step cannot run on one atom.
    monatomic_donors = {
        donor.id
        for metal in summary.metal_ions
        for donor, _, _ in metal_donor_atoms(summary.structure.residues, metal)
        if len(donor.atoms) == 1
    }
    # A ligand that coordinates a metal belongs to that site's quantum model.
    donor_residue_ids = {
        donor.id
        for metal in summary.metal_ions
        for donor, _, _ in metal_donor_atoms(summary.structure.residues, metal)
    }
    # Heterogens that are not ligands: a modified residue of a chain, a
    # glycan, anything else bonded to the protein, a crystallisation
    # additive. Each group is asked about as a group, not as ligands.
    groups = classify_heterogens(
        summary.structure.residues,
        getattr(summary.structure, "links", ()),
        getattr(summary.structure, "modified_residue_parents", {}),
        metal_donor_ids=frozenset(donor_residue_ids),
    )
    ligands = [
        residue
        for residue in summary.likely_ligands
        if not is_metal_ion_residue(residue)
        and residue.id not in monatomic_donors
        and residue.id not in groups.not_ligands
    ]
    # What mdprep cannot parameterise is refused before anything is asked,
    # with the reason, rather than failing in antechamber or tleap later --
    # or worse, building.
    unsupported = unsupported_ligands(ligands, summary.structure.residues)
    unsupported += unsupported_heterogens(
        groups,
        summary,
        donor_residue_ids,
        getattr(summary.structure, "assigned_resnames", frozenset()),
    )
    if unsupported:
        raise DecisionError(" ".join(unsupported))

    # A ligand carrying a name the force fields already define cannot be
    # built: tleap resolves residues by name, so it would use the protein,
    # water, ion or nucleic template rather than the ligand's own parameters.
    # mdprep used to rename it for the build, which only moved the problem --
    # the prepared system then described a residue under a name the input
    # never used, and every later reader had to know about the substitution.
    # Refuse here, before a single question is asked, so the fix happens once
    # in the input. The GUI shows this as the reason the file was rejected.
    reserved = [
        residue
        for residue in ligands
        if residue.id.resname in LEAP_RESERVED_RESIDUE_NAMES
    ]
    if reserved:
        taken = {residue.id.resname for residue in summary.structure.residues}
        suggested: dict[str, str] = {}
        for residue in reserved:
            # One replacement per original name, so the four copies of a
            # complex in an asymmetric unit all become the same new residue
            # rather than four unrelated ones.
            if residue.id.resname not in suggested:
                suggested[residue.id.resname] = free_residue_name(
                    taken | set(suggested.values())
                )
        raise ReservedResidueNameError(
            describe_reserved_residue_names(
                [(residue.id.display(), residue.id.resname) for residue in reserved]
            ),
            [
                {
                    "chain": residue.id.chain_id,
                    "resname": residue.id.resname,
                    "resid": residue.id.resid,
                    "icode": residue.id.icode,
                    "display": residue.id.display(),
                    "new_resname": suggested[residue.id.resname],
                }
                for residue in reserved
            ],
        )

    oligomer = analyse_oligomer(summary.structure.residues)

    decisions: list[Decision] = []
    decisions.extend(_project_decisions(path))
    decisions.extend(_chain_decision(summary))
    decisions.extend(_structure_decisions(summary, groups))
    decisions.extend(_repair_decisions(summary, groups))
    decisions.extend(_protein_decisions())
    decisions.extend(_protonation_decisions(summary))
    decisions.extend(_disulfide_decisions(summary))
    additive_ids = {residue.id for residue in groups.additives}
    for residue in ligands:
        decisions.extend(
            _ligand_decisions(
                residue,
                coordinates_metal=residue.id in donor_residue_ids,
                # An additive is only offered as a ligand once the user has
                # chosen to keep the additives.
                additive=residue.id in additive_ids,
            )
        )
    for residue in summary.metal_ions:
        decisions.extend(
            _metal_decisions(
                residue,
                summary.structure.residues,
                getattr(summary.structure, "assigned_resnames", frozenset()),
            )
        )
    decisions.extend(_qm_region_decisions(summary))
    decisions.extend(_solvation_decisions())
    decisions.extend(_md_decisions())

    # A ligand or metal in a copy the user did not keep must not be asked
    # about, and must not reach the manifest. The copy is the protein chain
    # it sits in or against, which is not its own chain letter when a
    # prediction writes every ligand as a chain of its own.
    selectable = set(protein_chain_ids(summary))
    if any(decision.id == "structure.chains" for decision in decisions):
        decisions = [
            replace(
                decision,
                when=decision.when + (Condition("structure.chains", ("all", chain)),),
            )
            if (
                decision.section in {"ligands", "metals"}
                and decision.targets
                and (chain := oligomer.chain_of_residue.get(_residue_id_of(decision.subject)))
                in selectable
            )
            else decision
            for decision in decisions
        ]

    decisions, shared = _share_between_copies(decisions, oligomer)
    if shared:
        after = next(
            index for index, decision in enumerate(decisions) if decision.id == "structure.chains"
        )
        decisions.insert(after + 1, _same_parametrization_decision(oligomer, shared))

    findings = {
        "total_atoms": len(summary.structure.atoms),
        "total_residues": len(summary.structure.residues),
        "protein_residues": len(summary.protein_residues),
        "water_residues": len(summary.water_residues),
        "chains": summary.chain_ids,
        "protein_chains": protein_chain_ids(summary),
        # Kept whichever protein chain is chosen, so they are never offered.
        "non_protein_chains": non_protein_chain_ids(summary),
        "ligands": [residue.id.display() for residue in ligands],
        # A ligand whose hydrogens the structure never resolved cannot be
        # parameterised until its protonation is stated. Recorded here so the
        # builders can say so without asking for a SMILES they cannot check.
        "ligands_without_hydrogens": [
            ligand_id_for(residue)
            for residue in ligands
            if not _ligand_has_hydrogens(residue)
        ],
        "metal_ions": [residue.id.display() for residue in summary.metal_ions],
        "crystal_ions": [residue.id.display() for residue in summary.crystal_ions],
        "coordinating_waters": [
            residue.id.display() for residue in coordinating_water_residues(summary)
        ],
        "histidines": [residue.id.display() for residue in summary.histidines],
        "possible_disulfides": [
            f"{candidate.a.display()} -- {candidate.b.display()} "
            f"({candidate.distance_angstrom:.2f} A)"
            for candidate in summary.possible_disulfides
        ],
        "incomplete_residues": [
            item.describe() for item in incomplete_residues(summary.structure.residues)
        ],
        "chain_breaks": [
            item.describe()
            for item in chain_breaks(
                summary.structure.residues,
                also_in_chain={item.residue.id for item in groups.modified_residues},
            )
        ],
        "modified_residues": [item.describe() for item in groups.modified_residues],
        # What replacing them means, residue by residue, for the manifest.
        "modified_residue_replacements": [
            {
                "chain": item.residue.id.chain_id,
                "resname": item.residue.id.resname,
                "resid": item.residue.id.resid,
                "icode": item.residue.id.icode,
                "parent": item.parent,
            }
            for item in groups.modified_residues
            if item.parent is not None
        ],
        "glycans": [item.residue.id.display() for item in groups.glycans],
        "covalent_heterogens": [item.residue.id.display() for item in groups.covalent],
        "additives": [residue.id.display() for residue in groups.additives],
        "structure_warnings": list(summary.structure.warnings),
        "oligomer": oligomer.to_dict(),
        # The atoms each ligand is written with, so a manifest can only let
        # two residues share one template when tleap would accept it.
        "ligand_atom_names": {
            ligand_id_for(residue): residue.atom_names() for residue in ligands
        },
        # Cofactors whose metal mdprep moved into a residue of its own and
        # whose metal-free form it knows, so their SMILES can be proposed.
        "known_cofactors": {
            ligand_id_for(residue): entry.resname
            for residue in ligands
            if (entry := known_metal_free_cofactor(residue)) is not None
        },
        # Organic cofactors -- a flavin, SAM, a nicotinamide -- whose SMILES
        # and charge their names fix, so they can be proposed too.
        "known_organic_cofactors": {
            ligand_id_for(residue): entry.resname
            for residue in ligands
            if (entry := known_organic_cofactor(residue)) is not None
        },
    }
    return DecisionPlan(
        structure_path=path,
        decisions=tuple(decisions),
        findings=findings,
        oligomer=oligomer,
    )


# Cofactors that are bonded to the protein, so that mdprep would build them
# as a separate molecule at bond length from it. Refused by name: a bond
# cannot be told from a clash by distance alone, and predicted complexes
# have clashes this short.
COVALENT_COFACTORS = {
    "HEC": "heme c, bonded to two cysteines through thioethers",
}


def unsupported_ligands(
    ligands: Sequence[ResidueRecord], residues: Sequence[ResidueRecord]
) -> list[str]:
    """Why each of these ligands cannot be prepared, if it cannot.

    A metal inside a cofactor has already been read out into a residue of its
    own, so a ligand that still holds metal atoms holds several: a cluster,
    such as an iron-sulfur cluster, which no single metal site describes.
    And a cofactor bonded to the protein would be built beside it rather than
    onto it, which is the one outcome worse than refusing.
    """

    problems: list[str] = []
    for ligand in ligands:
        metals = [
            atom for atom in ligand.atoms
            if (atom.element or "").strip().upper() in COORDINATING_METAL_ELEMENTS
        ]
        if len(metals) == 1:
            # The reader takes a lone metal out of its cofactor unless what
            # remains would fall apart: an oxoanion.
            problems.append(
                f"{ligand.id.display()} holds a {(metals[0].element or '').strip().capitalize()} "
                "bonded to atoms that are bonded to nothing else -- an oxoanion "
                "such as vanadate. It is neither a metal site mdprep can build "
                "around a one-atom metal nor a ligand antechamber can "
                "parameterise, so mdprep cannot prepare it."
            )
            continue
        if metals:
            elements = sorted({(atom.element or "").strip().capitalize() for atom in metals})
            problems.append(
                f"{ligand.id.display()} holds {len(metals)} metal atoms "
                f"({', '.join(elements)}): a metal cluster. mdprep models a metal "
                "as one atom in a site of its own and cannot parameterise a "
                "cluster."
            )
            continue
        covalent = COVALENT_COFACTORS.get(ligand.id.resname.strip().upper())
        if covalent is not None:
            problems.append(
                f"{ligand.id.display()} is {covalent}: a covalently attached "
                "ligand, which mdprep does not support."
            )
    return problems


def unsupported_heterogens(
    groups: HeterogenGroups,
    summary: InspectionSummary,
    donor_residue_ids: set[ResidueId],
    assigned_resnames: frozenset[str] = frozenset(),
) -> list[str]:
    """Why each non-ligand heterogen cannot be prepared, if it cannot.

    A modified residue with no standard parent cannot be replaced, and
    cannot be built as it is. A bonded cofactor that held a metal cannot be
    removed without the metal site it carries, nor built bonded. A bonded
    heterogen on a metal cannot be removed without changing the site. And a
    crosslink between two protein residues is a bond tleap never makes.
    """

    problems: list[str] = []
    for item in groups.modified_residues:
        if item.parent is None:
            problems.append(
                f"{item.residue.id.display()} is a residue of a protein chain "
                "that is not one of the twenty standard amino acids, and mdprep "
                "knows no standard one it is a modified form of: noncanonical "
                "amino acids inside a chain are not supported."
            )
    bonded = {item.residue.id: item for item in groups.glycans + groups.covalent}
    for metal in summary.metal_ions:
        cofactor = source_cofactor(metal, summary.structure.residues, assigned_resnames)
        if cofactor is not None and cofactor.id in bonded:
            problems.append(
                f"{cofactor.id.display()} is bonded to the protein "
                f"({'; '.join(bonded[cofactor.id].bonds)}) and holds the metal "
                f"{metal.id.display()}: a covalently attached cofactor, which "
                "mdprep does not support. It cannot be left out either without "
                "taking its metal site with it."
            )
            del bonded[cofactor.id]
    for residue_id, item in bonded.items():
        if residue_id in donor_residue_ids:
            problems.append(
                f"{residue_id.display()} is bonded to the protein "
                f"({'; '.join(item.bonds)}) and coordinates a metal: removing it "
                "would change the metal site, and keeping it needs a covalent "
                "ligand, which mdprep does not support."
            )
    for crosslink in groups.crosslinks:
        problems.append(
            f"The structure declares a bond between two protein residues, "
            f"{crosslink}, that is neither a peptide bond nor a disulfide. "
            "tleap would build the two residues unbonded, so mdprep does not "
            "prepare it."
        )
    return problems


def _project_decisions(path: Path) -> list[Decision]:
    name = path.stem
    return [
        Decision(
            id="project.name",
            section="project",
            question="Project name",
            kind="text",
            why="Labels the run and names the default output directory.",
            default=name,
        ),
        Decision(
            id="project.output_dir",
            section="project",
            question="Output directory",
            kind="text",
            why=(
                "Where prepared structures, logs and Amber files are written, "
                "on the machine that runs the preparation. A front end with no "
                "filesystem of its own leaves this at its default and lets "
                "whoever runs the manifest decide."
            ),
            default=f"prepared/{name}",
            needs_local_filesystem=True,
        ),
    ]


def protein_chain_ids(summary: InspectionSummary) -> list[str]:
    """Chains containing protein, in input order."""

    seen: list[str] = []
    for residue in summary.protein_residues:
        chain = residue.id.chain_id
        if chain and chain not in seen:
            seen.append(chain)
    return seen


def non_protein_chain_ids(summary: InspectionSummary) -> list[str]:
    """Chains holding only heterogens or water, which are never dropped."""

    protein = set(protein_chain_ids(summary))
    seen: list[str] = []
    for residue in summary.structure.residues:
        chain = residue.id.chain_id
        if chain and chain not in protein and chain not in seen:
            seen.append(chain)
    return seen


def _chain_decision(summary: InspectionSummary) -> list[Decision]:
    """Which chains to build, when the input holds more than one."""

    # Only chains carrying protein are copies of the complex. A chain holding
    # nothing but a metal or a ligand is part of whichever copy it belongs to
    # and goes wherever that copy goes.
    chains = protein_chain_ids(summary)
    if len(chains) < 2:
        return []
    # A crystal asymmetric unit is often several copies of a complex that is
    # a monomer in solution, and a prediction of a dimer is usually meant as
    # one. With a metal in it the difference is also a bonded site fitted
    # once or a system several times the size, so it is not guessed.
    has_metal = bool(summary.metal_ions)
    choices = [
        Choice(
            chain,
            f"Chain {chain} only",
            "One copy of the complex, with the ligands and metals that belong to it.",
        )
        for chain in chains
    ]
    choices.append(
        Choice(
            "all",
            "All chains",
            "The whole assembly in the file: a dimer, trimer, tetramer...",
        )
    )
    return [
        Decision(
            id="structure.chains",
            section="structure",
            question="Which chains should be prepared?",
            kind="choice",
            why=(
                "The input contains more than one protein chain. A predicted "
                "oligomer is usually the assembly you mean to simulate; a "
                "crystal structure's chains are often copies of a complex that "
                "is a monomer in solution, packed together by the crystal. "
                "Keep all of them for the assembly, or one for a single copy. "
                "Ligands and metals go with the chain they belong to."
            ),
            default=None if has_metal else "all",
            requires_user_input=has_metal,
            choices=tuple(choices),
            evidence=(
                f"chains present: {', '.join(chains)}",
                *(
                    (f"metal ions found: {len(summary.metal_ions)}",)
                    if has_metal
                    else ()
                ),
            ),
            focus="chains",
        )
    ]


def _share_between_copies(
    decisions: list[Decision], oligomer: OligomerAnalysis
) -> tuple[list[Decision], list[tuple[ResidueId, ResidueId]]]:
    """Mark each copy's questions as copies of the reference copy's.

    A ligand or metal in a copy with a counterpart in the reference copy has
    the same questions, about the same chemistry. Each of them is given
    ``copy_of`` -- the reference question it stands for -- and is asked only
    when the parameters are not to be shared, or when its own chain is the
    one being kept.

    A counterpart whose questions are not the same, one for one, is not a
    copy after all: a ligand resolved with hydrogens in one chain and without
    in the other is asked for a SMILES in one only. It is reported and asked
    about on its own.

    Returns the decisions and the (copy, reference) residues now shared.
    """

    if not oligomer.homomeric:
        return decisions, []
    ids = {decision.id for decision in decisions}
    items: dict[tuple[str, str], list[int]] = {}
    subjects: dict[tuple[str, str], ResidueId] = {}
    for index, decision in enumerate(decisions):
        kind = decision.id.split(".", 1)[0]
        if kind not in {"ligand", "metal"} or decision.id.count(".") < 2:
            continue
        key = (kind, decision.id.split(".")[1])
        items.setdefault(key, []).append(index)
        if key not in subjects and decision.targets:
            subjects[key] = _residue_id_of(decision.subject)

    shared: list[tuple[ResidueId, ResidueId]] = []
    for (kind, item), indices in items.items():
        subject = subjects.get((kind, item))
        if subject is None:
            continue
        chain = oligomer.chain_of_residue.get(subject)
        reference = oligomer.counterpart(subject)
        if chain is None or reference is None or oligomer.reference_chain(chain) == chain:
            continue
        reference_item = _id_stem_of(reference) + ("_site" if kind == "metal" else "")
        mapped: dict[int, str] = {}
        for index in indices:
            decision = decisions[index]
            field_name = decision.id.split(".", 2)[2]
            if field_name.startswith("donor_state."):
                donor = oligomer.counterpart(_residue_id_of(decision.subject))
                if donor is None:
                    break
                field_name = "donor_state." + _id_stem_of(donor)
            mapped[index] = f"{kind}.{reference_item}.{field_name}"
        reference_ids = {
            decision_id for decision_id in ids if decision_id.startswith(f"{kind}.{reference_item}.")
        }
        if len(mapped) != len(indices) or set(mapped.values()) != reference_ids:
            oligomer.unmatched[subject] = (
                f"its questions are not the same as {reference.display()}'s"
            )
            continue
        for index, reference_id in mapped.items():
            decisions[index] = replace(
                decisions[index],
                copy_of=reference_id,
                when_any=(
                    Condition("structure.chains", (chain,)),
                    Condition("structure.same_parametrization", ("false",)),
                ),
            )
        shared.append((subject, reference))
    return decisions, shared


def _same_parametrization_decision(
    oligomer: OligomerAnalysis, shared: list[tuple[ResidueId, ResidueId]]
) -> Decision:
    evidence: list[str] = []
    for group in oligomer.groups:
        if len(group) < 2:
            continue
        evidence.append(f"Chains {', '.join(group)} are copies of one protein.")
        for chain in group[1:]:
            alignment = oligomer.alignments[chain]
            evidence.append(
                f"Chain {chain}: {alignment.identity:.0%} identical to chain "
                f"{alignment.reference} over {len(alignment.pairs)} residues"
                + (", numbered differently" if alignment.renumbered else "")
            )
    for copy, reference in shared:
        evidence.append(f"{copy.display()} would share {reference.display()}'s parameters")
    for residue, reason in oligomer.unmatched.items():
        evidence.append(f"Asked about on its own either way: {residue.display()}, as {reason}")
    return Decision(
        id="structure.same_parametrization",
        section="structure",
        question="Give every copy of the protein the same parameters?",
        kind="boolean",
        why=(
            "These chains are copies of one protein. Answer yes and the "
            "questions that follow are asked for the first copy only; each "
            "answer is then given to the matching residue of every other "
            "copy. mdprep matches residues by sequence and by position, not "
            "by residue number, which often differs between copies. A ligand "
            "is parameterised once, and a bonded metal site is fitted once, "
            "with its templates and bonds then reused in each copy. Answer no "
            "to answer for every copy separately, as for copies in different "
            "states -- a substrate bound in one and not in another. Anything "
            "listed as asked on its own has no counterpart, and is asked "
            "about either way."
        ),
        default="true",
        when=(Condition("structure.chains", ("all",)),),
        evidence=tuple(evidence),
        targets=tuple(
            ResidueTarget(
                chain=residue.chain_id,
                resname=residue.resname,
                resid=residue.resid,
                icode=residue.icode,
            )
            for pair in shared
            for residue in pair
        ),
        # Both copies at once, each its own colour: the question is whether
        # they are alike, which a view of one of them cannot show.
        focus="chains",
    )


def coordinating_water_residues(summary: InspectionSummary) -> list[ResidueRecord]:
    """Waters in the first shell of a metal, which are part of its site.

    The same waters normalisation keeps with their metal whatever the answer
    about crystal waters, because both ask structure.coordination.
    """

    return metal_bound_waters(summary.structure.residues, summary.metal_ions)


def _listed(lines: Sequence[str], limit: int = 12) -> tuple[str, ...]:
    """Evidence lines, the first ``limit`` of them and a count of the rest."""

    shown = tuple(lines[:limit])
    if len(lines) > limit:
        shown += (f"... and {len(lines) - limit} more",)
    return shown


def _heterogen_group_decisions(groups: HeterogenGroups) -> list[Decision]:
    """What to do with each group of heterogens that are not ligands.

    Each is one question about the group, with every member named in its
    evidence and its targets: the manifest it produces lists each residue,
    so the targets must be all of them, not a sample.
    """

    decisions: list[Decision] = []
    if groups.modified_residues:
        labelling = all(
            item.residue.id.resname.strip().upper() in LABELLING_MODIFICATIONS
            for item in groups.modified_residues
        )
        rebuilt = [
            item.residue.id.display()
            for item in groups.modified_residues
            if item.side_chain_rebuilt
        ]
        decisions.append(
            Decision(
                id="structure.replace_modified_residues",
                section="structure",
                question=(
                    "Replace the modified residues in the protein chains by their "
                    "standard amino acids?"
                ),
                kind="boolean",
                # A selenomethionine is how a methionine was labelled to solve
                # the phases; anything else may be the chemistry of the
                # protein, and losing it is the user's call.
                default="true" if labelling else None,
                requires_user_input=not labelling,
                why=(
                    "No force field template bonds a modified amino acid into a "
                    "chain, so mdprep cannot build one as it is. Replacing it by "
                    "its standard parent keeps the backbone and the atoms the two "
                    "share and builds the rest -- the methionine a "
                    "selenomethionine stands for, but also a serine where a "
                    "phosphoserine was, whose phosphate is then gone from the "
                    "simulation. Answering no stops the build. "
                    + (
                        f"{', '.join(rebuilt)}: its side chain is bonded to its own "
                        "backbone, as a pyroglutamate's lactam is, and its parent's "
                        "is not. Replacing it builds the side chain again beyond CB, "
                        "in PDBFixer's template rotamer, which can sit close to its "
                        "neighbours until the system is minimised; and it frees the "
                        "nitrogen: a glutamine at the N-terminus is a charged amine "
                        "where the pyroglutamate was a neutral lactam. "
                        if rebuilt
                        else ""
                    )
                    + (
                        "Proposed yes: these are selenomethionines, the label a "
                        "methionine is given to solve the phases, not chemistry of "
                        "the protein."
                        if labelling
                        else "Nothing is proposed: whether the modification matters "
                        "to what you are simulating is yours to say."
                    )
                ),
                evidence=_listed(
                    [
                        f"{item.describe()}, from {item.parent_source}"
                        for item in groups.modified_residues
                    ]
                ),
                targets=tuple(_target_for(item.residue) for item in groups.modified_residues),
                focus="targets",
            )
        )
    if groups.glycans:
        decisions.append(
            Decision(
                id="structure.remove_glycans",
                section="structure",
                question="Remove the glycans bonded to the protein?",
                kind="boolean",
                # Keeping them cannot be built, so the only other answer stops.
                default="true",
                why=(
                    "These sugars are bonded to the protein, most through an "
                    "asparagine, serine or threonine. mdprep does not build "
                    "glycans -- their force field, GLYCAM, is not part of it -- and "
                    "a sugar built as a free molecule would sit at bond length from "
                    "the residue it was attached to. Removing them prepares the "
                    "protein as if deglycosylated, which is the usual choice when "
                    "they are far from the site you study: check that none is near "
                    "yours. Answering no stops the build."
                ),
                evidence=_listed(
                    [
                        item.residue.id.display()
                        + (f", bonded {item.bonds[0]}" if item.bonds else "")
                        for item in groups.glycans
                    ]
                ),
                targets=tuple(_target_for(item.residue) for item in groups.glycans),
                focus="targets",
            )
        )
    if groups.covalent:
        decisions.append(
            Decision(
                id="structure.remove_covalent_heterogens",
                section="structure",
                question="Remove the heterogens bonded to the protein?",
                kind="boolean",
                # A covalent intermediate may be the reason for the structure.
                default=None,
                requires_user_input=True,
                why=(
                    "The structure declares these bonded to the protein: a "
                    "covalent intermediate, an inhibitor, a cofactor attached to a "
                    "side chain. mdprep does not build covalent ligands, and one "
                    "built as a free molecule would sit at bond length from the "
                    "residue it is attached to. Removing it may remove what the "
                    "structure was solved to show, so nothing is proposed. "
                    "Answering no stops the build."
                ),
                evidence=_listed(
                    [
                        f"{item.residue.id.display()}: {'; '.join(item.bonds)}"
                        if item.bonds
                        else f"{item.residue.id.display()}: bonded to one of the above"
                        for item in groups.covalent
                    ]
                ),
                targets=tuple(_target_for(item.residue) for item in groups.covalent),
                focus="targets",
            )
        )
    if groups.additives:
        decisions.append(
            Decision(
                id="structure.remove_additives",
                section="structure",
                question="Remove the crystallisation additives?",
                kind="boolean",
                default="true",
                why=(
                    "These are components of the buffer, cryoprotectant or "
                    "precipitant the crystal was grown in. They are not part of "
                    "the protein in solution, and each one kept is a ligand that "
                    "needs a SMILES, a charge and parameters of its own. Some are "
                    "not artefacts -- a sulfate can sit where a substrate's "
                    "phosphate binds, an alcohol in a substrate tunnel -- so check "
                    "the list against your active site. Answer no to be asked about "
                    "each one as a ligand instead. A heterogen that coordinates a "
                    "metal is never on this list."
                ),
                evidence=_listed(
                    [
                        f"{residue.id.display()} "
                        f"({CRYSTALLISATION_ADDITIVES[residue.id.resname.strip().upper()]})"
                        for residue in groups.additives
                    ]
                ),
                targets=tuple(_target_for(residue) for residue in groups.additives),
                focus="targets",
            )
        )
    return decisions


def _structure_decisions(
    summary: InspectionSummary, groups: HeterogenGroups | None = None
) -> list[Decision]:
    groups = groups if groups is not None else HeterogenGroups()
    decisions: list[Decision] = []
    # A water bound to a metal is part of that site and is kept with it
    # whatever the answer here (normalisation sees to that, from the same
    # geometry), so the question is about every other crystal water. It
    # used to be skipped entirely as soon as any water bound any metal --
    # 6NIE's surface iron in chain B, with three waters on it, hid the
    # question for all 847 waters, even after chain A had been chosen.
    bound = coordinating_water_residues(summary)
    if summary.water_residues:
        evidence = [f"{len(summary.water_residues)} water residues in the input"]
        if bound:
            evidence.append(
                "Bound to a metal, and kept with it whatever you answer: "
                + ", ".join(residue.id.display() for residue in bound)
            )
        decisions.append(
            Decision(
                id="structure.keep_crystal_waters",
                section="structure",
                question="Keep the crystallographic waters?",
                kind="boolean",
                why=(
                    "Buried or bridging waters are often mechanistically important. "
                    "Discarding them is a chemistry decision, not a cleanup step. "
                    "A water bound to a metal is one of its ligands and is kept "
                    "with it either way. A bonded metal site needs them all: its QM/MM "
                    "refinement protonates the active-site waters."
                ),
                default="true",
                evidence=tuple(evidence),
                target_resnames=tuple(
                    sorted({residue.id.resname for residue in summary.water_residues})
                ),
                focus="targets",
            )
        )
    # Free monatomic ions: a halide in an anion pocket, a sodium on the
    # surface. tleap builds them from its ion library, so the only decision is
    # whether they stay. A halide bound to a metal is that site's business.
    site_ions = {
        donor.id
        for metal in summary.metal_ions
        for donor, _, _ in metal_donor_atoms(summary.structure.residues, metal)
    }
    free_ions = [residue for residue in summary.crystal_ions if residue.id not in site_ions]
    bound_ions = [residue for residue in summary.crystal_ions if residue.id in site_ions]
    if free_ions:
        evidence = [residue.id.display() for residue in free_ions]
        if bound_ions:
            # Named here because a nonbonded site does not claim it, and then
            # it goes with this answer like any other crystal ion.
            evidence.append(
                "Bound to a metal: "
                + ", ".join(residue.id.display() for residue in bound_ions)
                + ". A bonded site keeps it whatever you answer; with a "
                "nonbonded site it follows this answer."
            )
        decisions.append(
            Decision(
                id="structure.keep_crystal_ions",
                section="structure",
                question="Keep the crystallographic ions?",
                kind="boolean",
                why=(
                    "A chloride or bromide the crystal resolved often marks a real "
                    "anion-binding site, which in a halogenase may be the point. "
                    "They are built with Amber's standard ion parameters. Remove "
                    "them only if you know they are crystallisation additives."
                ),
                default="true",
                evidence=tuple(evidence),
                targets=tuple(_target_for(residue) for residue in free_ions),
                focus="targets",
            )
        )
    decisions.extend(_heterogen_group_decisions(groups))
    # A heterogen asked about in a group above is not an unknown one.
    unknown = [
        residue
        for residue in summary.heterogen_residues
        if not is_metal_ion_residue(residue)
        and residue not in summary.crystal_ions
        and residue.id not in groups.not_ligands
    ]
    if unknown:
        additive_ids = {residue.id for residue in groups.additives}
        others = [residue for residue in unknown if residue.id not in additive_ids]
        kept_additives = [residue for residue in unknown if residue.id in additive_ids]
        decisions.append(
            Decision(
                id="structure.remove_unknown_heterogens",
                section="structure",
                question=(
                    "Remove heterogens that are not configured as ligands or metal sites?"
                ),
                kind="boolean",
                # No default: mdprep must never drop an unknown heterogen without
                # the user having said so in the manifest.
                default=None,
                requires_user_input=True,
                why=(
                    "Every heterogen listed here is offered to you as a ligand in the "
                    "questions that follow. This decides what happens to one you then "
                    "decline: true removes it, false stops the build so nothing is "
                    "dropped without your say. If you will keep all of them, answer "
                    "false."
                ),
                evidence=tuple(residue.id.display() for residue in others)
                + (
                    (
                        "The crystallisation additives, now that you are keeping "
                        "them: " + ", ".join(residue.id.display() for residue in kept_additives),
                    )
                    if kept_additives
                    else ()
                ),
                targets=tuple(_target_for(residue) for residue in unknown),
                focus="targets",
                # Only additives, and they are going: nothing is left to decline.
                when=()
                if others
                else (Condition("structure.remove_additives", ("false",)),),
            )
        )
    return decisions


def _repair_decisions(
    summary: InspectionSummary, groups: HeterogenGroups | None = None
) -> list[Decision]:
    """Ask about incomplete residues before any expensive stage runs."""

    incomplete = incomplete_residues(summary.structure.residues)
    # A modified residue is part of its chain, not a gap in it.
    breaks = chain_breaks(
        summary.structure.residues,
        also_in_chain={
            item.residue.id for item in (groups.modified_residues if groups else ())
        },
    )
    if not incomplete and not breaks:
        return []

    shown = [item.describe() for item in incomplete[:8]]
    if len(incomplete) > len(shown):
        evidence = (
            f"{len(incomplete)} residues have heavy atoms missing, including:",
            *shown,
            f"... and {len(incomplete) - len(shown)} more",
        )
    else:
        evidence = (
            (f"{len(incomplete)} residues have heavy atoms missing:", *shown)
            if incomplete
            else ()
        )
    if breaks:
        evidence = evidence + (
            f"{len(breaks)} chain break(s), which pdbfixer does NOT close here:",
            *[item.describe() for item in breaks[:4]],
        )

    decisions = [
        Decision(
            id="structure.repair.add_missing_heavy_atoms",
            section="structure",
            question="Rebuild the side-chain atoms that are missing from the input?",
            kind="boolean",
            # No default: adding coordinates that were never observed is a
            # modelling decision, and so is simulating a truncated side chain.
            default=None,
            requires_user_input=True,
            why=(
                "tleap cannot build a residue whose heavy atoms are absent, so "
                "leaving these will stop the run. Rebuilding them with PDBFixer "
                "invents coordinates that no experiment measured; they are usually "
                "surface side chains where that is acceptable, but check the list "
                "for anything near your active site."
            ),
            evidence=evidence,
            targets=tuple(_target_for(item.residue) for item in incomplete[:25]),
            focus="targets" if incomplete else "structure",
        ),
        Decision(
            id="structure.repair.unresolved_residues",
            section="structure",
            question="What should happen to residues the structure does not resolve at all?",
            kind="choice",
            # Rebuilding a whole residue from nothing is a bigger claim than
            # completing a side chain, so it gets its own question rather than
            # riding on the one above.
            default="leave_absent",
            why=(
                "A deposition often lists residues it never resolved -- "
                "expression tags, disordered termini, a loop with no density. "
                "Building them places a whole residue where no experiment saw "
                "one. Leaving them out is usually right, and leaves a chain "
                "break that nothing closes."
            ),
            choices=(
                Choice(
                    "leave_absent",
                    "Leave them out",
                    "Prepare what the structure actually resolved.",
                ),
                Choice(
                    "fail",
                    "Stop and let me look",
                    "Refuse to continue until each one has been considered.",
                ),
            ),
            evidence=evidence,
            focus="structure",
        ),
    ]
    return decisions


def _protein_decisions() -> list[Decision]:
    return [
        Decision(
            id="protein.forcefield",
            section="protein",
            question="Protein force field",
            kind="choice",
            why="Sets the amino-acid parameters used by tleap.",
            default="ff14SB",
            focus="structure",
            choices=(
                Choice("ff14SB", "ff14SB", "Established general-purpose choice; pairs with TIP3P."),
                Choice("ff19SB", "ff19SB", "Newer backbone parameters; intended to be paired with OPC."),
            ),
        ),
        Decision(
            id="protein.water_model",
            section="protein",
            question="Water model",
            kind="choice",
            why="Must match the force field the protein parameters were fitted against.",
            default="TIP3P",
            focus="structure",
            choices=(
                Choice("TIP3P", "TIP3P", "The conventional partner for ff14SB."),
                Choice("OPC", "OPC", "The water model ff19SB was parameterised with."),
            ),
        ),
    ]


def _protonation_decisions(summary: InspectionSummary) -> list[Decision]:
    histidine_evidence = tuple(residue.id.display() for residue in summary.histidines)
    decisions = [
        Decision(
            id="protonation.method",
            section="protonation",
            question="How should residue protonation states be assigned?",
            kind="choice",
            why=(
                "mdprep never guesses a catalytic protonation state. Each method "
                "differs in how much it decides for you and what it needs "
                "installed. A neutral histidine is the case that matters: "
                "tleap will not accept one left as HIS, so a structure with "
                "histidines needs either an explicit state for every one of "
                "them or a method that decides them."
            ),
            # Manual only when there is nothing to decide. A structure with
            # histidines would otherwise generate a manifest that stops at
            # tleap for want of a state nobody was asked for.
            default="propka_xtb_his" if summary.histidines else "manual_only",
            choices=(
                Choice(
                    "manual_only",
                    "Manual only",
                    "You name every non-default state yourself, including every "
                    "neutral histidine. No external tools needed.",
                ),
                Choice(
                    "propka",
                    "PropKa",
                    "PropKa assigns pH-dependent states. Fails if a neutral histidine "
                    "stays ambiguous. Requires propka3.",
                ),
                Choice(
                    "propka_xtb_his",
                    "PropKa + xTB histidines",
                    "PropKa plus an xTB HID/HIE comparison for neutral histidines. "
                    "Requires propka3 and xtb.",
                ),
            ),
            evidence=(
                (f"{len(summary.histidines)} histidines: " + ", ".join(histidine_evidence),)
                if histidine_evidence
                else ("no histidines found",)
            ),
            targets=tuple(_target_for(residue) for residue in summary.histidines),
            focus="targets" if summary.histidines else "structure",
        ),
        Decision(
            id="protonation.ph",
            section="protonation",
            question="Target pH",
            kind="number",
            why="PropKa assigns states at this pH.",
            default="7.0",
            focus="structure",
            when=(Condition("protonation.method", ("propka", "propka_xtb_his")),),
        ),
    ]
    if summary.histidines:
        decisions.append(
            Decision(
                id="protonation.histidine.num_threads",
                section="protonation",
                question="OpenMP threads per xTB histidine calculation",
                kind="integer",
                why=(
                    "xTB otherwise opens one thread per hardware thread for every "
                    "tautomer cluster, which wastes large amounts of CPU time without "
                    "reducing wall time."
                ),
                default="1",
                minimum=1,
                when=(Condition("protonation.method", ("propka_xtb_his",)),),
            )
        )
    return decisions


def _disulfide_decisions(summary: InspectionSummary) -> list[Decision]:
    if not summary.possible_disulfides:
        return []
    evidence = tuple(
        f"{candidate.a.display()} -- {candidate.b.display()} "
        f"({candidate.distance_angstrom:.2f} A)"
        for candidate in summary.possible_disulfides
    )
    return [
        Decision(
            id="disulfides.auto_detect",
            section="disulfides",
            question="Assign CYX automatically for the close SG-SG pairs found?",
            kind="boolean",
            why=(
                "Answer false to list the pairs explicitly in the manifest instead, "
                "which is the safer option when a cysteine is catalytic."
            ),
            default="true",
            evidence=evidence,
            targets=tuple(
                target
                for candidate in summary.possible_disulfides
                for target in (
                    _target_from_id(candidate.a),
                    _target_from_id(candidate.b),
                )
            ),
            focus="targets",
        )
    ]


def _ligand_has_hydrogens(residue: ResidueRecord) -> bool:
    return any(
        (atom.element or "").strip().upper() == "H" for atom in residue.atoms
    )


def _ligand_decisions(
    residue: ResidueRecord, *, coordinates_metal: bool = False, additive: bool = False
) -> list[Decision]:
    ligand_id = ligand_id_for(residue)
    display = residue.id.display()
    evidence = (f"{display}, {len(residue.atoms)} atoms",)
    targets = (_target_for(residue),)
    include_id = f"ligand.{ligand_id}.include"
    decisions = [
        Decision(
            id=include_id,
            section="ligands",
            question=f"Parameterise {display} as a ligand?",
            kind="boolean",
            why=(
                "Answer false only if this residue is handled another way; an "
                "unconfigured heterogen will stop the build."
            ),
            default="true",
            evidence=evidence,
            targets=targets,
            focus="targets",
            # An additive is a ligand only once the additives are kept.
            when=(Condition("structure.remove_additives", ("false",)),) if additive else (),
        ),
        Decision(
            id=f"ligand.{ligand_id}.net_charge",
            section="ligands",
            question=f"Formal net charge of {display}",
            kind="integer",
            # No default: a wrong ligand charge silently produces a wrong system.
            default=None,
            requires_user_input=True,
            why=(
                "The total charge cannot be read from a PDB file. A wrong value "
                "produces a plausible-looking topology with the wrong electrostatics."
            ),
            evidence=evidence,
            targets=targets,
            focus="targets",
            when=(Condition(include_id, ("true",)),),
        ),
        Decision(
            id=f"ligand.{ligand_id}.smiles",
            section="ligands",
            question=f"SMILES for {display}, including its hydrogens",
            kind="text",
            # No default. A net charge does not determine a molecule: the keto
            # and enol forms of alpha-ketoglutarate are both C5H4O5(2-), and
            # differ in where two hydrogens sit and in two bond orders.
            default=None,
            requires_user_input=True,
            why=(
                "This structure resolved no hydrogens on this residue. The "
                "formal charge does not settle the chemistry -- a carboxylate "
                "and a carboxylic acid differ by one hydrogen, and two "
                "tautomers can share a formula and a charge -- so mdprep takes "
                "the bond orders and the protonation from this SMILES rather "
                "than guessing them. It must describe the same heavy atoms, in "
                "the same connectivity, as the residue in the file, and its "
                "formal charge must match the net charge above."
            ),
            evidence=evidence
            + ("No hydrogens are present on this residue in the input.",),
            targets=targets,
            focus="targets",
            when=(Condition(include_id, ("true",)),),
        ),
        Decision(
            id=f"ligand.{ligand_id}.atom_types",
            section="ligands",
            question=f"Atom types for {display}",
            kind="choice",
            why="GAFF2 is the current general Amber force field for organic molecules.",
            default="gaff2",
            choices=(
                Choice("gaff2", "GAFF2", "Current general Amber force field."),
                Choice("gaff", "GAFF", "Original version; use for consistency with older work."),
            ),
            when=(Condition(include_id, ("true",)),),
            targets=targets,
            focus="targets",
        ),
        Decision(
            id=f"ligand.{ligand_id}.charge_method",
            section="ligands",
            question=f"Partial-charge method for {display}",
            kind="choice",
            why=(
                (
                    "This residue coordinates a metal, so it is part of the metal "
                    "site's quantum model. Taking its charges from that same fit "
                    "keeps the site electrostatically consistent; fitting them "
                    "separately does not."
                )
                if coordinates_metal
                else (
                    "AM1-BCC is fast and standard. The RESP routes are more accurate "
                    "and much more expensive, and need extra QM settings in the "
                    "manifest."
                )
            ),
            default="mcpb_resp_pyscf" if coordinates_metal else "am1bcc",
            choices=(
                (
                    Choice(
                        "mcpb_resp_pyscf",
                        "From the metal site's MCPB fit",
                        "Charges come from the bonded site's large-model RESP.",
                    ),
                )
                if coordinates_metal
                else ()
            )
            + (
                Choice("am1bcc", "AM1-BCC", "Fast semi-empirical charges via antechamber."),
                Choice(
                    "gas_resp_pyscf",
                    "Gas-phase RESP (PySCF)",
                    "Adds a qmmesp block you must complete; see docs/ligands.md.",
                ),
                Choice(
                    "qmmesp_pyscf",
                    "QMMESP RESP (PySCF)",
                    "Environment-polarised ESP fit; adds a qmmesp block; see docs/qmmesp_pyscf.md.",
                ),
                Choice(
                    "user_mol2",
                    "Supply my own mol2",
                    "You provide charges in a mol2 file; also set user_mol2.",
                ),
            ),
            when=(Condition(include_id, ("true",)),),
            targets=targets,
            focus="targets",
        ),
    ]
    if _ligand_has_hydrogens(residue):
        # Already protonated, so mdprep leaves it alone and there is
        # nothing to ask. Adding hydrogens to a residue that has them is
        # the failure the question exists to prevent.
        decisions = [
            decision
            for decision in decisions
            if decision.id != f"ligand.{ligand_id}.smiles"
        ]
    return decisions


# Formal charges of the donor groups that appear in a first-shell metal site at
# neutral pH. Used only to show a user the arithmetic behind the capped-model
# charge they must enter; it is never used as a default.
# Oxidation states a residue name settles on its own. The PDB chemical
# component dictionary and Amber's ion libraries both name an ion after its
# state where there is more than one -- FE2 is Fe(II) and FE is Fe(III), CU1
# is Cu(I) and CU is Cu(II) -- so the name is evidence when it is one of
# these, and says nothing when a file calls the residue LIG or UNK.
OXIDATION_STATE_BY_RESNAME = {
    "ZN": 2, "MG": 2, "CA": 2, "MN": 2, "NI": 2, "CO": 2, "CD": 2, "HG": 2,
    "NA": 1, "K": 1, "LI": 1, "RB": 1, "CS": 1,
    "FE": 3, "FE2": 2, "FE3": 3,
    "CU": 2, "CU1": 1, "CU2": 2, "CU3": 3,
    "AL": 3, "CR": 3, "GA": 3, "IN": 3, "LA": 3, "PB": 2, "SR": 2, "BA": 2,
}

# Metals with one biological oxidation state worth assuming when the residue
# name gives nothing -- a prediction naming every heterogen LIG, for instance.
# Iron, copper, manganese and cobalt are deliberately absent: each has more
# than one common state, they differ in spin as well as charge, and a wrong
# guess yields a converged calculation with wrong parameters.
OXIDATION_STATE_BY_ELEMENT = {
    "ZN": 2, "MG": 2, "CA": 2, "NA": 1, "K": 1, "LI": 1,
}


def proposed_oxidation_state(
    resname: str,
    element: str | None,
    *,
    name_was_assigned: bool = False,
) -> tuple[int | None, str]:
    """An oxidation state the evidence settles, and why -- or no proposal.

    ``name_was_assigned`` marks a residue name mdprep chose itself, because
    the input's would not fit a PDB column. Those carry no chemistry. A
    prediction calls its iron LIG1, mdprep renames it FE so the rest of the
    toolchain can recognise a metal, and FE in the chemical component
    dictionary means Fe(III) -- so reading the state back out of the name
    would propose +3, with a citation, for an iron whose input said nothing
    at all about its oxidation state. The element table deliberately omits
    Fe, Cu, Mn and Co for exactly this reason; coming at it through an
    invented name must not be a way around that.
    """

    name = (resname or "").strip().upper()
    if name in OXIDATION_STATE_BY_RESNAME and not name_was_assigned:
        return (
            OXIDATION_STATE_BY_RESNAME[name],
            f"the residue name {name} is that oxidation state in the PDB "
            "chemical component dictionary",
        )
    symbol = (element or "").strip().upper()
    if symbol in OXIDATION_STATE_BY_ELEMENT:
        return (
            OXIDATION_STATE_BY_ELEMENT[symbol],
            f"{symbol.capitalize()} has one common biological oxidation state",
        )
    if name_was_assigned and name in OXIDATION_STATE_BY_RESNAME:
        return (
            None,
            f"the name {name} was assigned by mdprep because the input's did "
            "not fit a PDB residue field, so it says nothing about the "
            f"oxidation state, and {symbol.capitalize() or 'this element'} "
            "has more than one common state",
        )
    return (
        None,
        f"the residue name {name or 'given'} does not identify one, and "
        f"{symbol.capitalize() or 'this element'} has more than one common "
        "state, which differ in spin as well as charge",
    )


# Closed-shell metals bound by electrostatics, which an ion model describes.
ELECTROSTATIC_METALS = frozenset({"MG", "CA"})

METAL_NAMES = {
    "FE": "iron", "ZN": "zinc", "CU": "copper", "MN": "manganese",
    "CO": "cobalt", "NI": "nickel", "MG": "magnesium", "CA": "calcium",
    "MO": "molybdenum", "W": "tungsten", "V": "vanadium", "CD": "cadmium",
    "HG": "mercury",
}


STANDARD_DONOR_FORMAL_CHARGE = {
    "HIS": 0, "HID": 0, "HIE": 0, "HIP": 1,
    "ASP": -1, "ASH": 0, "GLU": -1, "GLH": 0,
    "CYS": 0, "CYM": -1, "SER": 0, "THR": 0, "TYR": 0,
    "ASN": 0, "GLN": 0, "MET": 0, "LYS": 1, "ARG": 1,
    "HOH": 0, "WAT": 0,
    "CL": -1, "BR": -1, "F": -1, "I": -1,
}


def source_cofactor(
    residue: ResidueRecord,
    all_residues: Sequence[ResidueRecord],
    assigned_resnames: frozenset[str] = frozenset(),
) -> ResidueRecord | None:
    """The cofactor a metal was read out of, if it was.

    A metal mdprep took out of a cofactor as it read the structure -- a
    heme's iron -- keeps the cofactor's chain and number, and its name is one
    mdprep assigned. That is how the cofactor is found again.
    """

    if residue.id.resname.strip().upper() not in {
        name.strip().upper() for name in assigned_resnames
    }:
        return None
    return next(
        (
            other
            for other in all_residues
            if other is not residue
            and len(other.atoms) > 1
            and (other.id.chain_id, other.id.resid, other.id.icode)
            == (residue.id.chain_id, residue.id.resid, residue.id.icode)
        ),
        None,
    )


def _metal_decisions(
    residue: ResidueRecord,
    all_residues: Sequence[ResidueRecord],
    assigned_resnames: frozenset[str] = frozenset(),
) -> list[Decision]:
    site_id = metal_site_id_for(residue)
    display = residue.id.display()
    element = _residue_element(residue)
    # A name mdprep chose cannot also be evidence for what mdprep should
    # infer. See proposed_oxidation_state.
    name_assigned = residue.id.resname.strip().upper() in {
        name.strip().upper() for name in assigned_resnames
    }
    cofactor = source_cofactor(residue, all_residues, assigned_resnames)
    donors = metal_donor_atoms(all_residues, residue)
    donor_residues: list[ResidueRecord] = []
    for donor_residue, _, _ in donors:
        if all(donor_residue.id != seen.id for seen in donor_residues):
            donor_residues.append(donor_residue)
    neighbours = contact_residues(all_residues, residue)
    contacts = tuple(_target_for(neighbour) for neighbour in neighbours)
    donor_evidence = tuple(
        f"{donor_residue.id.display()} {atom.name} at {distance:.2f} A"
        for donor_residue, atom, distance in donors
    )
    evidence = (
        (
            f"{display} is the {METAL_NAMES.get(element.upper(), element.capitalize())} "
            f"of {cofactor.id.display()}, which mdprep moved into a residue of its "
            f"own; the rest of {cofactor.id.resname} is a ligand that coordinates it"
        )
        if cofactor is not None
        else f"{display}, element {element}",
        (
            "within "
            f"{CONTACT_CUTOFF_ANGSTROM:.1f} A: "
            + ", ".join(neighbour.id.display() for neighbour in neighbours)
        )
        if neighbours
        else f"nothing within {CONTACT_CUTOFF_ANGSTROM:.1f} A",
    )
    # A predicted model has no waters at all, so a water the crystal has on
    # this metal is simply absent -- and the shell then looks complete with
    # one ligand missing. Say so where the shell is decided.
    if donors and not any(is_water_residue(other) for other in all_residues):
        evidence += (
            "This structure has no waters at all, as a predicted model has none. "
            "If the closest crystal structure has a water on this metal, bring it in "
            "before going on: mdprep transplant-waters (docs/predicted_structures.md).",
        )
    targets = (_target_for(residue, element=element),) + contacts
    proposed_state, state_reason = proposed_oxidation_state(
        residue.id.resname, element, name_was_assigned=name_assigned
    )
    if cofactor is not None and proposed_state is None:
        state_reason = (
            f"{display} came out of {cofactor.id.display()}, and a cofactor's "
            "name does not record the oxidation state of its metal"
            + (
                " -- a heme's iron is Fe(III) in a resting cytochrome P450 and "
                "Fe(II) in deoxymyoglobin"
                if element.upper() == "FE"
                else ""
            )
        )
    model_id = f"metal.{site_id}.model"
    bonded = (Condition(model_id, ("bonded_mcpb",)),)
    nonbonded = (Condition(model_id, ("nonbonded",)),)
    kept = (Condition(model_id, ("bonded_mcpb", "nonbonded")),)
    # Magnesium and calcium are closed-shell ions held by electrostatics:
    # an ion model describes them, and a quantum calculation adds cost and
    # nothing else. Every other metal with a shell is proposed bonded.
    electrostatic = element.upper() in ELECTROSTATIC_METALS
    if not donors:
        proposed_model = "nonbonded"
        model_reason = (
            "No donor atom was found within bonding distance, so there is no "
            "coordination sphere to hold together."
        )
    elif electrostatic:
        proposed_model = "nonbonded"
        model_reason = (
            f"{len(donors)} donor atom(s) were found within bonding distance, but "
            f"{METAL_NAMES.get(element.upper(), element.capitalize())} is a closed-shell "
            "ion held by electrostatics, which an ion model describes; the "
            "nonbonded model is proposed."
        )
    else:
        proposed_model = "bonded_mcpb"
        model_reason = (
            f"{len(donors)} donor atom(s) were found within bonding distance, which "
            "is why the bonded model is proposed here."
        )
    # A metal read out of a cofactor is that cofactor's; one that is free
    # may be an artefact -- a nickel from the purification tag -- and can be
    # left out. Its cofactor's metal cannot, without the cofactor.
    removable = cofactor is None
    confirm_id = f"metal.{site_id}.confirm_donors"
    confirmed = bonded + (Condition(confirm_id, ("true",)),)

    decisions = [
        Decision(
            id=model_id,
            section="metals",
            question=f"How should the metal site at {display} be modelled?",
            kind="choice",
            # An ion with a coordination shell is almost always there to do
            # something, and a nonbonded model lets that shell fall apart
            # during dynamics. One with no donors in range has nothing to hold
            # together, so it gets the cheap model.
            #
            # Accepting this does not commit anyone silently: a bonded site
            # then has to confirm its donors, its spin and its model charge,
            # none of which can be defaulted, so a metal that should have been
            # nonbonded is three questions away from being noticed.
            default=proposed_model,
            why=(
                "A nonbonded model keeps the ion free and suits a structural or "
                "loosely bound metal. A bonded MCPB model holds the coordination "
                "sphere together and is what a catalytic centre needs; it costs a "
                "quantum chemistry calculation. "
                + model_reason
                + (
                    " A metal that is not part of the system -- a nickel from a "
                    "His-tag purification, a cadmium from the crystallisation "
                    "buffer -- can be removed instead."
                    if removable
                    else ""
                )
            ),
            choices=(
                Choice(
                    "bonded_mcpb",
                    "Bonded (MCPB.py)",
                    "Keeps the coordination sphere; needs a Hessian and an ESP fit.",
                ),
                Choice(
                    "nonbonded",
                    "Nonbonded (Li/Merz)",
                    "Cheap. Pick an Amber ion parameter set below.",
                ),
            )
            + (
                (
                    Choice(
                        "remove",
                        "Remove it",
                        "Leave this metal out of the build: an artefact of "
                        "purification or crystallisation, not part of the system.",
                    ),
                )
                if removable
                else ()
            ),
            evidence=evidence + donor_evidence,
            targets=targets,
            focus="targets",
        ),
        # Needed by both models, and by neither once the metal is removed.
        Decision(
            id=f"metal.{site_id}.charge",
            section="metals",
            question=f"Formal oxidation state (charge) of {display}",
            kind="integer",
            default=str(proposed_state) if proposed_state is not None else None,
            requires_user_input=proposed_state is None,
            why=(
                "This sets the electron count, and with the spin state it sets "
                "what the quantum calculation is solving. "
                + (
                    f"Proposed as {proposed_state:+d} because {state_reason}; "
                    "change it if that is not this site."
                    if proposed_state is not None
                    else f"No value is proposed here because {state_reason}. "
                    "Guessing would give a converged calculation with the wrong "
                    "parameters."
                )
            ),
            minimum=1,
            maximum=4,
            evidence=evidence,
            targets=targets,
            focus="targets",
            when=kept,
        ),
        Decision(
            id=f"metal.{site_id}.parameter_set",
            section="metals",
            question=f"Amber ion parameter set for {display}",
            kind="choice",
            why="Li/Merz sets are fitted to reproduce different target properties.",
            default="12_6",
            choices=(
                Choice("12_6", "12-6", "Standard Lennard-Jones ion model."),
                Choice("12_6_4", "12-6-4", "Adds ion-induced polarisation; needs a compatible engine."),
                Choice("hfe", "HFE", "Fitted to hydration free energies."),
                Choice("iod", "IOD", "Fitted to ion-oxygen distances."),
                Choice("cm", "CM", "Compromise set."),
            ),
            when=nonbonded,
            targets=targets,
            focus="targets",
        ),
    ]

    if not donors:
        # Nothing to bond to, so the bonded model cannot be built here.
        return decisions

    decisions.append(
        Decision(
            id=confirm_id,
            section="metals",
            question=(
                f"Are these the atoms coordinating {display}?"
            ),
            kind="boolean",
            default=None,
            requires_user_input=True,
            why=(
                "mdprep proposes this shell from interatomic distances alone. "
                "Every bond it contains becomes a bonded parameter, so it has "
                "to be confirmed by someone who knows the chemistry. Answer no "
                "to write the site out by hand instead."
            ),
            evidence=donor_evidence,
            # targets[0] is the metal; the rest are the donor ATOMS, named,
            # because they become the bonds in the generated manifest.
            targets=(_target_for(residue, element=element),)
            + tuple(
                _target_for(donor_residue, atom_name=atom.name)
                for donor_residue, atom, _ in donors
            ),
            focus="targets",
            when=bonded,
        )
    )
    # Each titratable side chain on the metal next, before the charge of the
    # model it sits in is asked: its state is one of that charge's terms, and
    # a cysteine on a heme is CYM, not the neutral CYS its name says.
    decisions.extend(
        _metal_donor_protonation_decisions(
            site_id,
            # The residues that become bonds, which is what the build
            # refuses over -- not the shorter donor list. The two use
            # different radii (3.0 A against 2.6 A), so asking about donors
            # left a side chain between them bonded and unasked about,
            # which is the gap that sent the refusal to Colab.
            neighbours,
            distances={
                donor_residue.id.display(): (atom.name, distance)
                for donor_residue, atom, distance in donors
            },
            confirmed=confirmed,
            metal_display=display,
        )
    )
    decisions.extend(
        [
            Decision(
                id=f"metal.{site_id}.model_charge",
                section="metals",
                question=(
                    f"Total formal charge of the capped QM model around {display}"
                ),
                kind="integer",
                default=None,
                requires_user_input=True,
                why=(
                    "This is the charge of the metal plus every residue in the "
                    "coordination sphere, and it sets the electron count of the "
                    "quantum calculation. A wrong value gives converged, wrong "
                    "parameters. mdprep proposes the sum of the charges already "
                    "given -- the metal's oxidation state, each cofactor's net "
                    "charge, a halide's ionic charge, a protein donor's formal "
                    "charge -- and shows the arithmetic. Change it if the site "
                    "is not what those terms describe."
                ),
                evidence=donor_evidence,
                targets=targets,
                focus="targets",
                when=confirmed,
            ),
            Decision(
                id=f"metal.{site_id}.hessian_backend",
                section="metals",
                question=f"Which method should compute the Hessian for {display}?",
                kind="choice",
                why=(
                    "The Seminario method turns a Hessian into bonded "
                    "parameters, and a tight-binding one costs seconds where "
                    "DFT costs minutes to hours. GFN2 is parameterised for "
                    "main-group chemistry and gets transition metals wrong: on "
                    "an Fe(II) site it can give a metal-ligand bond a negative "
                    "force constant, which mdprep refuses to write out."
                ),
                default="gxtb",
                choices=(
                    Choice(
                        "gxtb",
                        "g-xTB",
                        "Tight-binding, handles transition metals. Needs the "
                        "official g-xTB build, which is separate from the xtb "
                        "conda installs.",
                    ),
                    Choice(
                        "gfn2",
                        "GFN2-xTB",
                        "Runs on the ordinary xtb with nothing to download, but "
                        "may give a force constant mdprep rejects. Check them.",
                    ),
                    Choice(
                        "pyscf",
                        "DFT (PySCF)",
                        "A reference rather than a default: minutes to hours "
                        "for this one quantity.",
                    ),
                ),
                evidence=evidence,
                targets=targets,
                focus="targets",
                when=confirmed,
            ),
        ]
    )
    return decisions


# Which states a coordinating side chain can sensibly be in. Histidine is
# absent on purpose: a bond through ND1 forces HIE and one through NE2 forces
# HID, so the geometry settles it and nothing needs asking.
COORDINATING_DONOR_STATES: dict[str, tuple[tuple[str, str], ...]] = {
    "ASP": (("ASP", "Ionised carboxylate (-1)"), ("ASH", "Neutral, protonated (0)")),
    "GLU": (("GLU", "Ionised carboxylate (-1)"), ("GLH", "Neutral, protonated (0)")),
    "CYS": (
        ("CYM", "Thiolate (-1)"),
        ("CYS", "Neutral thiol (0)"),
        ("CYX", "Disulfide-bonded"),
    ),
    "LYS": (("LYN", "Neutral amine (0)"), ("LYS", "Protonated (+1)")),
    "ARG": (("ARG", "Protonated guanidinium (+1)"),),
}


def _metal_donor_protonation_decisions(
    site_id: str,
    coordinators: Sequence[ResidueRecord],
    *,
    distances: Mapping[str, tuple[str, float]],
    confirmed: tuple[Condition, ...],
    metal_display: str,
) -> list[Decision]:
    """Ask the protonation of each titratable side chain on the metal.

    Preparation refuses to build a bonded site whose ASP, GLU, CYS, LYS or
    ARG donor has no stated protonation, because coordination does not
    distinguish a carboxylate from a neutral acid and the two differ by a
    unit of charge on the model the parameters come from. That refusal is
    right, and nothing asked the question -- so the first a student heard of
    it was the run stopping in Colab, after the install and most of the
    preparation. The answer belongs in the manifest, which means it belongs
    in the plan.
    """

    decisions: list[Decision] = []
    seen: set[str] = set()
    for donor_residue in coordinators:
        resname = donor_residue.id.resname.strip().upper()
        states = COORDINATING_DONOR_STATES.get(resname)
        if states is None:
            continue
        donor_id = _id_stem(donor_residue)
        if donor_id in seen:
            continue
        seen.add(donor_id)
        atom_name, distance = distances.get(donor_residue.id.display(), ("", 0.0))
        decisions.append(
            Decision(
                id=f"metal.{site_id}.donor_state.{donor_id}",
                section="protonation",
                question=(
                    f"Protonation state of {donor_residue.id.display()}, "
                    f"which coordinates {metal_display}"
                ),
                kind="choice",
                choices=tuple(
                    Choice(value, label) for value, label in states
                ),
                # No default. The deposited geometry shows the contact, not
                # the proton, and the two states differ by a unit of charge
                # on the QM model the force constants come from.
                default=None,
                requires_user_input=True,
                why=(
                    "A carboxylate and a neutral acid both sit this close to a "
                    "metal, and they differ by one unit of charge on the capped "
                    "model, so this changes the parameters rather than just the "
                    "bookkeeping. mdprep will not guess it, and the build stops "
                    "without it."
                ),
                evidence=(
                    (
                        f"{donor_residue.id.display()} {atom_name} at "
                        f"{distance:.2f} A from {metal_display}"
                    )
                    if atom_name
                    else (
                        f"{donor_residue.id.display()} is bonded to "
                        f"{metal_display} in the confirmed shell"
                    ),
                ),
                targets=(_target_for(donor_residue),),
                focus="targets",
                when=confirmed,
            )
        )
    return decisions


def _qm_region_decisions(summary: InspectionSummary) -> list[Decision]:
    """The spin state of the QM region: the one spin question there is.

    One bonded site is fitted per preparation, and the QM region is that
    site and the cofactors in it, so its spin is the site's: the MCPB.py
    models the force constants and charges come from, and the QM/MM
    refinement, all take this one number. Asking the metal's spin as well
    asked the same thing twice, and the region's could not be inferred from
    the metal's anyway -- put an Fe(III) and a substrate radical in one
    region and the total is 5 or 7 depending on how the two spins align, a
    chemical fact no arithmetic over the parts can settle. So it is asked,
    once, with what usually decides it.
    """

    sites = list(summary.metal_ions)
    if not sites:
        return []
    # Refinement only runs for a site that is modelled as bonded MCPB and
    # whose coordination shell was confirmed -- a rejected shell is written
    # out by hand and generates nothing. A confirmed shell is only asked
    # about once the site is bonded, so that one answer is the condition,
    # for whichever site it turns out to be. A metal with no donor in range
    # has no shell to confirm and can never be bonded here.
    when_any = tuple(
        Condition(f"metal.{metal_site_id_for(site)}.confirm_donors", ("true",))
        for site in sites
        if metal_donor_atoms(summary.structure.residues, site)
    )
    if not when_any:
        return []
    hemes = [
        site
        for site in sites
        if (cofactor := source_cofactor(
            site,
            summary.structure.residues,
            getattr(summary.structure, "assigned_resnames", frozenset()),
        )) is not None
        and cofactor.id.resname.strip().upper() == "HEM"
    ]
    return [
        Decision(
            id="refinement.total_qm_multiplicity",
            section="refinement",
            question=(
                "Spin multiplicity of the QM region: the metal, everything "
                "bonded to it, and the cofactors"
            ),
            kind="integer",
            default=None,
            requires_user_input=True,
            why=(
                "This one number is the spin of the quantum models the metal "
                "site's force constants and charges come from, and of the "
                "QM/MM refinement. With the metal as the only unpaired spin it "
                "is the metal's: high-spin Fe(II), d6, is 5; high-spin "
                "Fe(III), d5, is 6; Zn(II) and other closed-shell ions are 1; "
                "low-spin states are lower."
                + (
                    " In a heme the ligands across from each other decide it: "
                    "with a water opposite the cysteine, as in a resting P450, "
                    "Fe(III) is low-spin, 2; five-coordinate, with the water "
                    "gone, it is high-spin, 6."
                    if hemes
                    else ""
                )
                + " A second open-shell component, a radical substrate say, "
                "can couple to the metal either way, and then the answer is "
                "chemistry, not arithmetic."
            ),
            minimum=1,
            when_any=when_any,
        )
    ]


def _solvation_decisions() -> list[Decision]:
    return [
        Decision(
            id="solvation.enabled",
            section="solvation",
            question="Solvate and neutralise the system?",
            kind="boolean",
            why="Answer false only for a gas-phase or pre-solvated build.",
            default="true",
            focus="structure",
        ),
        Decision(
            id="solvation.box",
            section="solvation",
            question="Box shape",
            kind="choice",
            why="A truncated octahedron needs fewer waters than a box for the same padding.",
            default="truncated_octahedron",
            focus="structure",
            choices=(
                Choice("truncated_octahedron", "Truncated octahedron", "Fewer waters for the same buffer."),
                Choice("rectangular", "Rectangular", "Use for strongly elongated solutes."),
            ),
            when=(Condition("solvation.enabled", ("true",)),),
        ),
        Decision(
            id="solvation.buffer_angstrom",
            section="solvation",
            question="Solvent buffer (A)",
            kind="number",
            why="Distance from the solute to the box edge. Below about 10 A, periodic images interact.",
            default="10.0",
            focus="structure",
            minimum=0,
            when=(Condition("solvation.enabled", ("true",)),),
        ),
        Decision(
            id="solvation.salt_concentration_molar",
            section="solvation",
            question="Salt concentration (M)",
            kind="number",
            why="0.15 M approximates physiological ionic strength.",
            default="0.15",
            focus="structure",
            minimum=0,
            when=(Condition("solvation.enabled", ("true",)),),
        ),
    ]


def _md_decisions() -> list[Decision]:
    return [
        Decision(
            id="molecular_dynamics.enabled",
            section="molecular_dynamics",
            question="Run the Roe--Brooks equilibration and production MD after the build?",
            kind="boolean",
            why="Requires OpenMM. The build itself does not need it.",
            default="false",
        ),
        Decision(
            id="molecular_dynamics.production.steps",
            section="molecular_dynamics",
            question="Production steps (2 fs each; 500,000 steps = 1 ns)",
            kind="integer",
            why="Mandatory when MD is enabled; there is no sensible default length.",
            default=None,
            requires_user_input=True,
            minimum=1,
            when=(Condition("molecular_dynamics.enabled", ("true",)),),
        ),
    ]


# --------------------------------------------------------------------------
# Answers -> manifest
# --------------------------------------------------------------------------


def coerce_answer(decision: Decision, raw: Any) -> Any:
    """Convert a front end's raw answer into the manifest value type."""

    if decision.kind == "boolean":
        if isinstance(raw, bool):
            return raw
        token = str(raw).strip().lower()
        if token in {"true", "yes", "y", "1"}:
            return True
        if token in {"false", "no", "n", "0"}:
            return False
        raise DecisionError(f"{decision.id}: expected a yes/no answer, got {raw!r}")
    if decision.kind == "integer":
        try:
            value = int(str(raw).strip())
        except ValueError as exc:
            raise DecisionError(f"{decision.id}: expected a whole number, got {raw!r}") from exc
        _check_bounds(decision, value)
        return value
    if decision.kind == "number":
        try:
            value = float(str(raw).strip())
        except ValueError as exc:
            raise DecisionError(f"{decision.id}: expected a number, got {raw!r}") from exc
        _check_bounds(decision, value)
        return value
    token = str(raw).strip()
    if decision.kind == "choice":
        allowed = {choice.value for choice in decision.choices}
        if token not in allowed:
            raise DecisionError(
                f"{decision.id}: {token!r} is not one of {sorted(allowed)}"
            )
    if not token:
        raise DecisionError(f"{decision.id}: a value is required")
    return token


def _check_bounds(decision: Decision, value: int | float) -> None:
    if decision.minimum is not None and value < decision.minimum:
        raise DecisionError(f"{decision.id}: must be at least {decision.minimum}")
    if decision.maximum is not None and value > decision.maximum:
        raise DecisionError(f"{decision.id}: must be at most {decision.maximum}")


def resolve_answers(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> dict[str, Any]:
    """Fill in defaults, drop unreachable decisions, and validate the result.

    Raises if a decision that must be answered by a human was left out: that is
    the guarantee that makes the generated manifest trustworthy.
    """

    resolved: dict[str, Any] = {}
    missing: list[str] = []
    for decision in plan.decisions:
        if not decision.is_applicable(resolved):
            continue
        if decision.id in answers and answers[decision.id] is not None:
            resolved[decision.id] = coerce_answer(decision, answers[decision.id])
            continue
        if decision.requires_user_input or decision.default is None:
            missing.append(decision.id)
            continue
        resolved[decision.id] = coerce_answer(decision, decision.default)
    if missing:
        raise DecisionError(
            "These decisions have no safe default and must be answered: "
            + ", ".join(missing),
            decision_id=missing[0],
        )
    return resolved


def build_manifest(plan: DecisionPlan, answers: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble a manifest dictionary from a complete set of answers."""

    resolved = share_answers(plan, resolve_answers(plan, answers))
    get = resolved.get

    # A heterogen the user declined as a ligand is an unknown heterogen, and
    # normalisation stops on one unless removing them was allowed. Saying so
    # here costs nothing; saying so in Colab costs the whole setup cell.
    declined = [
        decision.subject.to_selector()
        for decision in plan.decisions
        if decision.section == "ligands"
        and decision.id.endswith(".include")
        and resolved.get(decision.id) is False
    ]
    if declined and not get("structure.remove_unknown_heterogens", False):
        listed = ", ".join(
            f"{item['chain']}:{item['resname']}{item['resid']}" for item in declined
        )
        raise DecisionError(
            f"You declined {listed} as a ligand but answered no to removing "
            "unconfigured heterogens, so the build would stop on it. Either "
            "include it, or answer yes to removing unconfigured heterogens.",
            decision_id="structure.remove_unknown_heterogens",
        )

    _require_buildable_heterogens(resolved)
    bonded_sites = [
        decision.id.split(".")[1]
        for decision in plan.decisions
        if decision.section == "metals"
        and decision.id.endswith(".model")
        and resolved.get(decision.id) == "bonded_mcpb"
    ]
    _require_one_bonded_fit(plan, resolved, bonded_sites)
    _require_a_fit_for_fitted_cofactors(plan, resolved)
    if bonded_sites and get("structure.keep_crystal_waters", True) is False:
        raise DecisionError(
            "A bonded metal site is refined with QM/MM, which protonates and "
            "includes the active-site waters, so the crystal waters have to be "
            "kept. Answer yes to keeping them, or model "
            f"{', '.join(bonded_sites)} nonbonded.",
            decision_id="structure.keep_crystal_waters",
        )

    # A water bound to a bonded metal joins the MCPB quantum model, and a
    # four-point water's extra point has no nucleus to put in it.
    if bonded_sites and get("protein.water_model") not in (None, "TIP3P"):
        waters_on_metal = [
            f"{target.chain}:{target.resname}{target.resid}"
            for decision in plan.decisions
            if decision.id.endswith(".confirm_donors")
            and decision.id.split(".")[1] in bonded_sites
            and resolved.get(decision.id)
            for target in decision.targets[1:]
            if target.resname in {"HOH", "WAT", "H2O", "DOD"}
        ]
        if waters_on_metal:
            raise DecisionError(
                f"{', '.join(waters_on_metal)} is a water bound to a bonded metal, "
                "which makes it part of the MCPB quantum model. That needs a "
                f"three-point water, and {get('protein.water_model')} has an extra "
                "point with no nucleus. Choose ff14SB with TIP3P for this system.",
                decision_id="protein.water_model",
            )

    ligand_blocks = _link_ligand_copies(plan, resolved, _ligand_blocks(plan, resolved))
    metal_blocks = _link_metal_copies(plan, resolved, _metal_blocks(plan, resolved))
    _require_one_template_per_metal_name(metal_blocks)
    keep_chains = _kept_chains(plan, resolved.get("structure.chains", "all"))
    removed = _in_chains(_removed_residues(plan, resolved), keep_chains)
    replacements = _in_chains(_modified_residue_replacements(plan, resolved), keep_chains)
    repair: dict[str, Any] = {}
    if resolved.get("structure.repair.add_missing_heavy_atoms") or replacements:
        repair = {
            "backend": "pdbfixer",
            "add_missing_heavy_atoms": bool(
                resolved.get("structure.repair.add_missing_heavy_atoms")
            ),
            "unresolved_residues": resolved.get(
                "structure.repair.unresolved_residues", "leave_absent"
            ),
        }
        if replacements:
            repair["replace_modified_residues"] = replacements

    manifest: dict[str, Any] = {
        "project": {
            "name": resolved["project.name"],
            "input_structure": str(plan.structure_path),
            "output_dir": resolved["project.output_dir"],
        },
        "structure": {
            "keep_chains": keep_chains,
            "keep_crystal_waters": get("structure.keep_crystal_waters", True),
            "keep_crystal_ions": get("structure.keep_crystal_ions", True),
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": get("structure.remove_unknown_heterogens", False),
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
            **({"remove_residues": removed} if removed else {}),
            **({"repair": repair} if repair else {}),
        },
        "protein": {
            "forcefield": resolved["protein.forcefield"],
            "water_model": resolved["protein.water_model"],
        },
        "protonation": _protonation_block(resolved, plan),
        "disulfides": {
            "auto_detect": get("disulfides.auto_detect", True),
            "detection_cutoff_angstrom": 2.2,
            "force": [],
            "forbid": [],
        },
        "ligands": ligand_blocks,
        "metals": metal_blocks,
        "refinement": {},
        "solvation": _solvation_block(resolved),
        "validation": {
            "run_openmm_energy_check": True,
            "fail_on_warnings": False,
            "fail_on_missing_parameters": True,
            "fail_on_noninteger_ligand_charge": True,
        },
        "molecular_dynamics": _md_block(resolved),
    }
    manifest["refinement"] = _refinement_block(
        plan, resolved, manifest["metals"], manifest["ligands"]
    )
    return manifest


# What stops the build when a group of heterogens is kept: none of them can
# be built as it is.
KEEPING_STOPS_THE_BUILD = {
    "structure.replace_modified_residues": (
        "mdprep cannot build a modified amino acid inside a chain. Answer yes to "
        "replace each by its standard parent, or prepare this structure with a "
        "tool that has templates for them."
    ),
    "structure.remove_glycans": (
        "mdprep does not build glycans bonded to a protein. Answer yes to "
        "prepare the protein without them, or prepare the glycoprotein with "
        "GLYCAM, outside mdprep."
    ),
    "structure.remove_covalent_heterogens": (
        "mdprep does not build heterogens bonded to the protein. Answer yes to "
        "prepare the structure without them, or prepare it with a tool that "
        "builds covalent ligands."
    ),
}


def _require_buildable_heterogens(resolved: Mapping[str, Any]) -> None:
    for decision_id, message in KEEPING_STOPS_THE_BUILD.items():
        if resolved.get(decision_id) is False:
            raise DecisionError(message, decision_id=decision_id)


# Why each grouped removal removes what it does, for the structure report.
REMOVAL_REASONS = {
    "structure.remove_glycans": "glycan bonded to the protein, which mdprep does not build",
    "structure.remove_covalent_heterogens": (
        "bonded to the protein, which mdprep does not build"
    ),
}


def _removed_residues(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every residue the answers leave out, each with the reason."""

    removed: list[dict[str, Any]] = []
    for decision in plan.decisions:
        if decision.id in REMOVAL_REASONS and resolved.get(decision.id) is True:
            removed.extend(
                {"selector": target.to_selector(), "reason": REMOVAL_REASONS[decision.id]}
                for target in decision.targets
            )
        elif decision.id == "structure.remove_additives" and resolved.get(decision.id) is True:
            removed.extend(
                {
                    "selector": target.to_selector(),
                    "reason": "crystallisation additive ("
                    + CRYSTALLISATION_ADDITIVES.get(target.resname.strip().upper(), target.resname)
                    + ")",
                }
                for target in decision.targets
            )
        elif (
            decision.section == "metals"
            and decision.id.endswith(".model")
            and resolved.get(decision.id) == "remove"
        ):
            removed.append(
                {
                    "selector": decision.subject.to_selector(),
                    "reason": "metal left out as answered: not part of the system",
                }
            )
    return removed


def _modified_residue_replacements(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if resolved.get("structure.replace_modified_residues") is not True:
        return []
    return [
        {
            "selector": {
                "chain": item["chain"],
                "resname": item["resname"],
                "resid": item["resid"],
                "icode": item["icode"],
            },
            "parent": item["parent"],
        }
        for item in plan.findings.get("modified_residue_replacements", [])
    ]


def _in_chains(items: list[dict[str, Any]], keep_chains: Sequence[str]) -> list[dict[str, Any]]:
    """Only what is in a chain being built: the rest is left out with its chain."""

    if not keep_chains:
        return items
    return [item for item in items if item["selector"]["chain"] in keep_chains]


def share_answers(plan: DecisionPlan, resolved: Mapping[str, Any]) -> dict[str, Any]:
    """Give each copy's skipped questions the reference copy's answers.

    Only when the parameters are to be shared, and only for questions that
    would have been asked had they not been: a copy's ligand the reference
    copy declined is declined in every copy, and nothing is answered for a
    copy that is not being built.
    """

    shared = dict(resolved)
    if shared.get("structure.same_parametrization") is not True:
        return shared
    for decision in plan.decisions:
        if decision.copy_of is None or decision.id in shared:
            continue
        if decision.copy_of not in shared:
            continue
        if not all(condition.is_met(shared) for condition in decision.when):
            continue
        shared[decision.id] = shared[decision.copy_of]
    return shared


def answer_conflicts(plan: DecisionPlan, answers: Mapping[str, Any]) -> DecisionError | None:
    """A refusal the answers given so far already make certain, if any.

    ``build_manifest`` refuses two bonded sites that would each need a fit of
    their own -- but only at the end, after asking for the spin of a QM region
    that would span them both, which nobody can answer. A front end that
    checks this as each answer is given can say so at the question that
    caused it. Answers to questions no longer reachable do not count, and
    nothing here insists on questions not yet answered.
    """

    # Every question up to the last one answered has been passed, and one
    # passed without an answer took its default -- as a front end records
    # it. Questions further on have not been reached, so their defaults say
    # nothing yet.
    last = max(
        (index for index, decision in enumerate(plan.decisions) if decision.id in answers),
        default=-1,
    )
    reachable: dict[str, Any] = {}
    for decision in plan.decisions[: last + 1]:
        if not decision.is_applicable(reachable):
            continue
        raw = answers.get(decision.id, decision.default)
        if raw is None or raw == "":
            continue
        try:
            reachable[decision.id] = coerce_answer(decision, raw)
        except DecisionError:
            continue
    shared = share_answers(plan, reachable)
    bonded = [
        decision.id.split(".")[1]
        for decision in plan.decisions
        if decision.section == "metals"
        and decision.id.endswith(".model")
        and shared.get(decision.id) == "bonded_mcpb"
    ]
    try:
        _require_buildable_heterogens(shared)
        _require_one_bonded_fit(plan, shared, bonded)
        _require_a_fit_for_fitted_cofactors(plan, shared)
    except DecisionError as exc:
        return exc
    return None


def _kept_chains(plan: DecisionPlan, chosen: str) -> list[str]:
    """Chain letters to keep: the chosen protein chain and what belongs to it.

    A chain with no protein in it -- a prediction's ligand, a deposition's
    separate chain of ions -- is kept with the protein chain it sits against,
    and kept regardless when it touches none. Keeping every such chain, as
    mdprep used to, left the other copy's cofactors floating in the box.
    """

    if chosen == "all":
        return []
    owners = (
        plan.oligomer.chain_of_nonprotein_chain if plan.oligomer is not None else {}
    )
    return [
        chosen,
        *(
            chain
            for chain in plan.findings.get("non_protein_chains", [])
            if owners.get(chain) in (None, chosen)
        ),
    ]


def _shared_from(decision: Decision, resolved: Mapping[str, Any]) -> str | None:
    """The reference item whose answers this copy's item took, if it did."""

    if decision.copy_of is None or resolved.get("structure.same_parametrization") is not True:
        return None
    return decision.copy_of.split(".")[1]


# The fields that make two ligand residues one template. tleap builds every
# residue of one name from one template, so two residues sharing a name
# either agree on all of these or cannot both be built.
LIGAND_PARAMETER_FIELDS = (
    "net_charge",
    "multiplicity",
    "atom_types",
    "charge_method",
    "smiles",
)


def _link_ligand_copies(
    plan: DecisionPlan,
    resolved: Mapping[str, Any],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mark ligands that take another ligand's parameters.

    A copy whose answers were shared takes its reference's parameters. So,
    too, does any other ligand with the same residue name and the same
    answers: tleap builds every residue called AKG from the one AKG template,
    so two of them cannot carry different charges -- and two that were
    answered alike would otherwise be fitted twice, from two conformations,
    into two templates that collide. Two with the same name and different
    answers cannot be built at all; that is refused here, with the fix.
    """

    by_id = {block["id"]: block for block in blocks}
    for block in blocks:
        include = plan.by_id(f"ligand.{block['id']}.include")
        source = _shared_from(include, resolved)
        if source is not None and source in by_id:
            block["parameters_from"] = source

    atom_names = plan.findings.get("ligand_atom_names", {})
    first_by_name: dict[str, dict[str, Any]] = {}
    for block in blocks:
        name = block["selector"]["resname"]
        if "parameters_from" in block:
            continue
        first = first_by_name.get(name)
        if first is None:
            first_by_name[name] = block
            continue
        differing = [
            field_name
            for field_name in LIGAND_PARAMETER_FIELDS
            if block.get(field_name) != first.get(field_name)
        ]
        if atom_names.get(block["id"]) != atom_names.get(first["id"]):
            differing.append("atom names")
        if differing:
            asked = f"ligand.{block['id']}.{differing[0]}"
            raise DecisionError(
                f"{_selector_label(block['selector'])} and "
                f"{_selector_label(first['selector'])} are both called {name} but "
                f"differ in {', '.join(differing)}. tleap builds every residue of one "
                "name from one template, so they cannot both be built as they are. "
                "Give them the same answers, or rename one of them in the input.",
                decision_id=asked
                if any(decision.id == asked for decision in plan.decisions)
                else f"ligand.{block['id']}.include",
            )
        block["parameters_from"] = first["id"]
    return blocks


def _link_metal_copies(
    plan: DecisionPlan,
    resolved: Mapping[str, Any],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mark the metal sites of copies that share a site's parameters.

    A bonded copy's bonds are written in the order of its reference's, so
    the n-th bond of one is the n-th of the other and the build can carry
    the reference's templates across residue by residue.
    """

    by_id = {block["id"]: block for block in blocks}
    oligomer = plan.oligomer
    for block in blocks:
        model = plan.by_id(f"metal.{block['id']}.model")
        source = _shared_from(model, resolved)
        if source is None or source not in by_id:
            continue
        block["parameters_from"] = source
        reference = by_id[source]
        if block["model"] != "bonded_mcpb" or oligomer is None:
            continue
        chain = oligomer.chain_of_residue.get(_selector_residue_id(block["ions"][0]["selector"]))

        def in_this_copy(selector: Mapping[str, Any]) -> tuple[ResidueId, str]:
            residue = _selector_residue_id(selector)
            for copy in oligomer.copies_of(residue):
                if oligomer.chain_of_residue.get(copy) == chain:
                    return copy, selector["atom_name"]
            raise DecisionError(
                f"{_selector_label(selector)} has no counterpart in chain {chain}"
            )

        def in_reference_order(own: list[dict[str, Any]], theirs: list[dict[str, Any]], key: str):
            by_residue = {
                (_selector_residue_id(item[key]), item[key]["atom_name"]): item for item in own
            }
            ordered = []
            for item in theirs:
                counterpart = in_this_copy(item[key])
                if counterpart not in by_residue:
                    raise DecisionError(
                        f"{block['id']} has no bond to the counterpart of "
                        f"{_selector_label(item[key])} {item[key]['atom_name']}, so it is "
                        f"not a copy of {source}; answer no to sharing parameters",
                        decision_id="structure.same_parametrization",
                    )
                ordered.append(by_residue[counterpart])
            return ordered

        block["mcpb"]["bonds"] = in_reference_order(
            block["mcpb"]["bonds"], reference["mcpb"]["bonds"], "coordinator"
        )
        if "coordinating_ions" in reference:
            block["coordinating_ions"] = in_reference_order(
                block.get("coordinating_ions", []), reference["coordinating_ions"], "selector"
            )
    return blocks


def _require_one_bonded_fit(
    plan: DecisionPlan, resolved: Mapping[str, Any], bonded_sites: list[str]
) -> None:
    """Refuse two bonded sites that would each need an MCPB fit of their own.

    One fit per preparation is what mdprep can build: MCPB.py names its
    residue templates and atom types the same way for every site it fits,
    so two separate fits cannot be loaded side by side. A copy that shares
    its reference's parameters is not a second fit -- it reuses the first.
    """

    independent = [
        site
        for site in bonded_sites
        if _shared_from(plan.by_id(f"metal.{site}.model"), resolved) is None
    ]
    if len(independent) <= 1:
        return
    sharing = resolved.get("structure.same_parametrization")
    if sharing is True:
        advice = (
            "Each copy that matches the first one already reuses its fit; these "
            "do not match it, so model all but one of them nonbonded."
        )
    elif sharing is False:
        advice = (
            "Answer yes to giving every copy the same parameters, so the copies "
            "reuse one fit, or model all but one of them nonbonded."
        )
    else:
        advice = "Model all but one of them nonbonded."
    labels = [_residue_id_of(plan.by_id(f"metal.{site}.model").subject).display() for site in independent]
    raise DecisionError(
        f"The metals at {' and '.join(labels)} would each need a bonded MCPB fit of "
        "their own, and mdprep builds one fit per preparation: MCPB.py gives every "
        "fit the same residue and atom-type names, so two cannot be loaded together. "
        f"{advice}",
        decision_id=f"metal.{independent[-1]}.model",
    )


def _require_a_fit_for_fitted_cofactors(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> None:
    """A cofactor whose charges come from an MCPB fit needs a site to be fitted.

    A ligand on a metal defaults to taking its charges from the site's fit,
    and its question comes before the site's. Modelling the site nonbonded
    afterwards left it pointing at a fit that would never run, which the
    schema caught only at the end, as an error the page could not place.
    """

    shells = [
        decision
        for decision in plan.decisions
        if decision.section == "metals" and decision.id.endswith(".confirm_donors")
    ]
    # A cofactor on two metals -- a heme whose propionate holds a magnesium
    # as well as its own iron -- is fitted with whichever site is fitted.
    fitted = {
        (target.chain, target.resid, target.icode)
        for decision in shells
        if resolved.get(f"metal.{decision.id.split('.')[1]}.model") == "bonded_mcpb"
        and resolved.get(decision.id) is not False
        for target in decision.targets[1:]
    }
    for decision in shells:
        site = decision.id.split(".")[1]
        model = resolved.get(f"metal.{site}.model")
        confirmed = resolved.get(decision.id)
        if model is None or (model == "bonded_mcpb" and confirmed is not False):
            continue
        donors = {
            (target.chain, target.resid, target.icode) for target in decision.targets[1:]
        } - fitted
        for ligand in plan.decisions:
            if ligand.section != "ligands" or not ligand.id.endswith(".charge_method"):
                continue
            if resolved.get(ligand.id) != "mcpb_resp_pyscf":
                continue
            subject = ligand.subject
            if (subject.chain, subject.resid, subject.icode) not in donors:
                continue
            metal = _residue_id_of(decision.subject).display()
            cofactor = _residue_id_of(subject).display()
            why = (
                "is modelled nonbonded, so there is no fit"
                if model != "bonded_mcpb"
                else "has its coordination shell written by hand, so mdprep fits nothing"
            )
            raise DecisionError(
                f"{cofactor}'s charges were to come from the MCPB fit of the site at "
                f"{metal}, which {why}. Go back and give {cofactor} another charge "
                "method, such as AM1-BCC, or model the site bonded.",
                decision_id=f"metal.{site}.model"
                if model != "bonded_mcpb"
                else decision.id,
            )


def _require_one_template_per_metal_name(blocks: list[dict[str, Any]]) -> None:
    """Two metals with one residue name must be one Amber ion.

    tleap builds a residue from the template of its name, so an FE2 that is
    Fe(II) in one site and Fe(III) in another, or carries two different
    Li/Merz sets, cannot be built. A bonded site's metal counts with the set
    it is built with before MCPB.py replaces it.
    """

    seen: dict[str, tuple[tuple[object, ...], str]] = {}
    for block in blocks:
        family = (
            block["nonbonded"]["parameter_set"]
            if block["model"] == "nonbonded"
            else block["mcpb"]["provisional_nonbonded_parameter_set"]
        )
        for ion in block["ions"]:
            name = ion["selector"]["resname"]
            signature = (ion["element"], ion["charge"], family)
            previous = seen.get(name)
            if previous is not None and previous[0] != signature:
                raise DecisionError(
                    f"{_selector_label(ion['selector'])} and {previous[1]} are both "
                    f"called {name} but differ in charge or ion parameter set, and "
                    "tleap builds every residue of one name from one template. "
                    "Give them the same answers, or rename one of them in the input.",
                    decision_id=f"metal.{block['id']}.charge",
                )
            seen[name] = (signature, _selector_label(ion["selector"]))


def _selector_residue_id(selector: Mapping[str, Any]) -> ResidueId:
    return ResidueId(
        selector["chain"], selector["resname"], selector["resid"], selector.get("icode")
    )


def _selector_label(selector: Mapping[str, Any]) -> str:
    return f"{selector['chain']}:{selector['resname']}{selector['resid']}{selector.get('icode') or ''}"


def _metal_donor_overrides(
    resolved: Mapping[str, Any], plan: "DecisionPlan"
) -> list[dict[str, Any]]:
    """The stated protonation of each titratable side chain on a metal.

    These travel as ordinary manual overrides, which is what preparation
    looks for and what the rule about user overrides winning is written
    about. Without them the build refuses the site -- correctly, since
    coordination does not distinguish a carboxylate from a neutral acid --
    and before this nothing put the question, so the refusal arrived in
    Colab rather than in the builder.
    """

    overrides: list[dict[str, Any]] = []
    for decision in plan.decisions:
        if ".donor_state." not in decision.id:
            continue
        state = resolved.get(decision.id)
        if not state:
            continue
        target = decision.subject
        overrides.append(
            {
                "selector": {
                    "chain": target.chain,
                    "resname": target.resname,
                    "resid": target.resid,
                    "icode": target.icode,
                },
                "state": str(state),
                "reason": (
                    "Coordinates a metal modelled with bonded MCPB; stated by "
                    "the user because coordination alone does not distinguish "
                    "the neutral and ionised forms."
                ),
            }
        )
    return overrides


def _protonation_block(
    resolved: Mapping[str, Any], plan: "DecisionPlan"
) -> dict[str, Any]:
    method = resolved["protonation.method"]
    block: dict[str, Any] = {
        "ph": resolved.get("protonation.ph", 7.0),
        "method": method,
        "overrides": _metal_donor_overrides(resolved, plan),
    }
    if method == "propka_xtb_his":
        block["histidine"] = {
            "neutral_tautomer_method": "xtb",
            "xtb": {
                "model": "gfn2",
                # A single point. The heavy atoms come from the structure and
                # are meant to stay where it put them, so the only thing an
                # optimisation relaxes is the hydrogen the two tautomers
                # differ by -- at the cost of an optimisation per tautomer per
                # histidine. "opt" is supported and properly constrained for
                # anyone who wants it.
                "mode": "sp",
                "num_threads": resolved.get("protonation.histidine.num_threads", 1),
            },
        }
    return block


def _ligand_blocks(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for decision in plan.decisions:
        if decision.section != "ligands" or not decision.id.endswith(".include"):
            continue
        if not resolved.get(decision.id):
            continue
        ligand_id = decision.id.split(".")[1]
        blocks.append(
            {
                "id": ligand_id,
                "selector": decision.subject.to_selector(),
                "net_charge": resolved[f"ligand.{ligand_id}.net_charge"],
                "multiplicity": 1,
                "atom_types": resolved[f"ligand.{ligand_id}.atom_types"],
                "charge_method": resolved[f"ligand.{ligand_id}.charge_method"],
            }
        )
        smiles = resolved.get(f"ligand.{ligand_id}.smiles")
        if smiles:
            blocks[-1]["smiles"] = smiles

    return blocks


def _metal_blocks(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for decision in plan.decisions:
        if decision.section != "metals" or not decision.id.endswith(".model"):
            continue
        site_id = decision.id.split(".")[1]
        model = resolved.get(decision.id)
        if model is None or model == "remove":
            continue
        ion = {
            "selector": decision.subject.to_atom_selector(),
            "element": decision.subject.element,
            "charge": resolved[f"metal.{site_id}.charge"],
        }
        if model == "nonbonded":
            blocks.append(
                {
                    "id": site_id,
                    "model": "nonbonded",
                    "ions": [ion],
                    "nonbonded": {
                        "parameter_set": resolved[f"metal.{site_id}.parameter_set"]
                    },
                }
            )
            continue
        # A bonded site is only written once its coordination shell has been
        # confirmed; otherwise it is reported for manual work instead.
        if not resolved.get(f"metal.{site_id}.confirm_donors"):
            continue
        confirm = plan.by_id(f"metal.{site_id}.confirm_donors")
        # The region's spin is the site's: one site is fitted, and the
        # cofactors in it are closed shell.
        multiplicity = resolved["refinement.total_qm_multiplicity"]
        model_charge = resolved[f"metal.{site_id}.model_charge"]
        ion["multiplicity"] = multiplicity
        backend = resolved[f"metal.{site_id}.hessian_backend"]
        # A donor that is a one-atom residue is an ion, not a cofactor: Amber
        # parameterises a halide directly, and antechamber refuses a single
        # atom, so it can be neither an ions: entry (those are metals) nor a
        # GAFF ligand. It is declared on the site instead.
        coordinating = [
            {
                "selector": donor.to_atom_selector(),
                "element": _monatomic_ion_element(donor),
                "charge": _monatomic_ion_charge(donor),
            }
            for donor in confirm.targets[1:]
            if _is_monatomic_ion_target(donor)
        ]
        site_block: dict[str, Any] = {
            "id": site_id,
            "model": "bonded_mcpb",
            "ions": [ion],
        }
        if coordinating:
            site_block["coordinating_ions"] = coordinating
        blocks.append(
            {
                **site_block,
                "mcpb": {
                    "workflow": "pyscf",
                    "provisional_nonbonded_parameter_set": "12_6",
                    "cutoff_angstrom": COORDINATION_CUTOFF_ANGSTROM,
                    "bonds": [
                        {
                            "ion": confirm.subject.to_atom_selector(),
                            "coordinator": donor.to_atom_selector(),
                        }
                        for donor in confirm.targets[1:]
                    ],
                    "additional_residues": [],
                    "force_constant_method": "seminario",
                    "charge_restraint": "backbone_heavy",
                    "software_version": "gau",
                    # The capped model carries the metal's spin; the donors are
                    # treated as closed shell, which the question text states.
                    "small_model_charge": model_charge,
                    "small_model_spin": multiplicity,
                    "large_model_charge": model_charge,
                    "large_model_spin": multiplicity,
                    "pyscf": _mcpb_pyscf_block(
                        backend,
                        porphyrin=any(
                            _id_stem_of(_residue_id_of(donor))
                            in plan.findings.get("known_cofactors", {})
                            for donor in confirm.targets[1:]
                        ),
                    ),
                },
            }
        )
    return blocks


# Halides are the coordinating ions that actually turn up in a metal site.
# Amber parameterises each directly, so the charge follows from the element.
MONATOMIC_ION_CHARGES = {"F": -1, "CL": -1, "BR": -1, "I": -1}


def _is_monatomic_ion_target(target: ResidueTarget) -> bool:
    """Whether this donor is a one-atom ion rather than part of a molecule.

    A lone ion is written with its residue and its atom both named after the
    element, which a donor target does carry; the element field is only
    populated for the metal itself.
    """

    resname = (target.resname or "").strip().upper()
    if resname not in MONATOMIC_ION_CHARGES:
        return False
    return (target.atom_name or "").strip().upper() == resname


def _monatomic_ion_element(target: ResidueTarget) -> str:
    resname = (target.resname or "").strip().upper()
    return resname[0] + resname[1:].lower()


def _monatomic_ion_charge(target: ResidueTarget) -> int:
    return MONATOMIC_ION_CHARGES[(target.resname or "").strip().upper()]


def _mcpb_pyscf_block(hessian_backend: str, *, porphyrin: bool = False) -> dict[str, Any]:
    """QM settings for one bonded site.

    The geometry comes from the QM/MM refinement stage rather than a
    pre-optimised PDB, so that nothing has to be prepared by hand first.

    No executable is written. g-xTB is a separate build and the xtb conda
    installs is GFN-only, so naming "xtb" here would hand a g-xTB run the
    wrong binary; leaving it out lets mdprep look for a real one and say what
    to install when there is none. method and basis are still required by the
    schema, and are what a later esp_backend: pyscf would use.
    """

    block: dict[str, Any] = {
        "geometry_source": "qmmm_refinement",
        "hessian_backend": "pyscf" if hessian_backend == "pyscf" else "xtb",
        # GFN2 for the ESP regardless of the Hessian method: fitting the large
        # model is where the DFT cost actually lands, and GFN2 has no trouble
        # with it. Switch to pyscf when the charges need to be RESP proper.
        "esp_backend": "pyscf" if hessian_backend == "pyscf" else "xtb",
        "method": "B3LYP",
        "basis": "6-31G*",
    }
    if hessian_backend in {"gfn2", "gxtb"}:
        block["xtb"] = {"model": hessian_backend, "num_threads": 8}
        if porphyrin:
            # An open-shell heme's SCF oscillates at xTB's own 300 K and does
            # not converge in any number of cycles; smeared at 1000 K a
            # low-spin Fe(III) converges in seconds. A high-spin one can take
            # a thousand cycles, and can settle in more than one state, so the
            # lowest of several starts is kept. See McpbXtbHessianConfig.
            block["xtb"]["electronic_temperature_kelvin"] = 1000.0
            block["xtb"]["max_scf_iterations"] = 3000
            block["xtb"]["scf_annealing_kelvin"] = [2000.0, 3000.0, 5000.0]
    return block


def _refinement_block(
    plan: DecisionPlan,
    resolved: Mapping[str, Any],
    metals: Sequence[Mapping[str, Any]],
    ligands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """QM/MM refinement, which a bonded site takes its geometry from."""

    # The site that is fitted. A copy that reuses its parameters is not
    # fitted, so its geometry is not refined for a fit either.
    bonded = [
        site
        for site in metals
        if site.get("model") == "bonded_mcpb" and "parameters_from" not in site
    ]
    if not bonded:
        return {"enabled": False}
    # The QM region is that site and the ligands of its own copy of the
    # protein. A ligand in another copy is tens of angstroms away; putting it
    # in the same quantum system would add a disconnected fragment, and with
    # it a spin state nobody asked about.
    oligomer = plan.oligomer
    region_chains = {
        oligomer.chain_of_residue.get(_selector_residue_id(ion["selector"]))
        if oligomer is not None
        else None
        for site in bonded
        for ion in site["ions"]
    }
    region_ligands = [
        entry["id"]
        for entry in ligands
        if "parameters_from" not in entry
        and (
            oligomer is None
            or oligomer.chain_of_residue.get(_selector_residue_id(entry["selector"]))
            in region_chains | {None}
        )
    ]
    return {
        "enabled": True,
        # The force-field backend holds every heavy atom and relaxes the
        # hydrogens in seconds. ASH is a manual checkout that no package
        # manager provides, so generating a manifest that requires it would
        # hand the user something they cannot run.
        "backend": "openmm",
        "qm_method": "gfn2_xtb",
        "qm_components": {
            "ligands": region_ligands,
            "metal_sites": [site["id"] for site in bonded],
            "metal_coordinating_residues": {},
        },
        "active_region_cutoff_angstrom": 4.0,
        "active_water_cutoff_angstrom": 8.0,
        "include_contact_waters": True,
        "movable_atoms": "active_region_hydrogens",
        "post_refinement_protonation": "reuse_initial",
        # Answered, not derived. The region's spin is not a function of
        # its parts once more than one of them is open-shell.
        "total_qm_multiplicity": resolved["refinement.total_qm_multiplicity"],
        "ash": {
            "python_executable": "python",
            "xtb_executable": "xtb",
            "allow_unusual_link_boundaries": True,
            "num_cores": 4,
            "platform": "CPU",
        },
    }


def ligands_needing_hydrogens(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> list[str]:
    """Included ligands with no hydrogens and no SMILES to place them from.

    mdprep will not guess a ligand's protonation -- a carboxylate and a
    carboxylic acid are the same heavy atoms -- so one of these needs a
    `smiles:` in its block, or hydrogens added to the input, before the run.
    The planner asks for the SMILES of every such ligand, so this is empty
    once its questions are answered. It used to list them regardless, and
    the builder then told a student who had just typed the SMILES to go and
    add one.
    """

    resolved = share_answers(plan, resolve_answers(plan, answers))
    needing: list[str] = []
    for decision in plan.decisions:
        if decision.section != "ligands" or not decision.id.endswith(".include"):
            continue
        if not resolved.get(decision.id):
            continue
        ligand_id = decision.id.split(".")[1]
        if ligand_id not in plan.findings.get("ligands_without_hydrogens", ()):
            continue
        if not resolved.get(f"ligand.{ligand_id}.smiles"):
            needing.append(ligand_id)
    return needing


def proposed_model_charge(
    plan: DecisionPlan, answers: Mapping[str, Any], site_id: str
) -> tuple[int | None, list[str]]:
    """The capped model's charge, summed from answers already given.

    The small and large models hold the metal and everything bonded to it, so
    the charge is the sum of their formal charges. mdprep asks for every one
    of those separately -- the metal's oxidation state, each ligand's net
    charge -- so by the time this question arrives it has the terms and
    should not make the user add them up again.

    A protein donor contributes the formal charge of its residue name, a
    coordinating halide its ionic charge, and a cofactor the net charge the
    user declared for it. Returns ``None`` when a term is still unknown,
    because half a sum is worse than no suggestion.

    The terms are returned alongside so a front end can show the arithmetic
    rather than a bare number nobody can check.
    """

    confirm = plan.by_id(f"metal.{site_id}.confirm_donors")
    if confirm is None or not confirm.targets:
        return None, []

    # An oxidation state accepted as proposed is as much an answer as one
    # typed in; the GUI happens to send both, the wizard and other front ends
    # need not.
    answers = _answers_with_defaults(plan, answers)
    raw_metal = answers.get(f"metal.{site_id}.charge")
    if raw_metal is None or str(raw_metal) == "":
        return None, []
    try:
        total = int(raw_metal)
    except (TypeError, ValueError):
        return None, []
    terms = [f"{confirm.targets[0].resname} {total:+d}"]

    # A ligand's own block carries the charge the user gave it, found by
    # position rather than by name: a cofactor's residue name says nothing
    # about its protonation.
    ligand_charges: dict[tuple[str, int, str | None], Any] = {}
    for decision in plan.decisions:
        if decision.section != "ligands" or not decision.id.endswith(".net_charge"):
            continue
        subject = decision.subject
        if subject is None:
            continue
        given = answers.get(decision.id)
        if given is None or str(given) == "":
            continue
        ligand_charges[(subject.chain, subject.resid, subject.icode)] = given

    # A titratable side chain on the metal carries the state it was given,
    # not the one its residue name suggests: the cysteine on a heme is named
    # CYS and is a thiolate. Until that is answered its term is unknown.
    donor_states = {
        decision.id.rsplit(".", 1)[1]: answers.get(decision.id)
        for decision in plan.decisions
        if decision.id.startswith(f"metal.{site_id}.donor_state.")
    }

    complete = True
    seen: set[tuple[str, int, str | None]] = set()
    for donor in confirm.targets[1:]:
        key = (donor.chain, donor.resid, donor.icode)
        # A bidentate cofactor appears once per donating atom and must be
        # counted once, and a porphyrin once for its four nitrogens.
        if key in seen:
            continue
        seen.add(key)
        resname = (donor.resname or "").strip().upper()
        stem = _id_stem_of(_residue_id_of(donor))
        if stem in donor_states:
            state = donor_states[stem]
            if state is None or str(state) == "":
                terms.append(f"{resname}{donor.resid} ?")
                complete = False
                continue
            resname = str(state).strip().upper()
        charge = STANDARD_DONOR_FORMAL_CHARGE.get(resname)
        if charge is None and key in ligand_charges:
            try:
                charge = int(ligand_charges[key])
            except (TypeError, ValueError):
                charge = None
        if charge is None:
            terms.append(f"{resname}{donor.resid} ?")
            complete = False
            continue
        terms.append(f"{resname}{donor.resid} {charge:+d}")
        total += charge
    return (total if complete else None), terms


def proposed_answers(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> dict[str, str]:
    """Values mdprep can work out from what has been answered so far.

    A front end shows these the way it shows a default: filled in, and
    changeable. They are not defaults on the decision itself because they
    depend on other answers, which a plan computed once cannot know.
    """

    return {key: value for key, (value, _) in _proposals(plan, answers).items()}


def proposal_explanations(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> dict[str, str]:
    """The arithmetic behind each proposal, so it can be checked not trusted."""

    return {key: why for key, (_, why) in _proposals(plan, answers).items()}


def _proposals(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> dict[str, tuple[str, str]]:
    proposals: dict[str, tuple[str, str]] = {}
    for decision in plan.decisions:
        if decision.section != "metals" or not decision.id.endswith(".model_charge"):
            continue
        site_id = decision.id.split(".")[1]
        value, terms = proposed_model_charge(plan, answers, site_id)
        if value is not None:
            proposals[decision.id] = (
                str(value),
                "Worked out from your earlier answers: "
                + " ".join(terms)
                + f" = {value:+d}. Change it if that is not this site.",
            )
    asked = {decision.id for decision in plan.decisions}
    for table, finding in (
        (KNOWN_METAL_FREE_COFACTORS, "known_cofactors"),
        (KNOWN_ORGANIC_COFACTORS, "known_organic_cofactors"),
    ):
        for ligand_id, resname in plan.findings.get(finding, {}).items():
            entry = table[resname]
            for field_name, value in (
                ("smiles", entry.smiles),
                ("net_charge", str(entry.net_charge)),
            ):
                # A cofactor the structure resolved with its hydrogens has
                # no SMILES question to propose an answer to.
                decision_id = f"ligand.{ligand_id}.{field_name}"
                if decision_id in asked:
                    proposals[decision_id] = (value, entry.explanation)
    return proposals


def _answers_with_defaults(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> dict[str, Any]:
    """Answers so far, plus the default of every question not yet answered.

    Unlike resolve_answers this never refuses: it is for proposals made while
    the form is still being filled in.
    """

    seen = dict(answers)
    for decision in plan.decisions:
        if decision.id in seen or decision.default is None:
            continue
        try:
            seen[decision.id] = coerce_answer(decision, decision.default)
        except DecisionError:
            continue
    return seen


def unresolved_mcpb_sites(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> list[str]:
    """Bonded sites that still need to be written out by hand.

    A bonded site is generated once its coordination shell has been confirmed.
    One stays unresolved when the proposed shell was rejected, or when no donor
    was found at all, because mdprep will not invent the bonds.
    """

    resolved = resolve_answers(plan, answers)
    unresolved: list[str] = []
    for decision in plan.decisions:
        if decision.section != "metals" or not decision.id.endswith(".model"):
            continue
        if resolved.get(decision.id) != "bonded_mcpb":
            continue
        site_id = decision.id.split(".")[1]
        if not resolved.get(f"metal.{site_id}.confirm_donors"):
            unresolved.append(site_id)
    return unresolved


def _solvation_block(resolved: Mapping[str, Any]) -> dict[str, Any]:
    if not resolved["solvation.enabled"]:
        return {"enabled": False}
    return {
        "enabled": True,
        "box": resolved["solvation.box"],
        "buffer_angstrom": resolved["solvation.buffer_angstrom"],
        "neutralize": True,
        "salt_concentration_molar": resolved["solvation.salt_concentration_molar"],
        "positive_ion": "Na+",
        "negative_ion": "Cl-",
    }


def _md_block(resolved: Mapping[str, Any]) -> dict[str, Any]:
    if not resolved["molecular_dynamics.enabled"]:
        return {"enabled": False}
    steps = resolved["molecular_dynamics.production.steps"]
    return {
        "enabled": True,
        "protocol": "roe_brooks_2020",
        "production": {
            "steps": steps,
            "timestep_fs": 2.0,
            "trajectory_interval_steps": max(1, min(10000, steps)),
            "state_interval_steps": max(1, min(5000, steps)),
            "checkpoint_interval_steps": max(1, min(50000, steps)),
        },
    }


def decision_sections(decisions: Sequence[Decision]) -> list[str]:
    """Section names in first-appearance order, for grouped rendering."""

    seen: list[str] = []
    for decision in decisions:
        if decision.section not in seen:
            seen.append(decision.section)
    return seen
