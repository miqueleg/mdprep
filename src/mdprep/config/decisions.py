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

from mdprep.structure.classify import is_metal_ion_residue
from mdprep.structure.completeness import chain_breaks, incomplete_residues
from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.models import ResidueRecord
from mdprep.structure.pdb import AltlocPolicy


DecisionKind = Literal["choice", "integer", "number", "text", "boolean"]

# What a 3D viewer should show while a question is on screen.
#   none      -- the question has no structural aspect (a project name, a step count)
#   targets   -- emphasise and frame `targets` / `target_resnames`
#   structure -- show the whole system (force field, solvation, box)
FocusMode = Literal["none", "targets", "structure"]


class DecisionError(ValueError):
    """Raised when a decision plan cannot be built or answered."""


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
    requires_user_input: bool = False
    evidence: tuple[str, ...] = ()
    # targets[0] is the subject of the decision -- the ligand being charged, the
    # ion being modelled -- and any further entries are surrounding context the
    # viewer should also show. The manifest builders rely on that ordering.
    targets: tuple[ResidueTarget, ...] = ()
    # Residue names standing in for a set too large to enumerate usefully, such
    # as every crystallographic water. A viewer selects on these directly.
    target_resnames: tuple[str, ...] = ()
    focus: FocusMode = "none"
    minimum: int | float | None = None
    maximum: int | float | None = None

    @property
    def subject(self) -> "ResidueTarget":
        """The residue this decision is about."""

        if not self.targets:
            raise DecisionError(f"{self.id}: no residue recorded")
        return self.targets[0]

    def is_applicable(self, answers: Mapping[str, Any]) -> bool:
        return all(condition.is_met(answers) for condition in self.when)

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
            "evidence": list(self.evidence),
            "targets": [target.to_dict() for target in self.targets],
            "target_resnames": list(self.target_resnames),
            "focus": self.focus,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class DecisionPlan:
    """Every decision implied by one input structure."""

    structure_path: Path
    decisions: tuple[Decision, ...]
    findings: dict[str, Any] = field(default_factory=dict)

    def by_id(self, decision_id: str) -> Decision:
        for decision in self.decisions:
            if decision.id == decision_id:
                return decision
        raise DecisionError(f"Unknown decision id: {decision_id}")

    def applicable(self, answers: Mapping[str, Any]) -> list[Decision]:
        """Decisions still reachable given the answers collected so far."""

        return [decision for decision in self.decisions if decision.is_applicable(answers)]

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

    parts = [part for part in (residue.id.chain_id.strip().lower(),) if part]
    parts.append(residue.id.resname.lower())
    parts.append(str(residue.id.resid))
    if residue.id.icode:
        parts.append(residue.id.icode.strip().lower())
    return "_".join(parts)


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

# First-shell metal-donor distances sit near 2.0-2.4 A. 2.6 A admits a long
# halide or carboxylate contact while excluding the second-shell carbons that
# sit behind a coordinating oxygen, which a 3.0 A search picks up.
COORDINATION_CUTOFF_ANGSTROM = 2.6

# Carbon and hydrogen are not metal donors; admitting them turns the atom
# behind a coordinating oxygen into a spurious bond.
METAL_DONOR_ELEMENTS = {"N", "O", "S", "SE", "P", "F", "CL", "BR", "I"}


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


def metal_donor_atoms(
    residues: Sequence[ResidueRecord],
    metal: ResidueRecord,
    *,
    cutoff_angstrom: float = COORDINATION_CUTOFF_ANGSTROM,
) -> list[tuple[ResidueRecord, Any, float]]:
    """Candidate first-shell donors of a metal, nearest first.

    This proposes a coordination shell for a user to confirm; it does not
    decide one. mdprep requires coordinating bonds to be explicit in the
    manifest, and the questions built from this keep them so.
    """

    origin = metal.atoms[0]
    found: list[tuple[ResidueRecord, Any, float]] = []
    for residue in residues:
        if residue.id == metal.id:
            continue
        for atom in residue.atoms:
            element = (atom.element or "").strip().upper()
            if element not in METAL_DONOR_ELEMENTS:
                continue
            distance = (
                (atom.x - origin.x) ** 2
                + (atom.y - origin.y) ** 2
                + (atom.z - origin.z) ** 2
            ) ** 0.5
            if distance <= cutoff_angstrom:
                found.append((residue, atom, distance))
    found.sort(key=lambda item: item[2])
    return found


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
    ligands = [
        residue
        for residue in summary.likely_ligands
        if not is_metal_ion_residue(residue) and residue.id not in monatomic_donors
    ]
    decisions: list[Decision] = []
    decisions.extend(_project_decisions(path))
    decisions.extend(_chain_decision(summary))
    decisions.extend(_structure_decisions(summary))
    decisions.extend(_repair_decisions(summary))
    decisions.extend(_protein_decisions())
    decisions.extend(_protonation_decisions(summary))
    decisions.extend(_disulfide_decisions(summary))
    # A ligand that coordinates a metal belongs to that site's quantum model.
    donor_residue_ids = {
        donor.id
        for metal in summary.metal_ions
        for donor, _, _ in metal_donor_atoms(summary.structure.residues, metal)
    }
    for residue in ligands:
        decisions.extend(
            _ligand_decisions(
                residue, coordinates_metal=residue.id in donor_residue_ids
            )
        )
    for residue in summary.metal_ions:
        decisions.extend(_metal_decisions(residue, summary.structure.residues))
    decisions.extend(_solvation_decisions())
    decisions.extend(_md_decisions())

    # A ligand or metal in a chain the user did not select must not be asked
    # about, and must not reach the manifest.
    selectable = set(protein_chain_ids(summary))
    if any(decision.id == "structure.chains" for decision in decisions):
        decisions = [
            replace(
                decision,
                when=decision.when
                + (
                    Condition(
                        "structure.chains", ("all", decision.targets[0].chain)
                    ),
                ),
            )
            if (
                decision.section in {"ligands", "metals"}
                and decision.targets
                and decision.targets[0].chain in selectable
            )
            else decision
            for decision in decisions
        ]

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
            item.describe() for item in chain_breaks(summary.structure.residues)
        ],
        "structure_warnings": list(summary.structure.warnings),
    }
    return DecisionPlan(
        structure_path=path, decisions=tuple(decisions), findings=findings
    )


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
            why="Where prepared structures, logs and Amber files are written.",
            default=f"prepared/{name}",
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
    """Which chain to build, when the input holds more than one."""

    # Only chains carrying protein are copies of the complex. A chain holding
    # nothing but a metal or a ligand is part of whichever copy it belongs to
    # and is kept regardless, so offering it here would invite discarding it.
    chains = protein_chain_ids(summary)
    if len(chains) < 2:
        return []
    # A bonded MCPB site is single-chain, and a crystal structure with several
    # copies of a metalloenzyme is the usual reason to be asked at all, so the
    # presence of a metal makes this a decision with no safe default.
    has_metal = bool(summary.metal_ions)
    choices = [
        Choice(
            chain,
            f"Chain {chain} only",
            "One copy of the complex; required for a bonded MCPB metal site.",
        )
        for chain in chains
    ]
    choices.append(
        Choice(
            "all",
            "All chains",
            "Build the whole asymmetric unit. Not usable with a bonded MCPB site.",
        )
    )
    return [
        Decision(
            id="structure.chains",
            section="structure",
            question="Which chain should be prepared?",
            kind="choice",
            why=(
                "The input contains more than one chain. MCPB metal-site inputs "
                "are single-chain, and a crystallographic multimer is usually "
                "several copies of the same complex rather than the assembly you "
                "want to simulate."
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
            focus="structure",
        )
    ]


def coordinating_water_residues(summary: InspectionSummary) -> list[ResidueRecord]:
    """Waters in the first shell of a metal, which are part of its site."""

    waters = {residue.id for residue in summary.water_residues}
    found: list[ResidueRecord] = []
    for metal in summary.metal_ions:
        for residue, _, _ in metal_donor_atoms(summary.structure.residues, metal):
            if residue.id in waters and all(
                residue.id != seen.id for seen in found
            ):
                found.append(residue)
    return found


def _structure_decisions(summary: InspectionSummary) -> list[Decision]:
    decisions: list[Decision] = []
    # A water coordinating a metal is part of that site. Discarding it would
    # leave a bond pointing at a residue that no longer exists, so it is not
    # offered as a choice, in the same way a hetero-only chain is not.
    if coordinating_water_residues(summary):
        return decisions
    if summary.water_residues:
        decisions.append(
            Decision(
                id="structure.keep_crystal_waters",
                section="structure",
                question="Keep the crystallographic waters?",
                kind="boolean",
                why=(
                    "Buried or bridging waters are often mechanistically important. "
                    "Discarding them is a chemistry decision, not a cleanup step."
                ),
                default="true",
                evidence=(f"{len(summary.water_residues)} water residues in the input",),
                target_resnames=tuple(
                    sorted({residue.id.resname for residue in summary.water_residues})
                ),
                focus="targets",
            )
        )
    unknown = [
        residue
        for residue in summary.heterogen_residues
        if not is_metal_ion_residue(residue)
    ]
    if unknown:
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
                    "Any heterogen left unconfigured will stop the build. Answer true "
                    "only if you have checked that every residue listed below is "
                    "genuinely disposable; otherwise answer false and configure them."
                ),
                evidence=tuple(residue.id.display() for residue in unknown),
                targets=tuple(_target_for(residue) for residue in unknown),
                focus="targets",
            )
        )
    return decisions


def _repair_decisions(summary: InspectionSummary) -> list[Decision]:
    """Ask about incomplete residues before any expensive stage runs."""

    incomplete = incomplete_residues(summary.structure.residues)
    breaks = chain_breaks(summary.structure.residues)
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
    residue: ResidueRecord, *, coordinates_metal: bool = False
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


def proposed_oxidation_state(resname: str, element: str | None) -> tuple[int | None, str]:
    """An oxidation state the evidence settles, and why -- or no proposal."""

    name = (resname or "").strip().upper()
    if name in OXIDATION_STATE_BY_RESNAME:
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
    return (
        None,
        f"the residue name {name or 'given'} does not identify one, and "
        f"{symbol.capitalize() or 'this element'} has more than one common "
        "state, which differ in spin as well as charge",
    )


STANDARD_DONOR_FORMAL_CHARGE = {
    "HIS": 0, "HID": 0, "HIE": 0, "HIP": 1,
    "ASP": -1, "ASH": 0, "GLU": -1, "GLH": 0,
    "CYS": 0, "CYM": -1, "SER": 0, "THR": 0, "TYR": 0,
    "ASN": 0, "GLN": 0, "MET": 0, "LYS": 1, "ARG": 1,
    "HOH": 0, "WAT": 0,
    "CL": -1, "BR": -1, "F": -1, "I": -1,
}


def model_charge_arithmetic(
    metal_charge: int,
    donor_residues: Sequence[ResidueRecord],
) -> tuple[int | None, list[str]]:
    """Capped-model charge implied by standard donor states, and its terms.

    Returns ``None`` when a donor's formal charge is not something that can be
    read off a residue name -- an organic cofactor, for instance -- because the
    sum is then incomplete and must not be presented as an answer.
    """

    terms = [f"metal {metal_charge:+d}"]
    total = metal_charge
    complete = True
    for residue in donor_residues:
        name = residue.id.resname.strip().upper()
        charge = STANDARD_DONOR_FORMAL_CHARGE.get(_normalise_ion_resname(name))
        if charge is None:
            charge = STANDARD_DONOR_FORMAL_CHARGE.get(name)
        if charge is None:
            terms.append(f"{residue.id.display()} ?")
            complete = False
            continue
        terms.append(f"{residue.id.display()} {charge:+d}")
        total += charge
    return (total if complete else None), terms


def _metal_decisions(
    residue: ResidueRecord, all_residues: Sequence[ResidueRecord]
) -> list[Decision]:
    site_id = metal_site_id_for(residue)
    display = residue.id.display()
    element = _residue_element(residue)
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
        f"{display}, element {element}",
        (
            "within "
            f"{CONTACT_CUTOFF_ANGSTROM:.1f} A: "
            + ", ".join(neighbour.id.display() for neighbour in neighbours)
        )
        if neighbours
        else f"nothing within {CONTACT_CUTOFF_ANGSTROM:.1f} A",
    )
    targets = (_target_for(residue, element=element),) + contacts
    model_id = f"metal.{site_id}.model"
    bonded = (Condition(model_id, ("bonded_mcpb",)),)
    nonbonded = (Condition(model_id, ("nonbonded",)),)
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
            default="bonded_mcpb" if donors else "nonbonded",
            why=(
                "A nonbonded model keeps the ion free and suits a structural or "
                "loosely bound metal. A bonded MCPB model holds the coordination "
                "sphere together and is what a catalytic centre needs; it costs a "
                "quantum chemistry calculation. "
                + (
                    f"{len(donors)} donor atom(s) were found within bonding "
                    "distance, which is why the bonded model is proposed here."
                    if donors
                    else "No donor atom was found within bonding distance, so "
                    "there is no coordination sphere to hold together."
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
            ),
            evidence=evidence + donor_evidence,
            targets=targets,
            focus="targets",
        ),
        # Needed by both models, so it carries no condition.
        Decision(
            id=f"metal.{site_id}.charge",
            section="metals",
            question=f"Formal oxidation state (charge) of {display}",
            kind="integer",
            default=(
                str(_proposed_state)
                if (_proposed_state := proposed_oxidation_state(
                    residue.id.resname, element
                )[0]) is not None
                else None
            ),
            requires_user_input=_proposed_state is None,
            why=(
                "This sets the electron count, and with the spin state it sets "
                "what the quantum calculation is solving. "
                + (
                    f"Proposed as {_proposed_state:+d} because "
                    f"{proposed_oxidation_state(residue.id.resname, element)[1]}; "
                    "change it if that is not this site."
                    if _proposed_state is not None
                    else "No value is proposed here because "
                    f"{proposed_oxidation_state(residue.id.resname, element)[1]}. "
                    "Guessing would give a converged calculation with the wrong "
                    "parameters."
                )
            ),
            minimum=1,
            maximum=4,
            evidence=evidence,
            targets=targets,
            focus="targets",
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

    decisions.extend(
        [
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
            ),
            Decision(
                id=f"metal.{site_id}.multiplicity",
                section="metals",
                question=f"Spin multiplicity of {display}",
                kind="integer",
                default=None,
                requires_user_input=True,
                why=(
                    "The spin state changes the electronic structure the Hessian "
                    "and the charges are derived from. High-spin Fe(II) d6 is 5 and "
                    "high-spin Fe(III) d5 is 6; low-spin states differ."
                ),
                minimum=1,
                evidence=evidence,
                targets=targets,
                focus="targets",
                when=confirmed,
            ),
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
            + ", ".join(missing)
        )
    return resolved


def build_manifest(plan: DecisionPlan, answers: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble a manifest dictionary from a complete set of answers."""

    resolved = resolve_answers(plan, answers)
    get = resolved.get

    manifest: dict[str, Any] = {
        "project": {
            "name": resolved["project.name"],
            "input_structure": str(plan.structure_path),
            "output_dir": resolved["project.output_dir"],
        },
        "structure": {
            "keep_chains": (
                []
                if resolved.get("structure.chains", "all") == "all"
                else [
                    resolved["structure.chains"],
                    *plan.findings.get("non_protein_chains", []),
                ]
            ),
            "keep_crystal_waters": get("structure.keep_crystal_waters", True),
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": get("structure.remove_unknown_heterogens", False),
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
            **(
                {
                    "repair": {
                        "backend": "pdbfixer",
                        "add_missing_heavy_atoms": True,
                        "unresolved_residues": resolved.get(
                            "structure.repair.unresolved_residues", "leave_absent"
                        ),
                    }
                }
                if resolved.get("structure.repair.add_missing_heavy_atoms")
                else {}
            ),
        },
        "protein": {
            "forcefield": resolved["protein.forcefield"],
            "water_model": resolved["protein.water_model"],
        },
        "protonation": _protonation_block(resolved),
        "disulfides": {
            "auto_detect": get("disulfides.auto_detect", True),
            "detection_cutoff_angstrom": 2.2,
            "force": [],
            "forbid": [],
        },
        "ligands": _ligand_blocks(plan, resolved),
        "metals": _metal_blocks(plan, resolved),
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


def _protonation_block(resolved: Mapping[str, Any]) -> dict[str, Any]:
    method = resolved["protonation.method"]
    block: dict[str, Any] = {
        "ph": resolved.get("protonation.ph", 7.0),
        "method": method,
        "overrides": [],
    }
    if method == "propka_xtb_his":
        block["histidine"] = {
            "neutral_tautomer_method": "xtb",
            "xtb": {
                "model": "gfn2",
                "mode": "opt",
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
        if model is None:
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
        multiplicity = resolved[f"metal.{site_id}.multiplicity"]
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
                    "pyscf": _mcpb_pyscf_block(backend),
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


def _mcpb_pyscf_block(hessian_backend: str) -> dict[str, Any]:
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
    return block


def _refinement_block(
    plan: DecisionPlan,
    resolved: Mapping[str, Any],
    metals: Sequence[Mapping[str, Any]],
    ligands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """QM/MM refinement, which a bonded site takes its geometry from."""

    bonded = [site for site in metals if site.get("model") == "bonded_mcpb"]
    if not bonded:
        return {"enabled": False}
    multiplicities = {site["ions"][0]["multiplicity"] for site in bonded}
    return {
        "enabled": True,
        # The force-field backend holds every heavy atom and relaxes the
        # hydrogens in seconds. ASH is a manual checkout that no package
        # manager provides, so generating a manifest that requires it would
        # hand the user something they cannot run.
        "backend": "openmm",
        "qm_method": "gfn2_xtb",
        "qm_components": {
            "ligands": [entry["id"] for entry in ligands],
            "metal_sites": [site["id"] for site in bonded],
            "metal_coordinating_residues": {},
        },
        "active_region_cutoff_angstrom": 4.0,
        "active_water_cutoff_angstrom": 8.0,
        "include_contact_waters": True,
        "movable_atoms": "active_region_hydrogens",
        "post_refinement_protonation": "reuse_initial",
        "total_qm_multiplicity": max(multiplicities),
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
    """Included ligands whose hydrogens the structure never resolved.

    mdprep will not guess a ligand's protonation -- a carboxylate and a
    carboxylic acid are the same heavy atoms -- so one of these needs a
    `smiles:` added to its block, or hydrogens added to the input before the
    run. The builder reports them rather than asking, because a SMILES typed
    into a form is not something it can check against the residue it is for.
    """

    resolved = resolve_answers(plan, answers)
    needing: list[str] = []
    for decision in plan.decisions:
        if decision.section != "ligands" or not decision.id.endswith(".include"):
            continue
        if not resolved.get(decision.id):
            continue
        ligand_id = decision.id.split(".")[1]
        if ligand_id in plan.findings.get("ligands_without_hydrogens", ()):
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

    complete = True
    seen: set[tuple[str, int, str | None]] = set()
    for donor in confirm.targets[1:]:
        key = (donor.chain, donor.resid, donor.icode)
        # A bidentate cofactor appears once per donating atom and must be
        # counted once.
        if key in seen:
            continue
        seen.add(key)
        resname = (donor.resname or "").strip().upper()
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
                " ".join(terms) + f" = {value:+d}",
            )
    return proposals


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
