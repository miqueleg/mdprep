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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from mdprep.structure.classify import is_metal_ion_residue
from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.models import ResidueRecord
from mdprep.structure.pdb import AltlocPolicy


DecisionKind = Literal["choice", "integer", "number", "text", "boolean"]


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
    target: ResidueTarget | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None

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
            "target": self.target.to_dict() if self.target is not None else None,
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


def ligand_id_for(residue: ResidueRecord) -> str:
    """Stable manifest id for a detected ligand residue."""

    return f"{residue.id.resname.lower()}_{residue.id.resid}"


def metal_site_id_for(residue: ResidueRecord) -> str:
    return f"{residue.id.resname.lower()}_{residue.id.resid}_site"


def _target_for(residue: ResidueRecord, *, element: str | None = None) -> "ResidueTarget":
    atom_name = residue.atoms[0].name if len(residue.atoms) == 1 else None
    return ResidueTarget(
        chain=residue.id.chain_id,
        resname=residue.id.resname,
        resid=residue.id.resid,
        icode=residue.id.icode,
        element=element,
        atom_name=atom_name,
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
    ligands = [
        residue for residue in summary.likely_ligands if not is_metal_ion_residue(residue)
    ]
    decisions: list[Decision] = []
    decisions.extend(_project_decisions(path))
    decisions.extend(_structure_decisions(summary))
    decisions.extend(_protein_decisions())
    decisions.extend(_protonation_decisions(summary))
    decisions.extend(_disulfide_decisions(summary))
    for residue in ligands:
        decisions.extend(_ligand_decisions(residue))
    for residue in summary.metal_ions:
        decisions.extend(_metal_decisions(residue))
    decisions.extend(_solvation_decisions())
    decisions.extend(_md_decisions())

    findings = {
        "total_atoms": len(summary.structure.atoms),
        "total_residues": len(summary.structure.residues),
        "protein_residues": len(summary.protein_residues),
        "water_residues": len(summary.water_residues),
        "chains": summary.chain_ids,
        "ligands": [residue.id.display() for residue in ligands],
        "metal_ions": [residue.id.display() for residue in summary.metal_ions],
        "histidines": [residue.id.display() for residue in summary.histidines],
        "possible_disulfides": [
            f"{candidate.a.display()} -- {candidate.b.display()} "
            f"({candidate.distance_angstrom:.2f} A)"
            for candidate in summary.possible_disulfides
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


def _structure_decisions(summary: InspectionSummary) -> list[Decision]:
    decisions: list[Decision] = []
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
            )
        )
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
                "differs in how much it decides for you and what it needs installed."
            ),
            default="manual_only",
            choices=(
                Choice(
                    "manual_only",
                    "Manual only",
                    "You name every non-default state yourself. No external tools needed.",
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
        ),
        Decision(
            id="protonation.ph",
            section="protonation",
            question="Target pH",
            kind="number",
            why="PropKa assigns states at this pH.",
            default="7.0",
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
        )
    ]


def _ligand_decisions(residue: ResidueRecord) -> list[Decision]:
    ligand_id = ligand_id_for(residue)
    display = residue.id.display()
    evidence = (f"{display}, {len(residue.atoms)} atoms",)
    target = _target_for(residue)
    include_id = f"ligand.{ligand_id}.include"
    return [
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
            target=target,
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
            target=target,
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
            target=target,
        ),
        Decision(
            id=f"ligand.{ligand_id}.charge_method",
            section="ligands",
            question=f"Partial-charge method for {display}",
            kind="choice",
            why=(
                "AM1-BCC is fast and standard. The RESP routes are more accurate and "
                "much more expensive, and need extra QM settings in the manifest."
            ),
            default="am1bcc",
            choices=(
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
            target=target,
        ),
    ]


def _metal_decisions(residue: ResidueRecord) -> list[Decision]:
    site_id = metal_site_id_for(residue)
    display = residue.id.display()
    element = _residue_element(residue)
    evidence = (f"{display}, element {element}",)
    target = _target_for(residue, element=element)
    model_id = f"metal.{site_id}.model"
    return [
        Decision(
            id=model_id,
            section="metals",
            question=f"How should the metal site at {display} be modelled?",
            kind="choice",
            # No default: the two models give different chemistry, and the choice
            # depends on whether the metal is structural or catalytic.
            default=None,
            requires_user_input=True,
            why=(
                "A nonbonded model keeps the ion free and is adequate for a "
                "structural or loosely bound metal. A bonded MCPB model is required "
                "when coordination must be maintained, and needs external QM work "
                "that this wizard cannot generate for you."
            ),
            choices=(
                Choice(
                    "nonbonded",
                    "Nonbonded (Li/Merz)",
                    "Generated here. Pick an Amber ion parameter set below.",
                ),
                Choice(
                    "bonded_mcpb",
                    "Bonded (MCPB.py)",
                    "NOT generated here: the wizard writes a commented stub and you "
                    "complete it following docs/metals.md.",
                ),
            ),
            evidence=evidence,
            target=target,
        ),
        Decision(
            id=f"metal.{site_id}.charge",
            section="metals",
            question=f"Formal oxidation state (charge) of {display}",
            kind="integer",
            # No default: neither a residue name nor an element implies a charge.
            default=None,
            requires_user_input=True,
            why=(
                "Neither the PDB residue name nor the element identifies an "
                "oxidation state. Zn is almost always +2; Fe and Cu are not."
            ),
            minimum=1,
            maximum=4,
            evidence=evidence,
            target=target,
            when=(Condition(model_id, ("nonbonded",)),),
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
            when=(Condition(model_id, ("nonbonded",)),),
            target=target,
        ),
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
        ),
        Decision(
            id="solvation.box",
            section="solvation",
            question="Box shape",
            kind="choice",
            why="A truncated octahedron needs fewer waters than a box for the same padding.",
            default="truncated_octahedron",
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
            "keep_crystal_waters": get("structure.keep_crystal_waters", True),
            "altloc_policy": "highest_occupancy",
            "remove_unknown_heterogens": get("structure.remove_unknown_heterogens", False),
            "preserve_chain_ids": True,
            "remove_input_hydrogens": True,
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
        "solvation": _solvation_block(resolved),
        "validation": {
            "run_openmm_energy_check": True,
            "fail_on_warnings": False,
            "fail_on_missing_parameters": True,
            "fail_on_noninteger_ligand_charge": True,
        },
        "molecular_dynamics": _md_block(resolved),
    }
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
        assert decision.target is not None
        blocks.append(
            {
                "id": ligand_id,
                "selector": decision.target.to_selector(),
                "net_charge": resolved[f"ligand.{ligand_id}.net_charge"],
                "multiplicity": 1,
                "atom_types": resolved[f"ligand.{ligand_id}.atom_types"],
                "charge_method": resolved[f"ligand.{ligand_id}.charge_method"],
            }
        )
    return blocks


def _metal_blocks(
    plan: DecisionPlan, resolved: Mapping[str, Any]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for decision in plan.decisions:
        if decision.section != "metals" or not decision.id.endswith(".model"):
            continue
        site_id = decision.id.split(".")[1]
        if resolved.get(decision.id) != "nonbonded":
            # A bonded MCPB site needs external QM artifacts that cannot be
            # invented here; the caller reports it instead of emitting a guess.
            continue
        assert decision.target is not None
        blocks.append(
            {
                "id": site_id,
                "model": "nonbonded",
                "ions": [
                    {
                        "selector": decision.target.to_atom_selector(),
                        "element": decision.target.element,
                        "charge": resolved[f"metal.{site_id}.charge"],
                    }
                ],
                "nonbonded": {
                    "parameter_set": resolved[f"metal.{site_id}.parameter_set"]
                },
            }
        )
    return blocks


def unresolved_mcpb_sites(
    plan: DecisionPlan, answers: Mapping[str, Any]
) -> list[str]:
    """Metal sites the user chose to model with MCPB, which need manual work."""

    resolved = resolve_answers(plan, answers)
    return [
        decision.id.split(".")[1]
        for decision in plan.decisions
        if decision.section == "metals"
        and decision.id.endswith(".model")
        and resolved.get(decision.id) == "bonded_mcpb"
    ]


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
