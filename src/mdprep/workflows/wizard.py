"""Interactive manifest generation driven by a decision plan.

This module owns only the conversation: which question to ask next, how to show
it, and how to turn the collected answers into a manifest file. Everything about
*what* to ask lives in :mod:`mdprep.config.decisions`, so a web form or a
notebook can reuse the same plan without reimplementing any of it.
"""

from __future__ import annotations

from dataclasses import replace

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from mdprep.config.decisions import (
    answer_conflicts,
    Decision,
    DecisionError,
    DecisionPlan,
    build_manifest,
    plan_decisions,
    ligands_needing_hydrogens,
    proposed_answers,
    unresolved_mcpb_sites,
)
from mdprep.config.models import ManifestConfig
from mdprep.structure.pdb import AltlocPolicy


Prompter = Callable[[Decision], str]


@dataclass(frozen=True)
class WizardResult:
    manifest_path: Path
    answers: dict[str, Any]
    questions_asked: int
    mcpb_sites_needing_manual_work: tuple[str, ...]
    # Ligands whose hydrogens the structure never resolved. Reported rather
    # than asked about: a SMILES cannot be checked against the residue it is
    # for at the point the question would be asked.
    ligands_needing_hydrogens: tuple[str, ...] = ()


def format_question(decision: Decision) -> str:
    """Render one decision as the text a terminal front end shows."""

    lines: list[str] = ["", decision.question]
    for item in decision.evidence:
        lines.append(f"    found: {item}")
    lines.append(f"    why:   {decision.why}")
    if decision.kind == "choice":
        for choice in decision.choices:
            marker = "*" if choice.value == decision.default else " "
            lines.append(f"    {marker} {choice.value}: {choice.label} -- {choice.detail}")
    elif decision.kind == "boolean":
        lines.append("    answer: yes or no")
    if decision.requires_user_input:
        lines.append("    This cannot be guessed from the structure; there is no default.")
    elif decision.default is not None:
        lines.append(f"    press Enter to accept: {decision.default}")
    return "\n".join(lines)


MAX_ATTEMPTS_PER_DECISION = 10


def collect_answers(
    plan: DecisionPlan,
    prompt: Prompter,
    *,
    preset: Mapping[str, Any] | None = None,
    max_attempts: int = MAX_ATTEMPTS_PER_DECISION,
    notify: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], int]:
    """Walk the plan, asking only the questions the answers keep reachable.

    ``preset`` answers are accepted without being asked, which is how
    command-line flags and a resumed session are honoured.

    A question that must be answered is re-asked until it is, but only
    ``max_attempts`` times: a front end that cannot supply a value has to fail
    rather than spin, which matters for every non-interactive caller. An
    answer that rules the build out on its own -- a second bonded metal site
    needing a fit of its own -- is refused at its question, as an invalid one
    is, rather than after the last question. ``notify`` is told why.
    """

    preset = dict(preset or {})
    answers: dict[str, Any] = {}
    asked = 0
    for decision in plan.decisions:
        if not decision.is_applicable(answers):
            continue
        if decision.id in preset and preset[decision.id] is not None:
            answers[decision.id] = _coerce(decision, preset[decision.id])
            continue
        # Some answers follow from earlier ones -- a capped model's charge is
        # the sum of parts already given -- so they are offered the way a
        # default is, and can still be overridden.
        proposal = proposed_answers(plan, answers).get(decision.id)
        if proposal is not None:
            decision = replace(decision, default=proposal)
        last_error: str | None = None
        for _ in range(max_attempts):
            raw = prompt(decision).strip()
            if not raw:
                if decision.default is None or (
                    decision.requires_user_input and proposal is None
                ):
                    # Refusing to default here is the point of the whole design.
                    last_error = "a value is required and has no default"
                    continue
                raw = decision.default
            try:
                answers[decision.id] = _coerce(decision, raw)
            except DecisionError as exc:
                last_error = str(exc)
                if notify is not None:
                    notify(last_error)
                continue
            conflict = answer_conflicts(plan, answers)
            if conflict is not None and conflict.decision_id == decision.id:
                del answers[decision.id]
                last_error = str(conflict)
                if notify is not None:
                    notify(last_error)
                continue
            asked += 1
            break
        else:
            raise DecisionError(
                f"{decision.id}: no usable answer after {max_attempts} attempts"
                + (f" ({last_error})" if last_error else "")
            )
    return answers, asked


def _coerce(decision: Decision, raw: Any) -> Any:
    from mdprep.config.decisions import coerce_answer

    return coerce_answer(decision, raw)


def render_manifest_yaml(
    plan: DecisionPlan,
    answers: Mapping[str, Any],
    *,
    mcpb_sites: tuple[str, ...] = (),
) -> str:
    """Build, validate and serialise the manifest for a set of answers."""

    manifest = build_manifest(plan, answers)
    # Validating before writing means the wizard can never emit a manifest that
    # `mdprep config-check` would reject.
    ManifestConfig.model_validate(manifest)
    header = [
        "# mdprep manifest generated by `mdprep init --interactive`.",
        f"# Source structure: {plan.structure_path}",
        "# Every chemistry-sensitive value here was entered by a user, not inferred.",
        "# Review before running `mdprep prepare`.",
    ]
    for site in mcpb_sites:
        header.append(
            f"# NOTE: metal site {site!r} was selected as bonded_mcpb and is NOT in this"
        )
        header.append(
            "#       file. A bonded MCPB site needs external QM artifacts; add it by"
        )
        header.append("#       hand following docs/metals.md before preparing.")
    # Copies of the protein built with another copy's parameters, named up
    # front: the blocks repeat every setting, so the one line that makes them
    # copies is easy to miss further down.
    shared = [
        f"#   {block['id']} <- {block['parameters_from']}"
        for block in (*manifest["ligands"], *manifest["metals"])
        if "parameters_from" in block
    ]
    if shared:
        header.append(
            "# Built with another block's parameters (parameters_from), see docs/oligomers.md:"
        )
        header.extend(shared)
    body = yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)
    return "\n".join(header) + "\n\n" + body


def run_wizard(
    input_structure: str | Path,
    *,
    output_path: str | Path,
    prompt: Prompter,
    overwrite: bool = False,
    preset: Mapping[str, Any] | None = None,
    altloc_policy: AltlocPolicy = "highest_occupancy",
    disulfide_cutoff_angstrom: float = 2.2,
    max_attempts: int = MAX_ATTEMPTS_PER_DECISION,
    notify: Callable[[str], None] | None = None,
) -> WizardResult:
    """Plan, ask, and write a validated manifest."""

    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Output manifest already exists: {target}. Use --overwrite to replace it."
        )
    plan = plan_decisions(
        input_structure,
        altloc_policy=altloc_policy,
        disulfide_cutoff_angstrom=disulfide_cutoff_angstrom,
    )
    answers, asked = collect_answers(
        plan, prompt, preset=preset, max_attempts=max_attempts, notify=notify
    )
    mcpb_sites = tuple(unresolved_mcpb_sites(plan, answers))
    hydrogenless = tuple(ligands_needing_hydrogens(plan, answers))
    text = render_manifest_yaml(plan, answers, mcpb_sites=mcpb_sites)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return WizardResult(
        manifest_path=target,
        answers=answers,
        questions_asked=asked,
        mcpb_sites_needing_manual_work=mcpb_sites,
        ligands_needing_hydrogens=hydrogenless,
    )
