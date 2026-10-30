"""The residue stage: what tleap needs to build each kept noncanonical residue.

A block built from a template needs nothing computed: its library is
AmberTools', sourced after the protein force field. A block parameterised
from its SMILES is parameterised once, from the first residue it lists, and
every other residue of it is built from the same library -- tleap has one
template per residue name -- so they must have the same atoms.

The stage runs before any tleap build that needs it. Its files sit at fixed
paths under the output directory, ``residues/<NAME>/``, so every build of the
system loads the same ones (``residue_leap_inputs``).

A block whose charges are fitted with QMMESP is parameterised here with
AM1-BCC, under ``residues/<NAME>/provisional_am1bcc/``: that is the residue
the provisional system holds while it polarises the fit, which runs after
protonation (``qmmesp.py``). Only the fitted library is written where the
final build looks, so a build can never take the provisional one for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mdprep.ambertools.commands import AmberToolsError
from mdprep.config.models import ManifestConfig, NoncanonicalResidueConfig
from mdprep.leap.forcefields import ResidueLibraryFiles, protein_leaprc
from mdprep.leap.log_parser import LeapLogError, assert_tleap_success
from mdprep.leap.runner import TLeapRunError, run_tleap
from mdprep.residues.capping import ResidueCappingError, chain_context
from mdprep.residues.charges import ResidueChargeError
from mdprep.residues.frcmod import ResidueFrcmodError
from mdprep.residues.parameterise import (
    ParameterisedResidue,
    ResidueParameterisationError,
    parameterise_residue,
)
from mdprep.residues.templates import AMBER_RESIDUE_TEMPLATES, AmberResidueTemplate
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector

if TYPE_CHECKING:
    from mdprep.residues.qmmesp import ResidueQmmespResult


class ResidueStageError(ValueError):
    """Raised when a kept noncanonical residue cannot be built."""


@dataclass(frozen=True)
class ResidueLeapInputs:
    """What every tleap build of the system loads for its kept residues."""

    sources: tuple[str, ...] = ()
    files: tuple[ResidueLibraryFiles, ...] = ()
    atom_types: tuple[str, ...] = ()


@dataclass
class ResidueBlockResult:
    resname: str
    final_resname: str
    parameters: str
    residues: list[str]
    net_charge: int
    template: str | None = None
    library: str | None = None
    leaprc: str | None = None
    # For a QMMESP block, the AM1-BCC parameterisation the provisional
    # system is built with; the fit that replaces its charges is ``qmmesp``.
    parameterised: ParameterisedResidue | None = None
    qmmesp: "ResidueQmmespResult | None" = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "resname": self.resname,
            "final_resname": self.final_resname,
            "parameters": self.parameters,
            "residues": list(self.residues),
            "net_charge": self.net_charge,
            "template": self.template,
            "library": self.library,
            "leaprc": self.leaprc,
            "parameterisation": (
                self.parameterised.to_dict() if self.parameterised is not None else None
            ),
            "qmmesp": self.qmmesp.to_dict() if self.qmmesp is not None else None,
            "warnings": list(self.warnings),
        }


@dataclass
class ResidueStageResult:
    blocks: list[ResidueBlockResult]
    leap_inputs: ResidueLeapInputs

    def to_report_dict(self) -> dict[str, object]:
        return {
            "noncanonical_residues": [block.to_dict() for block in self.blocks],
            "leap_sources": list(self.leap_inputs.sources),
            "leap_libraries": [
                {
                    "resname": files.resname,
                    "library_path": str(files.library_path),
                    "frcmod_path": str(files.frcmod_path),
                }
                for files in self.leap_inputs.files
            ],
        }


def residue_directory(output_dir: str | Path, resname: str) -> Path:
    return Path(output_dir) / "residues" / resname


def provisional_directory(output_dir: str | Path, resname: str) -> Path:
    """Where a QMMESP block's AM1-BCC parameterisation is written."""

    return residue_directory(output_dir, resname) / "provisional_am1bcc"


def parameterisation_directory(
    output_dir: str | Path, block: NoncanonicalResidueConfig
) -> Path:
    """Where the residue stage parameterises ``block`` from its SMILES."""

    if block.parameters == "qmmesp_pyscf":
        return provisional_directory(output_dir, block.resname)
    return residue_directory(output_dir, block.resname)


def final_resname(block: NoncanonicalResidueConfig) -> str:
    """The name the residues of ``block`` are built under."""

    return block.template if block.template is not None else block.resname


def kept_residue_names(manifest: ManifestConfig) -> frozenset[str]:
    """The names kept residues are built under: residues of the protein chains."""

    return frozenset(final_resname(block) for block in manifest.noncanonical_residues)


def kept_residue_net_charges(manifest: ManifestConfig) -> dict[str, int]:
    """Each kept residue's net charge, by the name it is built under."""

    return {
        final_resname(block): expected_charge(block) for block in manifest.noncanonical_residues
    }


def expected_charge(block: NoncanonicalResidueConfig) -> int:
    if block.template is not None:
        return AMBER_RESIDUE_TEMPLATES[block.template].net_charge
    assert block.net_charge is not None
    return block.net_charge


def residue_leap_inputs(
    manifest: ManifestConfig, output_dir: str | Path, *, provisional: bool = False
) -> ResidueLeapInputs:
    """The sources and libraries a tleap build of this system needs.

    The parameterised residues' files must exist: the residue stage writes
    them before anything else runs tleap. A QMMESP block's fitted library is
    written after protonation, from a provisional system built with its
    AM1-BCC one, which is what ``provisional`` asks for; the final build
    takes only the fitted one, and stops if it is not there.
    """

    forcefield = manifest.protein.forcefield
    sources: list[str] = []
    files: list[ResidueLibraryFiles] = []
    atom_types: list[str] = []
    for block in manifest.noncanonical_residues:
        if block.parameters == "amber_library":
            assert block.template is not None
            leaprc = AMBER_RESIDUE_TEMPLATES[block.template].leaprc(forcefield)
            if leaprc is not None and leaprc not in sources:
                sources.append(leaprc)
            continue
        fitted_later = block.parameters == "qmmesp_pyscf" and not provisional
        directory = (
            residue_directory(output_dir, block.resname)
            if fitted_later
            else parameterisation_directory(output_dir, block)
        )
        item = ResidueLibraryFiles(
            resname=block.resname,
            library_path=directory / f"{block.resname}.lib",
            frcmod_path=directory / f"{block.resname}.frcmod",
        )
        missing = [path for path in (item.library_path, item.frcmod_path) if not path.is_file()]
        if missing:
            raise ResidueStageError(
                f"noncanonical residue {block.resname} has not been "
                f"{'fitted' if fitted_later else 'parameterised'}: "
                f"{', '.join(str(path) for path in missing)} missing. "
                + (
                    "Its QMMESP charges are fitted after protonation, before the final "
                    "build; a build without them does not fall back on the provisional "
                    "AM1-BCC library."
                    if fitted_later
                    else "The residue stage writes them before any tleap build."
                )
            )
        files.append(item)
        if block.atom_types not in atom_types:
            atom_types.append(block.atom_types)
    return ResidueLeapInputs(
        sources=tuple(sources), files=tuple(files), atom_types=tuple(atom_types)
    )


def run_residue_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
) -> ResidueStageResult:
    """Make what tleap needs for every block of ``noncanonical_residues``.

    ``structure`` is the normalised one, in which a block built from a
    template already carries the template's name.
    """

    blocks: list[ResidueBlockResult] = []
    forcefield = manifest.protein.forcefield
    try:
        for block in manifest.noncanonical_residues:
            residues = block_residues(structure, block)
            displays = [residue.id.display() for residue in residues]
            if block.parameters == "amber_library":
                assert block.template is not None
                template = AMBER_RESIDUE_TEMPLATES[block.template]
                blocks.append(
                    ResidueBlockResult(
                        resname=block.resname,
                        final_resname=template.name,
                        parameters=block.parameters,
                        residues=displays,
                        net_charge=template.net_charge,
                        template=template.name,
                        library=template.library_name(forcefield),
                        leaprc=template.leaprc(forcefield),
                    )
                )
                continue
            first = residues[0]
            names = first.atom_names()
            different = [
                residue.id.display() for residue in residues[1:] if residue.atom_names() != names
            ]
            if different:
                raise ResidueStageError(
                    f"{', '.join(different)} and {first.id.display()} are all kept as "
                    f"{block.resname} and built from one library, made from "
                    f"{first.id.display()}, but their atoms differ from its {names}."
                )
            if block.parameters == "qmmesp_pyscf":
                # The fitted library is this run's or nobody's.
                stale = residue_directory(output_dir, block.resname)
                for leftover in ("lib", "frcmod", "mol2"):
                    (stale / f"{block.resname}.{leftover}").unlink(missing_ok=True)
                (stale / "parameters.json").unlink(missing_ok=True)
            context = chain_context(structure.residues, first)
            parameterised = parameterise_residue(
                block,
                context,
                protein_forcefield=forcefield,
                work_dir=parameterisation_directory(output_dir, block),
            )
            own = {
                name.strip()
                for name in first.atom_names()
            }
            if own != set(parameterised.atom_types):
                raise ResidueStageError(
                    f"{first.id.display()}'s atoms {sorted(own)} are not those its "
                    f"{block.resname} library was made with, "
                    f"{sorted(parameterised.atom_types)}."
                )
            blocks.append(
                ResidueBlockResult(
                    resname=block.resname,
                    final_resname=block.resname,
                    parameters=block.parameters,
                    residues=displays,
                    net_charge=parameterised.net_charge,
                    parameterised=parameterised,
                    warnings=list(parameterised.warnings),
                )
            )
        inputs = residue_leap_inputs(manifest, output_dir, provisional=True)
    except (
        AmberToolsError,
        ResidueCappingError,
        ResidueChargeError,
        ResidueFrcmodError,
        ResidueParameterisationError,
    ) as exc:
        write_stage_failure(output_dir, exc)
        raise ResidueStageError(str(exc)) from exc
    return ResidueStageResult(blocks=blocks, leap_inputs=inputs)


def template_mol2(
    template: AmberResidueTemplate, *, protein_forcefield: str, work_dir: Path
) -> Path:
    """An AmberTools template's unit as a mol2, with its Amber types and charges.

    For a program that reads a residue from a mol2 rather than from tleap's
    libraries, as MCPB.py does every residue it does not know. tleap writes
    it from the library itself, so it is the unit the build loads.
    """

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    mol2 = work / f"{template.name}.mol2"
    leaprc = template.leaprc(protein_forcefield)
    script = work / "tleap.savemol2.in"
    script.write_text(
        "\n".join(
            [
                f"source {protein_leaprc(protein_forcefield)}",
                *([f"source {leaprc}"] if leaprc else []),
                f"savemol2 {template.name} {mol2.name} 1",
                "quit",
                "",
            ]
        ),
        encoding="utf-8",
    )
    try:
        run = run_tleap(script, work_dir=work)
        assert_tleap_success(
            run.summary, fail_on_warnings=False, context=f"{template.name} template mol2"
        )
    except (TLeapRunError, LeapLogError) as exc:
        raise ResidueStageError(str(exc)) from exc
    if not mol2.is_file() or mol2.stat().st_size == 0:
        raise ResidueStageError(f"tleap did not write {mol2} from the {template.name} template.")
    return mol2


def block_residues(
    structure: PdbStructure, block: NoncanonicalResidueConfig
) -> list[ResidueRecord]:
    residues: list[ResidueRecord] = []
    name = final_resname(block)
    for selector in block.residues:
        try:
            residues.append(
                resolve_residue_selector(
                    structure, {**selector.model_dump(), "resname": name}
                )
            )
        except SelectorError as exc:
            raise ResidueStageError(
                f"noncanonical_residues {block.resname}: {exc}"
            ) from exc
    return residues


def write_stage_failure(output_dir: str | Path, exc: Exception) -> None:
    path = Path(output_dir) / "reports" / "residue_failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "failed", "exception_type": type(exc).__name__, "errors": [str(exc)]},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_residue_reports(
    result: ResidueStageResult, *, json_path: str | Path, markdown_path: str | Path
) -> dict[str, object]:
    report = result.to_report_dict()
    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(markdown_path).write_text(render_residue_markdown(result.blocks), encoding="utf-8")
    return report


def render_residue_markdown(blocks: Sequence[ResidueBlockResult]) -> str:
    lines = ["# Noncanonical Residue Report", ""]
    for block in blocks:
        lines.append(f"## {block.resname}")
        lines.append("")
        lines.append(f"- Residues: {', '.join(block.residues)}")
        if block.final_resname != block.resname:
            lines.append(f"- Built as: `{block.final_resname}` (renamed)")
        lines.append(f"- Net charge: {block.net_charge:+d}")
        if block.parameters == "amber_library":
            lines.append(
                f"- Parameters: AmberTools template `{block.template}` from "
                f"`{block.library}`"
                + (f" (`source {block.leaprc}`)" if block.leaprc else " (the protein force field)")
            )
        else:
            item = block.parameterised
            assert item is not None
            fit = block.qmmesp
            if block.parameters == "qmmesp_pyscf":
                lines.append(
                    f"- Parameters: QMMESP charges for the capped residue "
                    f"({item.parameterised_from}), fitted by RESP to its "
                    + (
                        f"{fit.method}/{fit.basis} ESP in {fit.point_charge_count} MM "
                        "point charges"
                        if fit is not None
                        else "PySCF ESP (not fitted yet)"
                    )
                    + f", {item.charges.backbone_set} backbone charges of "
                    f"{item.protein_forcefield} held; GAFF2 side chain"
                )
            else:
                lines.append(
                    f"- Parameters: AM1-BCC on the capped residue ({item.parameterised_from}), "
                    f"{item.charges.backbone_set} backbone charges of {item.protein_forcefield}, "
                    "GAFF2 side chain"
                )
            lines.append(f"- SMILES ({item.smiles_form}): `{item.smiles}`")
            if item.cmap:
                lines.append(f"- CMAP: {item.cmap}")
            files = fit.files if fit is not None else item.files
            lines.append(f"- Library: `{files.library_path}`")
            lines.append(f"- frcmod: `{files.frcmod_path}`")
            if fit is not None:
                lines.append(
                    f"- Embedding: {fit.point_charge_count} point charges, net "
                    f"{round(fit.net_embedding_charge, 3) or 0.0:+.3f} e; left out with "
                    f"the residue: {', '.join(fit.excluded_residues)}"
                )
                lines.append(
                    f"- RESP fit: RMS {fit.rms_error:.5f} au "
                    f"(relative {fit.relative_rms_error:.3f})"
                    if fit.relative_rms_error is not None
                    else f"- RESP fit: RMS {fit.rms_error:.5f} au"
                )
            lines.append("")
            if fit is not None:
                lines.append("| Atom | Type | AM1-BCC (provisional) | QMMESP (final) |")
                lines.append("| --- | --- | ---: | ---: |")
                for name, charge in fit.charges.charges.items():
                    lines.append(
                        f"| {name} | {item.atom_types[name]} | "
                        f"{item.charges.charges[name]:+.4f} | {charge:+.4f} |"
                    )
            else:
                heading = (
                    "Provisional" if block.parameters == "qmmesp_pyscf" else "Final"
                )
                lines.append(f"| Atom | Type | AM1-BCC | {heading} |")
                lines.append("| --- | --- | ---: | ---: |")
                for name, charge in item.charges.charges.items():
                    lines.append(
                        f"| {name} | {item.atom_types[name]} | "
                        f"{item.am1bcc_charges[name]:+.4f} | {charge:+.4f} |"
                    )
        if block.warnings:
            lines.append("")
            lines.extend(f"- Warning: {warning}" for warning in block.warnings)
        lines.append("")
    return "\n".join(lines)
