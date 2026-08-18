"""Compare MCPB.py Seminario bond/angle terms with a reviewed reference."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Iterable

from mdprep.config.models import McpbParameterComparisonConfig


class McpbParameterComparisonError(ValueError):
    """Raised when MCPB parameter files cannot be compared safely."""


@dataclass(frozen=True)
class McpbParameterTerm:
    section: str
    atom_types: tuple[str, ...]
    force_constant: float
    equilibrium_value: float
    source_line: str

    @property
    def key(self) -> str:
        return f"{self.section}:{'-'.join(self.atom_types)}"


@dataclass(frozen=True)
class McpbTermDifference:
    section: str
    atom_types: tuple[str, ...]
    candidate_force_constant: float
    reference_force_constant: float
    force_constant_difference: float
    force_constant_relative_difference: float | None
    candidate_equilibrium_value: float
    reference_equilibrium_value: float
    equilibrium_value_difference: float

    @property
    def key(self) -> str:
        return f"{self.section}:{'-'.join(self.atom_types)}"

    def to_dict(self) -> dict[str, object]:
        force_units = (
            "kcal mol^-1 angstrom^-2"
            if self.section == "BOND"
            else "kcal mol^-1 radian^-2"
        )
        equilibrium_units = "angstrom" if self.section == "BOND" else "degrees"
        return {
            "key": self.key,
            "section": self.section,
            "atom_types": list(self.atom_types),
            "candidate_force_constant": self.candidate_force_constant,
            "reference_force_constant": self.reference_force_constant,
            "force_constant_difference": self.force_constant_difference,
            "force_constant_relative_difference": (
                self.force_constant_relative_difference
            ),
            "candidate_equilibrium_value": self.candidate_equilibrium_value,
            "reference_equilibrium_value": self.reference_equilibrium_value,
            "equilibrium_value_difference": self.equilibrium_value_difference,
            "force_constant_units": force_units,
            "equilibrium_value_units": equilibrium_units,
        }


@dataclass(frozen=True)
class McpbParameterComparisonResult:
    candidate_frcmod: Path
    reference_frcmod: Path
    candidate_label: str
    reference_label: str
    differences: tuple[McpbTermDifference, ...]
    missing_from_candidate: tuple[str, ...]
    missing_from_reference: tuple[str, ...]
    nonpositive_candidate_force_constants: tuple[str, ...]
    nonpositive_reference_force_constants: tuple[str, ...]
    metrics: dict[str, dict[str, float | int | None]]
    thresholds: dict[str, float | None]
    violations: tuple[str, ...]
    json_path: Path
    csv_path: Path
    markdown_path: Path

    @property
    def passed(self) -> bool:
        return not self.violations

    @property
    def has_numeric_thresholds(self) -> bool:
        return any(value is not None for value in self.thresholds.values())

    @property
    def status(self) -> str:
        if self.violations:
            return "fail"
        return "pass" if self.has_numeric_thresholds else "report_only"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "candidate": {
                "label": self.candidate_label,
                "frcmod": str(self.candidate_frcmod),
                "sha256": _sha256(self.candidate_frcmod),
            },
            "reference": {
                "label": self.reference_label,
                "frcmod": str(self.reference_frcmod),
                "sha256": _sha256(self.reference_frcmod),
            },
            "scope": (
                "BOND and ANGL entries explicitly marked as created by the "
                "Seminario method using MCPB.py"
            ),
            "matched_term_count": len(self.differences),
            "missing_from_candidate": list(self.missing_from_candidate),
            "missing_from_reference": list(self.missing_from_reference),
            "nonpositive_candidate_force_constants": list(
                self.nonpositive_candidate_force_constants
            ),
            "nonpositive_reference_force_constants": list(
                self.nonpositive_reference_force_constants
            ),
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "violations": list(self.violations),
            "terms": [item.to_dict() for item in self.differences],
            "json_path": str(self.json_path),
            "csv_path": str(self.csv_path),
            "markdown_path": str(self.markdown_path),
            "interpretation": (
                "Agreement with the selected reference measures method sensitivity for "
                "this specific geometry and electronic state. It does not by itself "
                "validate the force field for production MD."
            ),
        }


def parse_mcpb_seminario_terms(
    path: str | Path,
    *,
    allow_nonpositive_force_constants: bool = False,
) -> dict[str, McpbParameterTerm]:
    """Parse the MCPB-created BOND/ANGL terms from an Amber frcmod file."""

    frcmod = Path(path)
    if not frcmod.is_file() or frcmod.stat().st_size == 0:
        raise McpbParameterComparisonError(
            f"MCPB parameter file is missing or empty: {frcmod}"
        )
    section: str | None = None
    terms: dict[str, McpbParameterTerm] = {}
    headings = {"MASS", "BOND", "ANGL", "DIHE", "IMPR", "HBON", "NONB"}
    for line_number, raw in enumerate(
        frcmod.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if stripped in headings:
            section = stripped
            continue
        if section not in {"BOND", "ANGL"}:
            continue
        if "seminario method using mcpb.py" not in raw.lower():
            continue
        term = _parse_term_line(
            raw,
            section=section,
            path=frcmod,
            line_number=line_number,
            allow_nonpositive_force_constants=allow_nonpositive_force_constants,
        )
        if term.key in terms:
            raise McpbParameterComparisonError(
                f"Duplicate MCPB Seminario term {term.key} in {frcmod}"
            )
        terms[term.key] = term
    if not terms:
        raise McpbParameterComparisonError(
            f"No MCPB.py Seminario-created BOND or ANGL terms were found in {frcmod}"
        )
    return terms


def compare_mcpb_parameter_files(
    candidate_frcmod: str | Path,
    *,
    config: McpbParameterComparisonConfig,
    output_dir: str | Path,
) -> McpbParameterComparisonResult:
    """Compare, report, and optionally enforce user-supplied acceptance limits."""

    candidate_path = Path(candidate_frcmod).resolve()
    reference_path = Path(config.reference_frcmod).resolve()
    if candidate_path == reference_path:
        raise McpbParameterComparisonError(
            "Candidate and reference frcmod paths resolve to the same file"
        )
    candidate = parse_mcpb_seminario_terms(
        candidate_path,
        allow_nonpositive_force_constants=True,
    )
    reference = parse_mcpb_seminario_terms(
        reference_path,
        allow_nonpositive_force_constants=True,
    )
    candidate_keys = set(candidate)
    reference_keys = set(reference)
    missing_candidate = tuple(sorted(reference_keys - candidate_keys))
    missing_reference = tuple(sorted(candidate_keys - reference_keys))
    nonpositive_candidate = tuple(
        sorted(key for key, term in candidate.items() if term.force_constant <= 0.0)
    )
    nonpositive_reference = tuple(
        sorted(key for key, term in reference.items() if term.force_constant <= 0.0)
    )
    differences = tuple(
        _difference(candidate[key], reference[key])
        for key in sorted(candidate_keys & reference_keys)
    )
    if not differences:
        raise McpbParameterComparisonError(
            "Candidate and reference contain no matching MCPB Seminario terms"
        )
    metrics = {
        "bonds": _metrics(item for item in differences if item.section == "BOND"),
        "angles": _metrics(item for item in differences if item.section == "ANGL"),
    }
    thresholds: dict[str, float | None] = {
        "max_bond_force_constant_relative_rmse": (
            config.max_bond_force_constant_relative_rmse
        ),
        "max_angle_force_constant_relative_rmse": (
            config.max_angle_force_constant_relative_rmse
        ),
        "max_bond_equilibrium_distance_rmse_angstrom": (
            config.max_bond_equilibrium_distance_rmse_angstrom
        ),
        "max_angle_equilibrium_value_rmse_degrees": (
            config.max_angle_equilibrium_value_rmse_degrees
        ),
    }
    violations: list[str] = []
    if nonpositive_candidate:
        violations.append(
            "Candidate contains non-positive MCPB Seminario force constants: "
            f"{list(nonpositive_candidate)}"
        )
    if nonpositive_reference:
        violations.append(
            "Reference contains non-positive MCPB Seminario force constants: "
            f"{list(nonpositive_reference)}"
        )
    if config.require_exact_term_set and (missing_candidate or missing_reference):
        violations.append(
            "Candidate/reference MCPB Seminario term sets differ: "
            f"missing from candidate={list(missing_candidate)}, "
            f"missing from reference={list(missing_reference)}"
        )
    _check_threshold(
        violations,
        label="bond force-constant relative RMSE",
        value=metrics["bonds"]["force_constant_relative_rmse"],
        limit=config.max_bond_force_constant_relative_rmse,
    )
    _check_threshold(
        violations,
        label="angle force-constant relative RMSE",
        value=metrics["angles"]["force_constant_relative_rmse"],
        limit=config.max_angle_force_constant_relative_rmse,
    )
    _check_threshold(
        violations,
        label="bond equilibrium-distance RMSE (angstrom)",
        value=metrics["bonds"]["equilibrium_value_rmse"],
        limit=config.max_bond_equilibrium_distance_rmse_angstrom,
    )
    _check_threshold(
        violations,
        label="angle equilibrium-value RMSE (degrees)",
        value=metrics["angles"]["equilibrium_value_rmse"],
        limit=config.max_angle_equilibrium_value_rmse_degrees,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = McpbParameterComparisonResult(
        candidate_frcmod=candidate_path,
        reference_frcmod=reference_path,
        candidate_label=config.candidate_label,
        reference_label=config.reference_label,
        differences=differences,
        missing_from_candidate=missing_candidate,
        missing_from_reference=missing_reference,
        nonpositive_candidate_force_constants=nonpositive_candidate,
        nonpositive_reference_force_constants=nonpositive_reference,
        metrics=metrics,
        thresholds=thresholds,
        violations=tuple(violations),
        json_path=output / "mcpb_parameter_comparison.json",
        csv_path=output / "mcpb_parameter_comparison.csv",
        markdown_path=output / "mcpb_parameter_comparison.md",
    )
    _write_reports(result)
    if violations and config.fail_on_thresholds:
        raise McpbParameterComparisonError(
            "MCPB parameter comparison failed configured acceptance criteria. "
            f"See {result.json_path}. Violations: " + "; ".join(violations)
        )
    return result


def _parse_term_line(
    raw: str,
    *,
    section: str,
    path: Path,
    line_number: int,
    allow_nonpositive_force_constants: bool,
) -> McpbParameterTerm:
    atom_count = 2 if section == "BOND" else 3
    minimum_width = 5 if section == "BOND" else 8
    if len(raw) < minimum_width:
        raise McpbParameterComparisonError(
            f"Malformed {section} term at {path}:{line_number}: {raw!r}"
        )
    if section == "BOND":
        raw_types = (raw[0:2].strip(), raw[3:5].strip())
        values = raw[5:].split()
    else:
        raw_types = (raw[0:2].strip(), raw[3:5].strip(), raw[6:8].strip())
        values = raw[8:].split()
    if len(raw_types) != atom_count or any(not value for value in raw_types):
        raise McpbParameterComparisonError(
            f"Malformed atom types in {section} term at {path}:{line_number}: {raw!r}"
        )
    try:
        force_constant = float(values[0])
        equilibrium = float(values[1])
    except (IndexError, ValueError) as exc:
        raise McpbParameterComparisonError(
            f"Malformed numeric values in {section} term at {path}:{line_number}: {raw!r}"
        ) from exc
    if not isfinite(force_constant) or (
        force_constant <= 0 and not allow_nonpositive_force_constants
    ):
        raise McpbParameterComparisonError(
            f"Non-positive or non-finite force constant in {section} term at "
            f"{path}:{line_number}: {force_constant}"
        )
    equilibrium_valid = equilibrium > 0 and (
        section == "BOND" or equilibrium <= 180.0
    )
    if not isfinite(equilibrium) or not equilibrium_valid:
        raise McpbParameterComparisonError(
            f"Invalid equilibrium value in {section} term at {path}:{line_number}: "
            f"{equilibrium}"
        )
    atom_types = _canonical_atom_types(section, raw_types)
    return McpbParameterTerm(
        section=section,
        atom_types=atom_types,
        force_constant=force_constant,
        equilibrium_value=equilibrium,
        source_line=raw,
    )


def _canonical_atom_types(section: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if section == "BOND":
        return tuple(sorted(values))
    forward = values
    reverse = tuple(reversed(values))
    return min(forward, reverse)


def _difference(
    candidate: McpbParameterTerm,
    reference: McpbParameterTerm,
) -> McpbTermDifference:
    force_delta = candidate.force_constant - reference.force_constant
    relative = (
        force_delta / abs(reference.force_constant)
        if reference.force_constant != 0.0
        else None
    )
    return McpbTermDifference(
        section=candidate.section,
        atom_types=candidate.atom_types,
        candidate_force_constant=candidate.force_constant,
        reference_force_constant=reference.force_constant,
        force_constant_difference=force_delta,
        force_constant_relative_difference=relative,
        candidate_equilibrium_value=candidate.equilibrium_value,
        reference_equilibrium_value=reference.equilibrium_value,
        equilibrium_value_difference=(
            candidate.equilibrium_value - reference.equilibrium_value
        ),
    )


def _metrics(
    items: Iterable[McpbTermDifference],
) -> dict[str, float | int | None]:
    values = list(items)
    if not values:
        return {
            "count": 0,
            "force_constant_relative_count": 0,
            "force_constant_mae": None,
            "force_constant_rmse": None,
            "force_constant_relative_mae": None,
            "force_constant_relative_rmse": None,
            "force_constant_max_absolute_relative_difference": None,
            "equilibrium_value_mae": None,
            "equilibrium_value_rmse": None,
            "equilibrium_value_max_absolute_difference": None,
        }
    force_differences = [item.force_constant_difference for item in values]
    relative_differences = [
        item.force_constant_relative_difference
        for item in values
        if item.force_constant_relative_difference is not None
    ]
    equilibrium_differences = [
        item.equilibrium_value_difference for item in values
    ]
    return {
        "count": len(values),
        "force_constant_relative_count": len(relative_differences),
        "force_constant_mae": _mae(force_differences),
        "force_constant_rmse": _rmse(force_differences),
        "force_constant_relative_mae": _mae(relative_differences),
        "force_constant_relative_rmse": _rmse(relative_differences),
        "force_constant_max_absolute_relative_difference": max(
            (abs(value) for value in relative_differences), default=None
        ),
        "equilibrium_value_mae": _mae(equilibrium_differences),
        "equilibrium_value_rmse": _rmse(equilibrium_differences),
        "equilibrium_value_max_absolute_difference": max(
            abs(value) for value in equilibrium_differences
        ),
    }


def _mae(values: list[float]) -> float | None:
    return sum(abs(value) for value in values) / len(values) if values else None


def _rmse(values: list[float]) -> float | None:
    return sqrt(sum(value * value for value in values) / len(values)) if values else None


def _check_threshold(
    violations: list[str],
    *,
    label: str,
    value: float | int | None,
    limit: float | None,
) -> None:
    if limit is None:
        return
    if value is None:
        violations.append(f"{label} could not be evaluated; configured limit is {limit}")
    elif float(value) > limit:
        violations.append(f"{label} {float(value):.8g} exceeds configured limit {limit:.8g}")


def _write_reports(result: McpbParameterComparisonResult) -> None:
    data = result.to_dict()
    result.json_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with result.csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(result.differences[0].to_dict())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.differences:
            row = item.to_dict()
            row["atom_types"] = "-".join(item.atom_types)
            writer.writerow(row)
    lines = [
        "# MCPB parameter comparison",
        "",
        f"- Status: {result.status.upper()}",
        f"- Candidate: `{result.candidate_label}` (`{result.candidate_frcmod}`)",
        f"- Reference: `{result.reference_label}` (`{result.reference_frcmod}`)",
        f"- Matched terms: {len(result.differences)}",
        "- Scope: MCPB.py Seminario-created BOND and ANGL terms only",
        "",
        "## Metrics",
        "",
    ]
    for section, metrics in result.metrics.items():
        lines.extend(
            [
                f"### {section.title()}",
                "",
                f"- Count: {metrics['count']}",
                f"- Relative force-constant count: "
                f"{metrics['force_constant_relative_count']}",
                f"- Force-constant relative RMSE: {metrics['force_constant_relative_rmse']}",
                f"- Force-constant maximum absolute relative difference: "
                f"{metrics['force_constant_max_absolute_relative_difference']}",
                f"- Equilibrium-value RMSE: {metrics['equilibrium_value_rmse']}",
                "",
            ]
        )
    lines.extend(["## Term-by-term differences", ""])
    lines.append(
        "| Term | Candidate K | Reference K | Relative K difference | "
        "Candidate equilibrium | Reference equilibrium | Difference |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in result.differences:
        lines.append(
            f"| `{item.key}` | {item.candidate_force_constant:.8g} | "
            f"{item.reference_force_constant:.8g} | "
            f"{item.force_constant_relative_difference} | "
            f"{item.candidate_equilibrium_value:.8g} | "
            f"{item.reference_equilibrium_value:.8g} | "
            f"{item.equilibrium_value_difference:.8g} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Agreement with the selected reference measures method sensitivity for this "
        "exact geometry and electronic state. It does not by itself validate "
        "production MD."
    )
    if (
        result.nonpositive_candidate_force_constants
        or result.nonpositive_reference_force_constants
    ):
        lines.extend(["", "## Invalid bonded terms", ""])
        if result.nonpositive_candidate_force_constants:
            lines.append(
                "- Candidate non-positive force constants: "
                + ", ".join(result.nonpositive_candidate_force_constants)
            )
        if result.nonpositive_reference_force_constants:
            lines.append(
                "- Reference non-positive force constants: "
                + ", ".join(result.nonpositive_reference_force_constants)
            )
    if result.violations:
        lines.extend(["", "## Violations", ""])
        lines.extend(f"- {item}" for item in result.violations)
    result.markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
