"""Protonation-stage report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mdprep.protonation.apply import ProtonationRecord, ProtonationResult


CSV_COLUMNS = [
    "chain",
    "resid",
    "icode",
    "original_resname",
    "final_resname",
    "source",
    "pka",
    "ph",
    "reason",
    "changed",
]


def write_protonation_reports(
    result: ProtonationResult,
    *,
    json_path: str | Path,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    report = result.to_report_dict()
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(result.records, csv_output)
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _write_csv(records: list[ProtonationRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            data = record.to_dict()
            writer.writerow({column: data[column] for column in CSV_COLUMNS})


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Protonation Report",
        "",
        "## Summary",
        "",
        f"- Method: `{report['method']}`",
        f"- pH: {report['ph']}",
        f"- Hydrogens removed: {report['hydrogen_atoms_removed']}",
        f"- Residues changed: {len(report['residues_changed'])}",
        f"- Explicit unchanged assignments: {len(report['residues_unchanged_but_explicitly_assigned'])}",
        "",
        "## Manual Overrides",
        "",
    ]
    lines.extend(_record_lines(report["manual_overrides_applied"]))
    lines.extend(["", "## Disulfide Assignments", ""])
    lines.extend(_record_lines(report["disulfide_assignments_applied"]))
    lines.extend(["", "## PropKa Assignments", ""])
    lines.extend(_record_lines(report.get("propka_assignments_applied", [])))
    lines.extend(["", "## Input States Preserved", ""])
    lines.extend(_record_lines(report.get("input_state_assignments_applied", [])))
    lines.extend(["", "## xTB Histidine Selections", ""])
    lines.extend(_xtb_lines(report.get("xtb_histidines", [])))
    lines.extend(["", "## Hydrogens Removed", ""])
    lines.append(f"- {report['hydrogen_atoms_removed']}")
    lines.extend(["", "## Remaining HIS Residues", ""])
    lines.extend(_residue_lines(report["unresolved_histidines_remaining_as_his"]))
    lines.extend(["", "## Unassigned Titratable Residues", ""])
    lines.extend(_residue_lines(report["titratable_residues_not_explicitly_assigned"]))
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _record_lines(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["- None"]
    return [
        f"- {_format_record(record)} via `{record['source']}`: {record['reason']}"
        for record in records
    ]


def _residue_lines(residues: list[dict[str, Any]]) -> list[str]:
    if not residues:
        return ["- None"]
    return [f"- {_format_residue(residue)}" for residue in residues]


def _xtb_lines(selections: list[dict[str, Any]]) -> list[str]:
    if not selections:
        return ["- None"]
    lines: list[str] = []
    for selection in selections:
        chain = selection["chain"] or "<blank>"
        icode = selection["icode"] or ""
        lines.append(
            f"- {chain}:HIS{selection['resid']}{icode} -> {selection['selected_state']} "
            f"(delta HID-HIE {selection['delta_kcal_mol']:.3f} kcal/mol, "
            f"model `{selection['model']}`, mode `{selection['mode']}`)"
        )
    return lines


def _format_record(record: dict[str, Any]) -> str:
    chain = record["chain"] or "<blank>"
    icode = record["icode"] or ""
    return (
        f"{chain}:{record['original_resname']}{record['resid']}{icode} "
        f"-> {record['final_resname']}"
    )


def _format_residue(residue: dict[str, Any]) -> str:
    chain = residue["chain_id"] or "<blank>"
    icode = residue["icode"] or ""
    return f"{chain}:{residue['resname']}{residue['resid']}{icode}"
