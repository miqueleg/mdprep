"""tleap report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdprep.leap.builder import TLeapStageResult


def write_tleap_reports(
    result: TLeapStageResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    report = result.to_report_dict()
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# tleap Report",
        "",
        "## Force Fields",
        "",
    ]
    lines.extend(f"- `{source}`" for source in report["force_fields_sourced"])
    lines.extend(
        [
            "",
            "## Ligands",
            "",
        ]
    )
    ligands = report.get("ligands", [])
    if ligands:
        for ligand in ligands:
            lines.append(
                f"- `{ligand['ligand_id']}`: mol2 `{ligand['final_mol2_path']}`, "
                f"frcmod `{ligand['final_frcmod_path']}`"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Disulfide Bonds", ""])
    bonds = report.get("disulfide_bond_commands", [])
    if bonds:
        lines.extend(f"- `{bond['command']}`" for bond in bonds)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Dry prmtop: `{report['dry_outputs']['prmtop']}`",
            f"- Dry inpcrd: `{report['dry_outputs']['inpcrd']}`",
            f"- Final prmtop: `{report['final_outputs']['prmtop']}`",
            f"- Final inpcrd: `{report['final_outputs']['inpcrd']}`",
            f"- Final PDB: `{report['final_outputs']['pdb']}`",
            "",
            "## Warnings",
            "",
        ]
    )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)
