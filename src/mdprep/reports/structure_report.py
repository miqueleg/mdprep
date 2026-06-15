"""Structure-stage report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdprep.structure.normalize import StructureNormalizationResult


def write_structure_reports(
    result: StructureNormalizationResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    report = result.to_report_dict()
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Structure Normalization Report",
        "",
        f"- Input: `{report['input_path']}`",
        f"- Normalized PDB: `{report['output_normalized_pdb_path']}`",
        f"- Atoms: {report['atom_count_before']} -> {report['atom_count_after']}",
        f"- Residues: {report['residue_count_before']} -> {report['residue_count_after']}",
        "",
        "## Waters",
        "",
        f"- Kept: {len(report['waters_kept'])}",
        f"- Removed: {len(report['waters_removed'])}",
        "",
        "## Configured Ligands",
        "",
    ]
    lines.extend(_residue_lines(report["configured_ligands_kept"]))
    lines.extend(["", "## Unknown Heterogens Removed", ""])
    lines.extend(_residue_lines(report["unknown_heterogens_removed"]))
    lines.extend(["", "## Histidines", ""])
    lines.extend(_residue_lines(report["histidines"]))
    lines.extend(["", "## Titratable Residues", ""])
    lines.extend(_residue_lines(report["titratable_residues"]))
    lines.extend(["", "## Possible Disulfides", ""])
    if report["possible_disulfides"]:
        for item in report["possible_disulfides"]:
            a = item["a"]
            b = item["b"]
            lines.append(
                f"- {_format_residue(a)} -- {_format_residue(b)}: {item['distance_angstrom']:.3f} A"
            )
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _residue_lines(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- None"]
    lines = []
    for item in items:
        prefix = f"{item['id']}: " if "id" in item else ""
        lines.append(f"- {prefix}{_format_residue(item)} ({item['atom_count']} atoms)")
    return lines


def _format_residue(item: dict[str, Any]) -> str:
    chain = item["chain_id"] or "<blank>"
    return f"{chain}:{item['resname']}{item['resid']}{item.get('icode') or ''}"

