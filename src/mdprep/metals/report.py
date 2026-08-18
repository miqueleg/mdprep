"""Metal-stage report writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdprep.metals.workflow import MetalStageResult


def write_metal_reports(
    result: MetalStageResult,
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
        "# Metal Preparation Report",
        "",
        f"- Complete: {report['complete']}",
        f"- Configured sites: {len(report['sites'])}",
        "",
        "## Nonbonded Sites",
        "",
    ]
    if not report["nonbonded_sites"]:
        lines.append("- None")
    for site in report["nonbonded_sites"]:
        lines.append(f"- `{site['site_id']}`: `{site['parameter_set']}`")
        for ion in site["ions"]:
            parameter = ion["parameter"]
            selector = ion["selector"]
            lines.append(
                f"  - {selector['chain'] or '<blank>'}:{selector['resname']}"
                f"{selector['resid']}@{selector['atom_name']}: {parameter['atom_type']}, "
                f"Rmin/2={parameter['rmin_over_2_angstrom']:.6f} A, "
                f"epsilon={parameter['epsilon_kcal_mol']:.8f} kcal/mol"
            )
    lines.extend(["", "## Bonded MCPB.py Site", ""])
    bonded = report["bonded_mcpb_site"]
    if bonded is None:
        lines.append("- None")
    else:
        lines.extend(
            [
                f"- Site: `{bonded['site_id']}`",
                f"- Workflow: `{bonded['workflow']}`",
                f"- Complete: {bonded['complete']}",
                f"- Work directory: `{bonded['work_dir']}`",
                f"- MCPB.py commands run: {len(bonded['runs'])}",
                f"- Residues renamed by MCPB.py: {len(bonded['residue_renames'])}",
            ]
        )
        if not bonded["complete"]:
            lines.append(
                "- Required QM artifacts for a `complete` rerun: "
                + ", ".join(f"`{name}`" for name in bonded["expected_qm_artifacts"])
            )
        refitted_ligands = bonded.get("refitted_ligands", [])
        lines.extend(["", "### Joint RESP-fitted ligands", ""])
        if refitted_ligands:
            for ligand in refitted_ligands:
                lines.append(
                    f"- `{ligand['ligand_id']}` -> `{ligand['final_resname']}`: "
                    f"mol2 `{ligand['final_mol2_path']}`, fragment charge "
                    f"{ligand['fragment_charge']:.6f} e"
                )
        else:
            lines.append("- None")
        comparison = bonded["parameter_comparison"]
        if comparison is not None:
            lines.extend(
                [
                    "",
                    "### MCPB parameter comparison",
                    "",
                    f"- Status: `{comparison['status']}`",
                    f"- Matched Seminario terms: {comparison['matched_term_count']}",
                    "- Non-positive candidate force constants: "
                    f"{len(comparison['nonpositive_candidate_force_constants'])}",
                    "- Non-positive reference force constants: "
                    f"{len(comparison['nonpositive_reference_force_constants'])}",
                    f"- Detailed report: `{comparison['json_path']}`",
                ]
            )
    lines.extend(["", "## Pre-MCPB Hydrogenation", ""])
    hydrogenation = report["pre_mcpb_hydrogenation"]
    if hydrogenation is None:
        lines.append("- None")
    else:
        lines.extend(
            [
                f"- Hydrogenated PDB: `{hydrogenation['restored_output_pdb_path']}`",
                f"- tleap script: `{hydrogenation['script_path']}`",
                f"- Metal-donor protonation checks: {len(hydrogenation['donor_protonation_checks'])}",
            ]
        )
    lines.extend(["", "## 12-6-4 Atom Types", ""])
    lines.extend(f"- `{atom_type}`" for atom_type in report["c4_atom_types"])
    if not report["c4_atom_types"]:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report["warnings"])
    if not report["warnings"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)
