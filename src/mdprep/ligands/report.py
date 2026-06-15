"""Ligand-stage report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mdprep.ligands.workflow import LigandStageResult, LigandWorkflowItem


CSV_COLUMNS = [
    "ligand_id",
    "chain",
    "resname",
    "resid",
    "icode",
    "atom_count",
    "charge_method",
    "atom_types",
    "net_charge",
    "multiplicity",
    "final_mol2",
    "final_frcmod",
    "charge_sum_final",
    "coordinate_max_deviation",
    "status",
]


def write_ligand_reports(
    result: LigandStageResult,
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
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(result.ligands, csv_output)
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _write_csv(items: list[LigandWorkflowItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in items:
            data = item.to_dict()
            residue = data["residue_identity"]
            validation = data.get("validation") or {}
            writer.writerow(
                {
                    "ligand_id": data["ligand_id"],
                    "chain": residue["chain_id"],
                    "resname": residue["resname"],
                    "resid": residue["resid"],
                    "icode": residue["icode"],
                    "atom_count": data["atom_count"],
                    "charge_method": data["charge_method"],
                    "atom_types": data["atom_types"],
                    "net_charge": data["net_charge"],
                    "multiplicity": data["multiplicity"],
                    "final_mol2": data["final_mol2_path"],
                    "final_frcmod": data["final_frcmod_path"],
                    "charge_sum_final": validation.get("charge_sum_final"),
                    "coordinate_max_deviation": validation.get("coordinate_max_deviation"),
                    "status": data["status"],
                }
            )


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Ligand Report", ""]
    for item in report["ligands"]:
        lines.extend(
            [
                f"## {item['ligand_id']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Charge method: `{item['charge_method']}`",
                f"- Atom types: `{item['atom_types']}`",
                f"- Net charge: {item['net_charge']}",
                f"- Extracted PDB: `{item['extracted_pdb_path']}`",
                f"- Final mol2: `{item['final_mol2_path']}`",
                f"- Final frcmod: `{item['final_frcmod_path']}`",
            ]
        )
        if item.get("antechamber"):
            lines.append(f"- antechamber command: `{' '.join(item['antechamber']['command'])}`")
        if item.get("parmchk2"):
            lines.append(f"- parmchk2 command: `{' '.join(item['parmchk2']['command'])}`")
        if item.get("qm"):
            qm = item["qm"]
            lines.extend(
                [
                    "- PySCF charge derivation:",
                    f"  - Method: `{qm['method']}`",
                    f"  - Output directory: `{qm['qm_dir']}`",
                    f"  - Grid points: {qm['grid_point_count']}",
                    f"  - Fitted charge sum: {qm['fit_result']['charge_sum_final']}",
                ]
            )
            if qm["fit_result"].get("confirmation"):
                lines.append(f"  - Interpretation: {qm['fit_result']['confirmation']}")
            if qm.get("embedding_summary"):
                embedding = qm["embedding_summary"]
                lines.append(f"  - MM point charges: {embedding['point_charge_count_after_cutoff']}")
                lines.append(f"  - Target atom count: {embedding['target_atom_count']}")
        warnings = item.get("warnings") or []
        lines.append("- Warnings: " + ("; ".join(warnings) if warnings else "None"))
        lines.append("")
    return "\n".join(lines)
