"""JSON and Markdown reporting for QM/MM refinement."""

from __future__ import annotations

import json
from pathlib import Path

from mdprep.refinement.workflow import RefinementResult


def write_refinement_reports(
    result: RefinementResult,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, object]:
    report = result.to_report_dict()
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selection = report["selection"]
    assert isinstance(selection, dict)
    ash = report["ash"]
    assert isinstance(ash, dict)
    lines = [
        "# QM/MM refinement report",
        "",
        "- Status: complete",
        "- Backend: ASH",
        f"- QM method: {report['qm_method']}",
        "- MM theory: provisional Amber/OpenMM",
        f"- Embedding: {report['embedding']}",
        "- Provisional ligand/cofactor charges: AM1-BCC",
        f"- QM atoms: {len(selection['qm_atom_indices_zero_based'])}",
        f"- Active atoms: {len(selection['active_atom_indices_zero_based'])}",
        f"- Movable-atom policy: {selection['movable_atom_policy']}",
        f"- Active waters: {report['active_water_residue_count']}",
        f"- Non-water active-region cutoff (Å): {selection['active_region_cutoff_angstrom']}",
        f"- Active-water cutoff (Å): {selection['active_water_cutoff_angstrom']}",
        f"- Total QM charge: {selection['total_qm_charge']}",
        f"- Total QM multiplicity: {selection['total_qm_multiplicity']}",
        f"- Final energy (Eh): {ash['final_energy_hartree']}",
        f"- Refined hydrogenated PDB: `{report['refined_hydrogenated_pdb_path']}`",
        "",
        "## QM charge derivation",
        "",
    ]
    for component in selection["charge_components"]:
        lines.append(
            f"- {component['label']}: {component['charge']:+d} ({component['kind']})"
        )
    lines.extend(["", "## External command", ""])
    lines.append(f"- Command: `{' '.join(ash['command'])}`")
    lines.append(f"- Working directory: `{ash['cwd']}`")
    lines.append(f"- Return code: {ash['returncode']}")
    lines.append(f"- Runtime (s): {ash['runtime_seconds']}")
    lines.extend(["", "## Notes", ""])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
