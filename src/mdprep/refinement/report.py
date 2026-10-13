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
    # The force-field backend is the default and reports no ASH run, so the
    # fields that only a QM/MM run produces -- the quantum method, the
    # embedding, the external command -- have to be read from whichever
    # backend actually ran.
    backend = report["backend"]
    ash = report.get("ash")
    openmm_run = report.get("openmm")

    lines = [
        "# Hydrogen refinement report",
        "",
        "- Status: complete",
        f"- Backend: {'ASH QM/MM' if backend == 'ash' else 'OpenMM force field'}",
    ]
    if backend == "ash":
        assert isinstance(ash, dict)
        lines.extend(
            [
                f"- QM method: {report['qm_method']}",
                "- MM theory: provisional Amber/OpenMM",
                f"- Embedding: {report['embedding']}",
            ]
        )
    else:
        assert isinstance(openmm_run, dict)
        lines.extend(
            [
                "- Theory: provisional Amber force field, no quantum step",
                f"- Moved: {report['moved']}",
                f"- Platform: {openmm_run['platform']}",
            ]
        )
    lines.extend(
        [
            "- Provisional ligand/cofactor charges: AM1-BCC",
            f"- QM atoms: {len(selection['qm_atom_indices_zero_based'])}",
            f"- Active atoms: {len(selection['active_atom_indices_zero_based'])}",
            f"- Movable-atom policy: {selection['movable_atom_policy']}",
            f"- Active waters: {report['active_water_residue_count']}",
            f"- Non-water active-region cutoff (Å): {selection['active_region_cutoff_angstrom']}",
            f"- Active-water cutoff (Å): {selection['active_water_cutoff_angstrom']}",
            f"- Total QM charge: {selection['total_qm_charge']}",
            f"- Total QM multiplicity: {selection['total_qm_multiplicity']}",
        ]
    )
    if backend == "ash":
        assert isinstance(ash, dict)
        lines.append(f"- Final energy (Eh): {ash['final_energy_hartree']}")
    else:
        assert isinstance(openmm_run, dict)
        lines.extend(
            [
                f"- Hydrogens refined: {openmm_run['hydrogen_count']}",
                f"- Heavy atoms held: {openmm_run['held_heavy_atoms']}",
                f"- Largest heavy-atom shift (Å): {openmm_run['max_heavy_atom_shift_angstrom']}",
                f"- Hydrogen RMS displacement (Å): {openmm_run['hydrogen_rms_displacement_angstrom']}",
                f"- Hydrogen largest displacement (Å): {openmm_run['hydrogen_max_displacement_angstrom']}",
                f"- Energy (kJ/mol): {openmm_run['initial_energy_kj_mol']} -> {openmm_run['final_energy_kj_mol']}",
            ]
        )
    lines.extend(
        [
            f"- Refined hydrogenated PDB: `{report['refined_hydrogenated_pdb_path']}`",
            "",
            "## QM charge derivation",
            "",
        ]
    )
    for component in selection["charge_components"]:
        lines.append(
            f"- {component['label']}: {component['charge']:+d} ({component['kind']})"
        )
    if backend == "ash":
        assert isinstance(ash, dict)
        lines.extend(["", "## External command", ""])
        lines.append(f"- Command: `{' '.join(ash['command'])}`")
        lines.append(f"- Working directory: `{ash['cwd']}`")
        lines.append(f"- Return code: {ash['returncode']}")
        lines.append(f"- Runtime (s): {ash['runtime_seconds']}")
    else:
        assert isinstance(openmm_run, dict)
        lines.extend(["", "## Run", ""])
        lines.append(f"- Runtime (s): {openmm_run['seconds']}")
        lines.append(f"- Atoms: {openmm_run['atom_count']}")
    lines.extend(["", "## Notes", ""])
    for warning in report["warnings"]:
        lines.append(f"- {warning}")
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
