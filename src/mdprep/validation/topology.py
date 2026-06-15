"""Final Amber output validation and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdprep.config.models import ManifestConfig
from mdprep.structure.classify import is_water_residue, likely_ligands_or_cofactors
from mdprep.structure.pdb import PdbParseError, read_pdb
from mdprep.validation.openmm_check import run_openmm_energy_check
from mdprep.validation.parmed_check import run_parmed_check


class FinalValidationError(ValueError):
    """Raised when final Amber outputs fail required validation."""


def validate_final_outputs(
    *,
    manifest: ManifestConfig,
    prmtop: str | Path,
    inpcrd: str | Path,
    pdb: str | Path,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    paths = {
        "prmtop": Path(prmtop),
        "inpcrd": Path(inpcrd),
        "pdb": Path(pdb),
    }
    file_checks = {}
    for label, path in paths.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        file_checks[label] = {"path": str(path), "exists": exists, "size_bytes": size}
        if not exists:
            errors.append(f"Missing final {label}: {path}")
        elif size == 0:
            errors.append(f"Final {label} is empty: {path}")

    final_structure = None
    if paths["pdb"].exists() and paths["pdb"].stat().st_size > 0:
        try:
            final_structure = read_pdb(paths["pdb"])
        except (PdbParseError, ValueError) as exc:
            errors.append(f"Final PDB is not parseable by mdprep: {exc}")

    ligand_checks: list[dict[str, object]] = []
    if final_structure is not None:
        for ligand in manifest.ligands:
            expected = ligand.selector.resname
            residues = [residue for residue in final_structure.residues if residue.id.resname == expected]
            expected_names = _expected_ligand_atom_names(ligand.id, paths["pdb"].parent.parent)
            atom_names_ok = bool(residues)
            if expected_names:
                atom_names_ok = any(residue.atom_names() == expected_names for residue in residues)
            ok = bool(residues) and atom_names_ok
            ligand_checks.append(
                {
                    "ligand_id": ligand.id,
                    "resname": expected,
                    "present": bool(residues),
                    "atom_names_ok": atom_names_ok,
                    "ok": ok,
                }
            )
            if not ok:
                errors.append(f"Configured ligand {ligand.id} ({expected}) was not preserved in final PDB.")

        water_count = sum(1 for residue in final_structure.residues if is_water_residue(residue))
        if manifest.solvation.enabled and water_count == 0:
            errors.append("Solvation is enabled but no waters were found in final PDB.")
        if not manifest.solvation.enabled and water_count > 0 and not manifest.structure.keep_crystal_waters:
            errors.append("Solvation is disabled but unexpected waters were found in final PDB.")

        unexpected = [
            residue.id.to_dict()
            for residue in likely_ligands_or_cofactors(final_structure.residues)
            if residue.id.resname not in {ligand.selector.resname for ligand in manifest.ligands}
        ]
        if unexpected:
            errors.append(f"Unexpected heterogen residues found in final PDB: {unexpected}")
    else:
        water_count = 0

    parmed = run_parmed_check(paths["prmtop"], paths["inpcrd"])
    if parmed.get("status") == "skipped":
        warnings.append(str(parmed.get("warning")))
    elif parmed.get("status") == "error":
        errors.append(f"ParmEd validation failed: {parmed.get('error')}")
    elif parmed.get("charge_near_integer") is False:
        errors.append("ParmEd total charge is not near an integer.")

    if manifest.validation.run_openmm_energy_check:
        openmm = run_openmm_energy_check(paths["prmtop"], paths["inpcrd"])
        if openmm.get("status") == "skipped":
            warnings.append(str(openmm.get("warning")))
        elif openmm.get("status") == "error":
            warnings.append(f"OpenMM energy check failed: {openmm.get('error')}")
        elif openmm.get("finite") is False:
            errors.append("OpenMM potential energy is not finite.")
    else:
        openmm = {"available": None, "status": "disabled"}

    report = {
        "final_prmtop_path": str(paths["prmtop"]),
        "final_inpcrd_path": str(paths["inpcrd"]),
        "final_pdb_path": str(paths["pdb"]),
        "file_checks": file_checks,
        "final_atom_count": len(final_structure.atoms) if final_structure is not None else None,
        "final_residue_count": len(final_structure.residues) if final_structure is not None else None,
        "ligand_presence_checks": ligand_checks,
        "water_presence": {"water_count": water_count, "solvation_enabled": manifest.solvation.enabled},
        "parmed": parmed,
        "openmm": openmm,
        "warnings": warnings,
        "errors": errors,
    }
    if errors:
        raise FinalValidationError("; ".join(errors))
    return report


def write_validation_reports(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_output.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _expected_ligand_atom_names(ligand_id: str, output_dir: Path) -> list[str]:
    identity = output_dir / "ligands" / ligand_id / "input" / "identity.json"
    if not identity.exists():
        return []
    try:
        data = json.loads(identity.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    names = data.get("atom_names", [])
    return [str(name) for name in names] if isinstance(names, list) else []


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Validation Report",
        "",
        f"- Final atom count: {report['final_atom_count']}",
        f"- Final residue count: {report['final_residue_count']}",
        f"- ParmEd status: `{report['parmed'].get('status')}`",
        f"- OpenMM status: `{report['openmm'].get('status')}`",
        "",
        "## Ligands",
        "",
    ]
    ligand_checks = report.get("ligand_presence_checks", [])
    if ligand_checks:
        for check in ligand_checks:
            lines.append(f"- `{check['ligand_id']}` present={check['present']} atom_names_ok={check['atom_names_ok']}")
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)
