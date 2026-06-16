"""Starter manifest generation."""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure


def generate_starter_manifest(
    input_structure: str | Path,
    *,
    output_path: str | Path,
    overwrite: bool = False,
    forcefield: str = "ff19SB",
    water_model: str = "OPC",
    ph: float = 7.0,
    output_dir: str | None = None,
    protonation_method: str = "manual_only",
    include_ligand_placeholders: bool = False,
) -> Path:
    input_path = Path(input_structure)
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output manifest already exists: {target}. Use --overwrite to replace it.")

    summary = inspect_pdb_structure(input_path)
    project_name = input_path.stem
    resolved_output_dir = output_dir or f"prepared/{project_name}"
    manifest_text = _render_manifest(
        summary,
        project_name=project_name,
        input_structure=str(input_path),
        output_dir=resolved_output_dir,
        forcefield=forcefield,
        water_model=water_model,
        ph=ph,
        protonation_method=protonation_method,
        include_ligand_placeholders=include_ligand_placeholders,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(manifest_text, encoding="utf-8")
    return target


def _render_manifest(
    summary: InspectionSummary,
    *,
    project_name: str,
    input_structure: str,
    output_dir: str,
    forcefield: str,
    water_model: str,
    ph: float,
    protonation_method: str,
    include_ligand_placeholders: bool,
) -> str:
    lines: list[str] = [
        "# mdprep starter manifest",
        "# Review all chemistry-sensitive settings before running later preparation stages.",
    ]
    lines.extend(_comment_residues("Likely ligands/cofactors found", summary.likely_ligands))
    lines.extend(_comment_residues("Histidines found", summary.histidines))
    lines.extend(_comment_residues("Titratable residues found", summary.titratable_residues))
    if summary.possible_disulfides:
        lines.append("# Possible disulfides found:")
        for candidate in summary.possible_disulfides:
            lines.append(
                f"#   - {candidate.a.display()} -- {candidate.b.display()} "
                f"({candidate.distance_angstrom:.3f} A)"
            )
    else:
        lines.append("# Possible disulfides found: none")
    lines.extend(
        [
            "",
            "project:",
            f"  name: {project_name}",
            f"  input_structure: {input_structure}",
            f"  output_dir: {output_dir}",
            "",
            "structure:",
            "  keep_crystal_waters: true",
            "  altloc_policy: highest_occupancy",
            "  remove_unknown_heterogens: false",
            "  preserve_chain_ids: true",
            "  remove_input_hydrogens: true",
            "",
            "protein:",
            f"  forcefield: {forcefield}",
            f"  water_model: {water_model}",
            "",
            "protonation:",
            f"  ph: {ph}",
            f"  method: {protonation_method}",
            "  propka:",
            "    executable: null",
            "    fallback_executables:",
            "      - propka3",
            "      - propka",
            "    extra_args: []",
            "    require_success: true",
            "  overrides: []",
            "  histidine:",
            "    neutral_tautomer_method: xtb",
            "    xtb:",
            "      executable: xtb",
            "      model: gfn2",
            "      mode: opt",
            "      opt_level: loose",
            "      solvent: water",
            "      cutoff_angstrom: 5.0",
            "      extra_args: []",
            "      energy_close_call_kcal_mol: 0.5",
            "      add_missing_water_hydrogens: true",
            "      water_oh_distance_angstrom: 0.9572",
            "      water_hoh_angle_degrees: 104.52",
            "",
            "disulfides:",
            "  auto_detect: true",
            "  detection_cutoff_angstrom: 2.2",
            "  force: []",
            "  forbid: []",
            "",
            "ligands:",
        ]
    )
    if summary.likely_ligands and include_ligand_placeholders:
        lines.append("  # Placeholder ligand entries were generated from likely heterogens.")
        lines.append("  # Check every net_charge before running ligand parameterization.")
        for residue in summary.likely_ligands:
            ligand_id = f"{residue.id.resname.lower()}_{residue.id.resid}"
            lines.extend(
                [
                    f"  - id: {ligand_id}",
                    "    selector:",
                    f"      chain: {_yaml_string(residue.id.chain_id)}",
                    f"      resname: {residue.id.resname}",
                    f"      resid: {residue.id.resid}",
                    f"      icode: {_yaml_null_or_string(residue.id.icode)}",
                    "    net_charge: 0",
                    "    multiplicity: 1",
                    "    atom_types: gaff2",
                    "    charge_method: am1bcc",
                    "    user_mol2: null",
                    "    user_frcmod: null",
                    "    preserve_atom_names: true",
                    "    preserve_coordinates: true",
                    "    allow_atom_renaming: false",
                    "    allow_coordinate_changes: false",
                    "    qmmesp: null",
                ]
            )
    else:
        lines.append("  []")
        lines.extend(_comment_ligand_placeholders(summary.likely_ligands))
    lines.extend(
        [
            "",
            "solvation:",
            "  enabled: true",
            "  box: truncated_octahedron",
            "  buffer_angstrom: 10.0",
            "  neutralize: true",
            "  salt_concentration_molar: 0.15",
            "  positive_ion: Na+",
            "  negative_ion: Cl-",
            "",
            "validation:",
            "  run_openmm_energy_check: true",
            "  fail_on_warnings: false",
            "  fail_on_missing_parameters: true",
            "  fail_on_noninteger_ligand_charge: true",
            "",
        ]
    )
    return "\n".join(lines)


def _comment_residues(title: str, residues: list[object]) -> list[str]:
    if not residues:
        return [f"# {title}: none"]
    lines = [f"# {title}:"]
    for residue in residues:
        residue_id = residue.id  # type: ignore[attr-defined]
        lines.append(f"#   - {residue_id.display()}")
    return lines


def _comment_ligand_placeholders(residues: list[object]) -> list[str]:
    if not residues:
        return [
            "# Example ligand block:",
            "#   - id: sub_501",
            "#     selector: {chain: B, resname: SUB, resid: 501, icode: null}",
            "#     net_charge: 0  # CHECK THIS VALUE",
            "#     multiplicity: 1",
            "#     atom_types: gaff2",
            "#     charge_method: am1bcc",
            "#     user_mol2: null",
            "#     user_frcmod: null",
        ]
    lines = [
        "# Detected ligand placeholder suggestions are commented out by default.",
        "# Re-run with --include-ligand-placeholders to make them active.",
        "# Check every net_charge before ligand parameterization.",
    ]
    for residue in residues:
        ligand_id = f"{residue.id.resname.lower()}_{residue.id.resid}"  # type: ignore[attr-defined]
        residue_id = residue.id  # type: ignore[attr-defined]
        lines.extend(
            [
                f"#   - id: {ligand_id}",
                f"#     selector: {{chain: {_yaml_string(residue_id.chain_id)}, resname: {residue_id.resname}, resid: {residue_id.resid}, icode: {_yaml_null_or_string(residue_id.icode)}}}",
                "#     net_charge: 0  # CHECK THIS VALUE",
                "#     multiplicity: 1",
                "#     atom_types: gaff2",
                "#     charge_method: am1bcc",
                "#     user_mol2: null",
                "#     user_frcmod: null",
            ]
        )
    return lines


def _yaml_string(value: str) -> str:
    if value == "":
        return '""'
    if any(char in value for char in [":", "#", "{", "}", "[", "]", ",", " "]):
        return f'"{value}"'
    return value


def _yaml_null_or_string(value: str | None) -> str:
    return "null" if value is None else _yaml_string(value)
