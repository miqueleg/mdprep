"""Preparation workflow."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

import yaml

from mdprep import __version__
from mdprep.config.loader import load_manifest
from mdprep.reports.structure_report import write_structure_reports
from mdprep.structure.normalize import StructureNormalizationResult, normalize_structure_stage
from mdprep.structure.writer import write_pdb


class PrepareWorkflowError(ValueError):
    """Raised when the requested preparation workflow is not available."""


def prepare_system(
    manifest_path: str | Path,
    *,
    stop_after: str | None = None,
    overwrite: bool = False,
) -> StructureNormalizationResult:
    if stop_after != "structure":
        raise PrepareWorkflowError(
            "Full Amber preparation is not implemented yet; use --stop-after structure for the currently supported workflow."
        )

    manifest_file = Path(manifest_path)
    manifest = load_manifest(manifest_file)
    output_dir = Path(manifest.project.output_dir)
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace mdprep outputs.")

    intermediate_dir = output_dir / "intermediate"
    reports_dir = output_dir / "reports"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_input_path = output_dir / "manifest.input.yaml"
    shutil.copyfile(manifest_file, manifest_input_path)

    normalized_pdb = intermediate_dir / "00_input_normalized.pdb"
    result = normalize_structure_stage(manifest, output_path=normalized_pdb)
    write_pdb(result.normalized_structure, normalized_pdb)
    result.output_path = normalized_pdb

    report = write_structure_reports(
        result,
        json_path=reports_dir / "structure_report.json",
        markdown_path=reports_dir / "structure_report.md",
    )
    _write_manifest_lock(
        manifest=manifest,
        report=report,
        path=output_dir / "manifest.lock.yaml",
    )
    _write_versions(output_dir / "versions.json")
    return result


def _write_manifest_lock(*, manifest: object, report: dict[str, object], path: Path) -> None:
    manifest_data = manifest.model_dump(mode="json")  # type: ignore[attr-defined]
    lock_data = {
        "mdprep_version": __version__,
        "manifest": manifest_data,
        "resolved": {
            "input_structure": manifest_data["project"]["input_structure"],
            "selected_altloc_policy": manifest_data["structure"]["altloc_policy"],
            "keep_crystal_waters": manifest_data["structure"]["keep_crystal_waters"],
            "remove_unknown_heterogens": manifest_data["structure"]["remove_unknown_heterogens"],
            "configured_ligand_selectors": [
                {"id": ligand["id"], "selector": ligand["selector"]}
                for ligand in manifest_data.get("ligands", [])
            ],
            "unknown_heterogens_removed": report["unknown_heterogens_removed"],
            "unknown_heterogens_causing_failure": report["unknown_heterogens_causing_failure"],
            "possible_disulfides": report["possible_disulfides"],
        },
    }
    path.write_text(yaml.safe_dump(lock_data, sort_keys=False), encoding="utf-8")


def _write_versions(path: Path) -> None:
    versions = {
        "mdprep": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "external_tools_required": [],
    }
    path.write_text(json.dumps(versions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
