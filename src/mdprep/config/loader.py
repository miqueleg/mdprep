"""YAML manifest loading."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

import yaml

from mdprep.config.models import ManifestConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a YAML mapping at the top level")
    return data


def load_manifest(
    path: str | Path,
    *,
    resolve_paths: bool = False,
) -> ManifestConfig:
    """Load a manifest, optionally resolving file paths from its directory.

    Schema validation keeps the portable user-authored strings unchanged.
    Execution workflows request resolution so their behavior is independent of
    the shell's current working directory.
    """

    manifest_path = Path(path)
    data = load_yaml(manifest_path)
    if resolve_paths:
        data = _resolve_manifest_paths(data, manifest_path.parent.resolve())
    return ManifestConfig.model_validate(data)


def _resolve_manifest_paths(data: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(data)

    def file_path(value: object, *, existing_cwd_fallback: bool = True) -> object:
        if not isinstance(value, str) or not value:
            return value
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        manifest_candidate = (base_dir / candidate).resolve()
        cwd_candidate = candidate.resolve()
        if (
            existing_cwd_fallback
            and not manifest_candidate.exists()
            and cwd_candidate.exists()
        ):
            return str(cwd_candidate)
        return str(manifest_candidate)

    def command_path(value: object) -> object:
        if not isinstance(value, str) or not value:
            return value
        candidate = Path(value).expanduser()
        if candidate.is_absolute() or "/" in value or "\\" in value:
            return str(candidate if candidate.is_absolute() else (base_dir / candidate).resolve())
        return value

    project = resolved.get("project")
    if isinstance(project, dict):
        if "input_structure" in project:
            project["input_structure"] = file_path(project["input_structure"])
        if "output_dir" in project:
            project["output_dir"] = file_path(
                project["output_dir"], existing_cwd_fallback=False
            )

    structure = resolved.get("structure")
    repair = structure.get("repair") if isinstance(structure, dict) else None
    if isinstance(repair, dict) and "sequence_source" in repair:
        repair["sequence_source"] = file_path(repair["sequence_source"])

    protonation = resolved.get("protonation")
    if isinstance(protonation, dict):
        propka = protonation.get("propka")
        if isinstance(propka, dict) and "executable" in propka:
            propka["executable"] = command_path(propka["executable"])
        histidine = protonation.get("histidine")
        xtb = histidine.get("xtb") if isinstance(histidine, dict) else None
        if isinstance(xtb, dict) and "executable" in xtb:
            xtb["executable"] = command_path(xtb["executable"])

    ligands = resolved.get("ligands")
    if isinstance(ligands, list):
        for ligand in ligands:
            if not isinstance(ligand, dict):
                continue
            for key in ("user_mol2", "user_frcmod"):
                if key in ligand:
                    ligand[key] = file_path(ligand[key])

    metals = resolved.get("metals")
    if isinstance(metals, list):
        for site in metals:
            mcpb = site.get("mcpb") if isinstance(site, dict) else None
            if not isinstance(mcpb, dict):
                continue
            if "executable" in mcpb:
                mcpb["executable"] = command_path(mcpb["executable"])
            artifacts = mcpb.get("artifacts")
            if isinstance(artifacts, dict):
                for key in ("small_opt_fchk", "small_fc_log", "large_mk_log"):
                    if key in artifacts:
                        artifacts[key] = file_path(artifacts[key])
            pyscf = mcpb.get("pyscf")
            if isinstance(pyscf, dict):
                for key in ("optimized_small_model_pdb", "large_model_restart_checkpoint"):
                    if key in pyscf:
                        pyscf[key] = file_path(pyscf[key])
                for backend_key in ("xtb", "gxtb"):
                    backend = pyscf.get(backend_key)
                    if isinstance(backend, dict) and "executable" in backend:
                        backend["executable"] = command_path(backend["executable"])
                mace = pyscf.get("mace_polar1")
                if isinstance(mace, dict) and "python_executable" in mace:
                    mace["python_executable"] = command_path(mace["python_executable"])
            comparison = mcpb.get("parameter_comparison")
            if isinstance(comparison, dict) and "reference_frcmod" in comparison:
                comparison["reference_frcmod"] = file_path(comparison["reference_frcmod"])

    refinement = resolved.get("refinement")
    if isinstance(refinement, dict):
        ash = refinement.get("ash")
        if isinstance(ash, dict):
            for key in ("python_executable", "xtb_executable"):
                if key in ash:
                    ash[key] = command_path(ash[key])
        mace = refinement.get("mace_polar1")
        if isinstance(mace, dict):
            if "model_path" in mace:
                mace["model_path"] = file_path(mace["model_path"])
            if isinstance(mace.get("python_search_paths"), list):
                mace["python_search_paths"] = [
                    file_path(value) for value in mace["python_search_paths"]
                ]
    return resolved
