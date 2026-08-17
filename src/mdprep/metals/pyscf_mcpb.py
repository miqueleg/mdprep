"""Fixed-geometry QM artifacts consumed by MCPB.py.

The small-model geometry comes either from an explicit external PDB or from a
completed ASH QM/MM refinement. mdprep evaluates the Cartesian Hessian required
by Seminario at those fixed coordinates with PySCF, GFN2-xTB, g-xTB, or
MACE-POLAR-1. The
large MCPB model is evaluated with a fixed-geometry, electrostatically embedded
PySCF SCF. Only the QM density is sampled on the ESP grid: environment charges
polarize that density but are never fitted as charge centers.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from mdprep.charges.esp_grid import (
    EspGrid,
    generate_merz_kollman_grid,
    write_grid_xyz,
)
from mdprep.config.models import McpbPySCFConfig
from mdprep.metals.coordination import ResolvedMetalSite
from mdprep.qm.point_charges import PointCharge
from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM, evaluate_ligand_esp
from mdprep.qm.pyscf_runner import (
    PySCFResult,
    PySCFRunnerError,
    pyscf_version,
    run_pyscf_analytic_hessian,
    run_pyscf_scf,
)
from mdprep.structure.classify import STANDARD_PROTEIN_RESIDUES, WATER_RESIDUES
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.pdb import infer_element, read_pdb
from mdprep.structure.writer import write_pdb

if TYPE_CHECKING:
    from mdprep.refinement.workflow import RefinementResult


class McpbPySCFError(ValueError):
    """Raised when PySCF cannot produce validated MCPB.py QM artifacts."""


@dataclass(frozen=True)
class McpbEmbeddingSelection:
    point_charges: tuple[PointCharge, ...]
    excluded_qm_residue_count: int
    candidate_atom_count: int
    excluded_by_minimum_distance: int
    excluded_by_cutoff: int
    minimum_distance_angstrom: float | None
    maximum_distance_angstrom: float | None
    net_embedding_charge: float
    categories: dict[str, int]

    @property
    def charge_array(self) -> np.ndarray:
        return np.asarray([item.charge for item in self.point_charges], dtype=float)

    @property
    def coordinate_array(self) -> np.ndarray:
        return np.asarray(
            [[item.x, item.y, item.z] for item in self.point_charges],
            dtype=float,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "point_charge_count": len(self.point_charges),
            "excluded_qm_residue_count": self.excluded_qm_residue_count,
            "candidate_atom_count": self.candidate_atom_count,
            "excluded_by_minimum_distance": self.excluded_by_minimum_distance,
            "excluded_by_cutoff": self.excluded_by_cutoff,
            "minimum_distance_angstrom": self.minimum_distance_angstrom,
            "maximum_distance_angstrom": self.maximum_distance_angstrom,
            "net_embedding_charge": self.net_embedding_charge,
            "categories": self.categories,
            "embedding_operator": "pyscf.qmmm.mm_charge",
            "interpretation": (
                "These MM charges polarize the MCPB large-model QM density. They are "
                "not RESP centers and are not written to MCPB ligand/residue mol2 files."
            ),
        }


@dataclass(frozen=True)
class SmallModelResult:
    mcpb_generated_pdb_path: Path
    user_optimized_pdb_path: Path
    qmmm_refinement_source_pdb_path: Path | None
    evaluated_pdb_path: Path
    evaluated_xyz_path: Path
    checkpoint_path: Path | None
    fchk_path: Path
    calculation_report_path: Path
    scf_stdout_path: Path
    scf_stderr_path: Path
    atom_count: int
    energy_hartree: float
    geometry_status: str
    scientific_warning: str | None
    geometry_optimization_performed: bool
    scf_calculation_count: int | None
    hessian_antisymmetry_max: float
    hessian_backend: str
    backend_version: str
    scf_runtime_seconds: float | None
    hessian_runtime_seconds: float
    total_runtime_seconds: float
    external_command_records: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mcpb_generated_pdb_path": str(self.mcpb_generated_pdb_path),
            "user_optimized_pdb_path": str(self.user_optimized_pdb_path),
            "qmmm_refinement_source_pdb_path": (
                str(self.qmmm_refinement_source_pdb_path)
                if self.qmmm_refinement_source_pdb_path is not None
                else None
            ),
            "evaluated_pdb_path": str(self.evaluated_pdb_path),
            "evaluated_xyz_path": str(self.evaluated_xyz_path),
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path is not None else None
            ),
            "fchk_path": str(self.fchk_path),
            "calculation_report_path": str(self.calculation_report_path),
            "scf_stdout_path": str(self.scf_stdout_path),
            "scf_stderr_path": str(self.scf_stderr_path),
            "atom_count": self.atom_count,
            "energy_hartree": self.energy_hartree,
            "geometry_status": self.geometry_status,
            "scientific_warning": self.scientific_warning,
            "geometry_optimization_performed": self.geometry_optimization_performed,
            "scf_calculation_count": self.scf_calculation_count,
            "hessian_antisymmetry_max": self.hessian_antisymmetry_max,
            "hessian_backend": self.hessian_backend,
            "backend_version": self.backend_version,
            "scf_runtime_seconds": self.scf_runtime_seconds,
            "hessian_runtime_seconds": self.hessian_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
            "external_commands": list(self.external_command_records),
            "mcpb_generated_pdb_sha256": _sha256(self.mcpb_generated_pdb_path),
            "user_optimized_pdb_sha256": _sha256(self.user_optimized_pdb_path),
            "qmmm_refinement_source_pdb_sha256": (
                _sha256(self.qmmm_refinement_source_pdb_path)
                if self.qmmm_refinement_source_pdb_path is not None
                else None
            ),
            "fchk_sha256": _sha256(self.fchk_path),
        }


@dataclass(frozen=True)
class LargeModelResult:
    input_pdb_path: Path
    checkpoint_path: Path
    mk_log_path: Path
    grid_xyz_path: Path
    esp_values_path: Path
    point_charge_csv_path: Path
    point_charge_xyz_path: Path
    point_charge_report_path: Path
    scf_stdout_path: Path
    scf_stderr_path: Path
    atom_count: int
    energy_hartree: float
    grid_point_count: int
    embedding: McpbEmbeddingSelection
    scf_algorithm: str
    initial_guess: str
    initial_density_checkpoint_path: str | None
    initial_density_checkpoint_sha256: str | None
    initial_density_checkpoint_max_displacement_angstrom: float | None
    initial_density_checkpoint_rms_displacement_angstrom: float | None
    initial_density_checkpoint_projected: bool
    adiis_precondition_cycles: int | None
    adiis_precondition_dft_grid_level: int | None
    precondition_cycles_completed: int
    level_shift_mode: str
    level_shift_hartree: float
    damping_factor: float
    diis_space: int
    density_fitting: bool
    auxiliary_basis: str | None
    spin_square: float | None
    effective_multiplicity: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_pdb_path": str(self.input_pdb_path),
            "checkpoint_path": str(self.checkpoint_path),
            "mk_log_path": str(self.mk_log_path),
            "grid_xyz_path": str(self.grid_xyz_path),
            "esp_values_path": str(self.esp_values_path),
            "point_charge_csv_path": str(self.point_charge_csv_path),
            "point_charge_xyz_path": str(self.point_charge_xyz_path),
            "point_charge_report_path": str(self.point_charge_report_path),
            "scf_stdout_path": str(self.scf_stdout_path),
            "scf_stderr_path": str(self.scf_stderr_path),
            "atom_count": self.atom_count,
            "energy_hartree": self.energy_hartree,
            "grid_point_count": self.grid_point_count,
            "embedding": self.embedding.to_dict(),
            "scf_algorithm": self.scf_algorithm,
            "initial_guess": self.initial_guess,
            "initial_density_checkpoint_path": self.initial_density_checkpoint_path,
            "initial_density_checkpoint_sha256": (
                self.initial_density_checkpoint_sha256
            ),
            "initial_density_checkpoint_max_displacement_angstrom": (
                self.initial_density_checkpoint_max_displacement_angstrom
            ),
            "initial_density_checkpoint_rms_displacement_angstrom": (
                self.initial_density_checkpoint_rms_displacement_angstrom
            ),
            "initial_density_checkpoint_projected": (
                self.initial_density_checkpoint_projected
            ),
            "adiis_precondition_cycles": self.adiis_precondition_cycles,
            "adiis_precondition_dft_grid_level": (
                self.adiis_precondition_dft_grid_level
            ),
            "precondition_cycles_completed": self.precondition_cycles_completed,
            "level_shift_mode": self.level_shift_mode,
            "level_shift_hartree": self.level_shift_hartree,
            "damping_factor": self.damping_factor,
            "diis_space": self.diis_space,
            "density_fitting": self.density_fitting,
            "auxiliary_basis": self.auxiliary_basis,
            "spin_square": self.spin_square,
            "effective_multiplicity": self.effective_multiplicity,
            "mk_log_sha256": _sha256(self.mk_log_path),
        }


@dataclass(frozen=True)
class McpbPySCFResult:
    method: str
    basis: str
    dft_grid_level: int
    small_model_charge: int
    small_model_multiplicity: int
    large_model_charge: int
    large_model_multiplicity: int
    small: SmallModelResult
    large: LargeModelResult
    report_path: Path

    def to_dict(self) -> dict[str, object]:
        geometry_description = {
            "user_optimized": "user-supplied optimized MCPB small-model geometry",
            "qmmm_refined": "completed ASH/GFN2-xTB QM/MM-refined geometry",
            "test_unoptimized": (
                "explicitly unoptimized MCPB small-model geometry for workflow testing"
            ),
        }[self.small.geometry_status]
        hessian_description = {
            "pyscf": "PySCF analytic Hessian",
            "gfn2_xtb": "GFN2-xTB numerical Hessian from analytic gradients",
            "gxtb": "g-xTB numerical Hessian from analytic gradients",
            "mace_polar1": "MACE-POLAR-1 analytical Hessian",
        }[self.small.hessian_backend]
        backend = (
            "PySCF"
            if self.small.hessian_backend == "pyscf"
            else f"PySCF+{self.small.hessian_backend}"
        )
        return {
            "backend": backend,
            "method": self.method,
            "basis": self.basis,
            "dft_grid_level": self.dft_grid_level,
            "small_model_hessian_backend": self.small.hessian_backend,
            "large_model_esp_backend": "PySCF",
            "small_model_charge": self.small_model_charge,
            "small_model_multiplicity": self.small_model_multiplicity,
            "large_model_charge": self.large_model_charge,
            "large_model_multiplicity": self.large_model_multiplicity,
            "small_model": self.small.to_dict(),
            "large_model": self.large.to_dict(),
            "report_path": str(self.report_path),
            "scientific_warning": self.small.scientific_warning,
            "scientific_workflow": (
                f"{geometry_description}; one fixed-geometry "
                f"small-model {hessian_description}; MCPB large-model "
                "electrostatically embedded fixed-geometry single-point SCF; "
                "QM-only ESP passed to MCPB.py two-stage RESP."
            ),
        }


def run_mcpb_pyscf(
    site: ResolvedMetalSite,
    *,
    group_name: str,
    work_dir: str | Path,
    provisional_prmtop: str | Path,
    provisional_inpcrd: str | Path,
    provisional_structure: PdbStructure,
    refinement_result: "RefinementResult | None" = None,
) -> McpbPySCFResult:
    """Run and validate both PySCF calculations needed by one MCPB site."""

    if site.config.mcpb is None or site.config.mcpb.pyscf is None:
        raise McpbPySCFError(
            f"Metal site {site.config.id!r} has no mcpb.pyscf configuration."
        )
    mcpb = site.config.mcpb
    config = mcpb.pyscf
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    small_pdb = work / f"{group_name}_small.pdb"
    large_pdb = work / f"{group_name}_large.pdb"
    for path in (small_pdb, large_pdb):
        if not path.is_file() or path.stat().st_size == 0:
            raise McpbPySCFError(f"MCPB.py step 1 did not produce model PDB: {path}")

    qmmm_refinement_source: Path | None = None
    if config.geometry_source == "qmmm_refinement":
        if refinement_result is None:
            raise McpbPySCFError(
                f"Metal site {site.config.id!r} requests geometry_source: "
                "qmmm_refinement, but no completed refinement result reached the metal stage."
            )
        qmmm_refinement_source = refinement_result.refined_hydrogenated_pdb_path
        if not qmmm_refinement_source.is_file():
            raise McpbPySCFError(
                "The completed refinement PDB recorded for MCPB geometry provenance is "
                f"missing: {qmmm_refinement_source}"
            )
    small = _run_small_model(
        small_pdb,
        output_fchk=work / f"{group_name}_small_opt.fchk",
        charge=mcpb.small_model_charge,
        multiplicity=mcpb.small_model_spin,
        config=config,
        user_optimized_pdb=(
            Path(config.optimized_small_model_pdb)
            if config.optimized_small_model_pdb is not None
            else None
        ),
        qmmm_refinement_source_pdb=qmmm_refinement_source,
        work_dir=work / "pyscf_small",
    )
    large = _run_large_model(
        site,
        large_pdb,
        output_mk_log=work / f"{group_name}_large_mk.log",
        charge=mcpb.large_model_charge,
        multiplicity=mcpb.large_model_spin,
        config=config,
        provisional_prmtop=Path(provisional_prmtop),
        provisional_inpcrd=Path(provisional_inpcrd),
        provisional_structure=provisional_structure,
        work_dir=work / "pyscf_large",
    )
    report_path = work / "pyscf_mcpb_report.json"
    result = McpbPySCFResult(
        method=config.method,
        basis=config.basis,
        dft_grid_level=config.dft_grid_level,
        small_model_charge=mcpb.small_model_charge,
        small_model_multiplicity=mcpb.small_model_spin,
        large_model_charge=mcpb.large_model_charge,
        large_model_multiplicity=mcpb.large_model_spin,
        small=small,
        large=large,
        report_path=report_path,
    )
    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _run_small_model(
    input_pdb: Path,
    *,
    output_fchk: Path,
    charge: int,
    multiplicity: int,
    config: McpbPySCFConfig,
    user_optimized_pdb: Path | None,
    qmmm_refinement_source_pdb: Path | None = None,
    work_dir: Path,
) -> SmallModelResult:
    generated_structure = read_pdb(input_pdb)
    generated_elements, _ = _elements_and_coordinates(generated_structure)
    if config.geometry_source == "qmmm_refinement":
        if qmmm_refinement_source_pdb is None:
            raise McpbPySCFError(
                "geometry_source: qmmm_refinement requires completed refinement provenance"
            )
        optimized_path = input_pdb.resolve()
        _, optimized_coordinates = _elements_and_coordinates(generated_structure)
        geometry_status = "qmmm_refined"
        geometry_label = "QM/MM-refined MCPB geometry"
        evaluated_stem = "small_qmmm_refined_fixed_geometry"
    else:
        if user_optimized_pdb is None:
            raise McpbPySCFError(
                "geometry_source: external_pdb requires optimized_small_model_pdb"
            )
        optimized_path = user_optimized_pdb.resolve()
        if optimized_path == input_pdb.resolve():
            raise McpbPySCFError(
                "mcpb.pyscf.optimized_small_model_pdb points to MCPB.py's unoptimized "
                f"step-1 model ({input_pdb}). Supply a separate, user-optimized PDB."
            )
        if not optimized_path.is_file() or optimized_path.stat().st_size == 0:
            raise McpbPySCFError(
                "The required user-optimized MCPB small-model geometry does not exist or is "
                f"empty: {optimized_path}. Optimize {input_pdb} externally without changing "
                "atom order or identity, set mcpb.pyscf.optimized_small_model_pdb to that "
                "PDB, and rerun."
            )
        optimized_structure = read_pdb(optimized_path)
        optimized_elements, optimized_coordinates = _elements_and_coordinates(
            optimized_structure
        )
        _validate_user_optimized_small_model(
            generated_structure,
            generated_elements=generated_elements,
            user_structure=optimized_structure,
            user_elements=optimized_elements,
        )
        assert config.small_model_geometry_status is not None
        geometry_status = config.small_model_geometry_status
        geometry_label = "MCPB user-supplied fixed geometry"
        evaluated_stem = "small_user_optimized_fixed_geometry"
    scientific_warning = (
        "TEST ONLY [UNOPTIMIZED_MCPB_GEOMETRY]: the small-model coordinates were "
        "explicitly accepted without external geometry optimization. Derived bonded "
        "parameters and the final topology are not production-quality."
        if geometry_status == "test_unoptimized"
        else None
    )
    spin = multiplicity - 1
    work_dir.mkdir(parents=True, exist_ok=True)

    evaluated_structure = _structure_with_coordinates(
        generated_structure,
        optimized_coordinates,
        path=work_dir / f"{evaluated_stem}.pdb",
    )
    write_pdb(evaluated_structure, evaluated_structure.path)
    evaluated_xyz = work_dir / f"{evaluated_stem}.xyz"
    _write_xyz(
        generated_elements,
        optimized_coordinates,
        evaluated_xyz,
        geometry_label,
    )

    total_started = perf_counter()
    checkpoint: Path | None
    scf_runtime: float | None
    external_command_records: tuple[dict[str, object], ...]
    if config.hessian_backend == "pyscf":
        checkpoint = work_dir / "small_fixed_geometry.chk"
        scf_started = perf_counter()
        try:
            scf = run_pyscf_scf(
                elements=generated_elements,
                coordinates=optimized_coordinates,
                charge=charge,
                spin=spin,
                method=config.method,
                basis=config.basis,
                max_cycle=config.max_cycle,
                conv_tol=config.conv_tol,
                checkpoint_path=checkpoint,
                work_dir=work_dir,
                num_threads=config.num_threads,
                max_memory_mb=config.max_memory_mb,
                dft_grid_level=config.dft_grid_level,
                scf_algorithm=config.scf_algorithm,
                initial_guess=config.initial_guess,
                adiis_precondition_cycles=config.adiis_precondition_cycles,
                adiis_precondition_dft_grid_level=(
                    config.adiis_precondition_dft_grid_level
                ),
                level_shift_mode=config.level_shift_mode,
                level_shift_hartree=config.level_shift_hartree,
                damping_factor=config.damping_factor,
                diis_space=config.diis_space,
            )
        except PySCFRunnerError as exc:
            raise McpbPySCFError(
                f"Fixed-geometry MCPB small-model SCF failed: {exc}"
            ) from exc
        scf_runtime = perf_counter() - scf_started
        stdout_path = work_dir / "small_fixed_geometry_scf.stdout.txt"
        stderr_path = work_dir / "small_fixed_geometry_scf.stderr.txt"
        stdout_path.write_text(scf.stdout, encoding="utf-8")
        stderr_path.write_text(scf.stderr, encoding="utf-8")
        (work_dir / "small_hessian_progress.json").write_text(
            json.dumps(
                {
                    "hessian_backend": "pyscf",
                    "backend_version": pyscf_version(),
                    "method": config.method,
                    "basis": config.basis,
                    "fixed_geometry_scf_complete": True,
                    "energy_hartree": scf.energy_hartree,
                    "scf_runtime_seconds": scf_runtime,
                    "analytic_hessian_complete": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        hessian_started = perf_counter()
        try:
            hessian_raw = run_pyscf_analytic_hessian(
                scf.mf,
                num_threads=config.num_threads,
            )
        except PySCFRunnerError as exc:
            raise McpbPySCFError(
                f"PySCF analytic Hessian failed for {config.method}/{config.basis}: {exc}"
            ) from exc
        hessian_runtime = perf_counter() - hessian_started
        energy_hartree = scf.energy_hartree
        backend_version = pyscf_version()
        scf_calculation_count: int | None = 1
        external_command_records = ()
        calculation_name = "fixed_geometry_single_point_scf_and_analytic_hessian"
        hessian_note = (
            "The PySCF analytic Hessian is required for Seminario force constants; "
            "all nuclei remained fixed at the selected coordinates."
        )
        hessian_backend = "pyscf"
    elif config.hessian_backend in {"xtb", "gxtb"}:
        xtb_config = config.xtb if config.hessian_backend == "xtb" else config.gxtb
        assert xtb_config is not None
        from mdprep.metals.gxtb_hessian import GxtbHessianError, run_gxtb_hessian

        try:
            gxtb_result = run_gxtb_hessian(
                xyz_path=evaluated_xyz,
                atom_count=len(generated_elements),
                charge=charge,
                spin=spin,
                config=xtb_config,
                work_dir=work_dir,
            )
        except GxtbHessianError as exc:
            raise McpbPySCFError(f"xTB fixed-geometry Hessian failed: {exc}") from exc
        if (
            geometry_status != "test_unoptimized"
            and gxtb_result.raw_hessian_asymmetry_warning_count > 0
        ):
            raise McpbPySCFError(
                "xTB reported "
                f"{gxtb_result.raw_hessian_asymmetry_warning_count} raw numerical "
                "Hessian-asymmetry warnings for a production geometry. Tighten "
                "the xTB accuracy or provide a better-converged geometry."
            )
        checkpoint = None
        scf_runtime = None
        hessian_runtime = gxtb_result.command_result.runtime_seconds
        hessian_raw = gxtb_result.hessian_hartree_per_bohr2
        energy_hartree = gxtb_result.energy_hartree
        backend_version = (
            f"{xtb_config.release_tag}; {gxtb_result.version_text}"
            if xtb_config.release_tag
            else gxtb_result.version_text
        )
        scf_calculation_count = None
        stdout_path = gxtb_result.stdout_path
        stderr_path = gxtb_result.stderr_path
        gxtb_metadata = gxtb_result.to_dict()
        external_command_records = (
            dict(gxtb_metadata["version_command"]),  # type: ignore[arg-type]
            dict(gxtb_metadata["hessian_command"]),  # type: ignore[arg-type]
        )
        hessian_backend = "gfn2_xtb" if xtb_config.model == "gfn2" else "gxtb"
        calculation_name = f"fixed_geometry_{hessian_backend}_numerical_hessian"
        hessian_note = (
            f"The {hessian_backend} Hessian was evaluated at the selected coordinates by "
            "numerical differentiation of analytic gradients; no geometry "
            "optimization was performed."
        )
    else:
        mace_config = config.mace_polar1
        assert mace_config is not None
        from mdprep.metals.mace_polar1_hessian import (
            MacePolar1HessianError,
            run_mace_polar1_hessian,
        )

        try:
            mace_result = run_mace_polar1_hessian(
                xyz_path=evaluated_xyz,
                atom_count=len(generated_elements),
                charge=charge,
                multiplicity=multiplicity,
                config=mace_config,
                work_dir=work_dir,
            )
        except MacePolar1HessianError as exc:
            raise McpbPySCFError(
                f"MACE-POLAR-1 fixed-geometry Hessian failed: {exc}"
            ) from exc
        checkpoint = None
        scf_runtime = None
        hessian_runtime = mace_result.command_result.runtime_seconds
        hessian_raw = mace_result.hessian_hartree_per_bohr2
        energy_hartree = mace_result.energy_hartree
        mace_version = mace_result.versions.get("mace_torch") or "unknown"
        backend_version = f"mace-torch {mace_version}; {mace_result.model}"
        scf_calculation_count = None
        stdout_path = mace_result.stdout_path
        stderr_path = mace_result.stderr_path
        mace_metadata = mace_result.to_dict()
        external_command_records = (
            dict(mace_metadata["probe_command"]),  # type: ignore[arg-type]
            dict(mace_metadata["hessian_command"]),  # type: ignore[arg-type]
        )
        hessian_backend = "mace_polar1"
        calculation_name = "fixed_geometry_mace_polar1_analytical_hessian"
        hessian_note = (
            "The float64 MACE-POLAR-1 analytical Hessian was evaluated at the "
            "selected charge, multiplicity, and coordinates and checked against a "
            "deterministic central difference of MACE forces; no geometry optimization "
            "was performed."
        )

    hessian = _cartesian_hessian_matrix(hessian_raw, len(generated_elements))
    antisymmetry = float(np.max(np.abs(hessian - hessian.T)))
    if antisymmetry > 1.0e-5:
        raise McpbPySCFError(
            f"Cartesian Hessian is not symmetric (max antisymmetry {antisymmetry:.3e})."
        )
    hessian = 0.5 * (hessian + hessian.T)
    optimized_bohr = optimized_coordinates.reshape(-1) * BOHR_PER_ANGSTROM
    _write_mcpb_fchk(
        output_fchk,
        coordinates_bohr=optimized_bohr,
        hessian_hartree_per_bohr2=hessian,
    )
    _validate_fchk_counts(output_fchk, atom_count=len(generated_elements))
    total_runtime = perf_counter() - total_started
    calculation_report = work_dir / "single_point_hessian.json"
    calculation_report.write_text(
        json.dumps(
            {
                "calculation": calculation_name,
                "hessian_backend": hessian_backend,
                "backend_version": backend_version,
                "geometry_optimization_performed": False,
                "geometry_source": config.geometry_source,
                "geometry_status": geometry_status,
                "qmmm_refinement_source_pdb": (
                    str(qmmm_refinement_source_pdb)
                    if qmmm_refinement_source_pdb is not None
                    else None
                ),
                "scientific_warning": scientific_warning,
                "mcpb_generated_pdb": str(input_pdb),
                "user_optimized_pdb": str(optimized_path),
                "evaluated_pdb": str(evaluated_structure.path),
                "atom_identity_validation": "exact",
                "atom_count": len(generated_elements),
                "molecular_charge": charge,
                "multiplicity": multiplicity,
                "spin": spin,
                "energy_hartree": energy_hartree,
                "scf_calculation_count": scf_calculation_count,
                "hessian_antisymmetry_max": antisymmetry,
                "scf_runtime_seconds": scf_runtime,
                "hessian_runtime_seconds": hessian_runtime,
                "total_runtime_seconds": total_runtime,
                "external_commands": list(external_command_records),
                "note": hessian_note,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return SmallModelResult(
        mcpb_generated_pdb_path=input_pdb,
        user_optimized_pdb_path=optimized_path,
        qmmm_refinement_source_pdb_path=qmmm_refinement_source_pdb,
        evaluated_pdb_path=evaluated_structure.path,
        evaluated_xyz_path=evaluated_xyz,
        checkpoint_path=checkpoint,
        fchk_path=output_fchk,
        calculation_report_path=calculation_report,
        scf_stdout_path=stdout_path,
        scf_stderr_path=stderr_path,
        atom_count=len(generated_elements),
        energy_hartree=energy_hartree,
        geometry_status=geometry_status,
        scientific_warning=scientific_warning,
        geometry_optimization_performed=False,
        scf_calculation_count=scf_calculation_count,
        hessian_antisymmetry_max=antisymmetry,
        hessian_backend=hessian_backend,
        backend_version=backend_version,
        scf_runtime_seconds=scf_runtime,
        hessian_runtime_seconds=hessian_runtime,
        total_runtime_seconds=total_runtime,
        external_command_records=external_command_records,
    )


def _run_large_model(
    site: ResolvedMetalSite,
    input_pdb: Path,
    *,
    output_mk_log: Path,
    charge: int,
    multiplicity: int,
    config: McpbPySCFConfig,
    provisional_prmtop: Path,
    provisional_inpcrd: Path,
    provisional_structure: PdbStructure,
    work_dir: Path,
) -> LargeModelResult:
    structure = read_pdb(input_pdb)
    elements, coordinates = _elements_and_coordinates(structure)
    work_dir.mkdir(parents=True, exist_ok=True)
    embedding = extract_mcpb_embedding_charges(
        site,
        prmtop=provisional_prmtop,
        inpcrd=provisional_inpcrd,
        provisional_structure=provisional_structure,
        qm_coordinates=coordinates,
        cutoff_angstrom=config.embedding_cutoff_angstrom,
        minimum_distance_angstrom=config.embedding_min_distance_angstrom,
    )
    if not embedding.point_charges:
        raise McpbPySCFError(
            "The configured MCPB PySCF electrostatic embedding selected zero MM point "
            "charges; use workflow: complete with reviewed external artifacts if a gas-phase "
            "large model is intended."
        )
    point_csv = work_dir / "mm_point_charges.csv"
    point_xyz = work_dir / "mm_point_charges.xyz"
    point_report = work_dir / "mm_point_charges.json"
    _write_embedding_files(
        embedding,
        csv_path=point_csv,
        xyz_path=point_xyz,
        report_path=point_report,
    )
    checkpoint = work_dir / "large_embedded.chk"
    try:
        scf_result = run_pyscf_scf(
            elements=elements,
            coordinates=coordinates,
            charge=charge,
            spin=multiplicity - 1,
            method=config.method,
            basis=config.basis,
            max_cycle=config.max_cycle,
            conv_tol=config.conv_tol,
            mm_charges=embedding.charge_array,
            mm_coordinates=embedding.coordinate_array,
            checkpoint_path=checkpoint,
            initial_density_checkpoint_path=(
                config.large_model_restart_checkpoint
            ),
            expected_initial_density_checkpoint_sha256=(
                config.large_model_restart_checkpoint_sha256
            ),
            initial_density_checkpoint_max_displacement_angstrom=(
                config.large_model_restart_max_displacement_angstrom
            ),
            work_dir=work_dir,
            num_threads=config.num_threads,
            max_memory_mb=config.max_memory_mb,
            dft_grid_level=config.dft_grid_level,
            scf_algorithm=config.scf_algorithm,
            initial_guess=config.initial_guess,
            adiis_precondition_cycles=config.adiis_precondition_cycles,
            adiis_precondition_dft_grid_level=(
                config.adiis_precondition_dft_grid_level
            ),
            level_shift_mode=config.level_shift_mode,
            level_shift_hartree=config.level_shift_hartree,
            damping_factor=config.damping_factor,
            diis_space=config.diis_space,
            density_fitting=config.large_model_density_fitting,
            auxiliary_basis=config.large_model_auxbasis,
        )
    except PySCFRunnerError as exc:
        raise McpbPySCFError(f"Embedded MCPB large-model SCF failed: {exc}") from exc
    if not scf_result.electrostatic_embedding_applied:
        raise McpbPySCFError(
            "MCPB large-model PySCF result did not record electrostatic embedding."
        )

    grid_config = config.grid
    grid = generate_merz_kollman_grid(
        elements=elements,
        coordinates=coordinates,
        vdw_scale_factors=grid_config.vdw_scale_factors,
        point_density_per_square_angstrom=(
            grid_config.point_density_per_square_angstrom
        ),
        exclude_inside_vdw_scale=grid_config.exclude_inside_vdw_scale,
        max_points=grid_config.max_points,
    )
    try:
        esp_values = evaluate_ligand_esp(
            mol=scf_result.mol,
            mf=scf_result.mf,
            grid_coordinates_angstrom=grid.points,
            batch_size=config.esp_batch_size,
        )
        atomic_esp_values = _evaluate_atomic_esp_without_self(scf_result)
    except Exception as exc:
        raise McpbPySCFError(f"PySCF large-model ESP evaluation failed: {exc}") from exc
    grid_path = work_dir / "large_model_esp_grid.xyz"
    values_path = work_dir / "large_model_esp_values.txt"
    write_grid_xyz(grid, grid_path)
    values_path.write_text(
        "".join(f"{float(value):.12e}\n" for value in esp_values),
        encoding="utf-8",
    )
    write_mcpb_gaussian_esp_adapter(
        output_mk_log,
        atomic_coordinates_angstrom=coordinates,
        atomic_esp_values=atomic_esp_values,
        fit_coordinates_angstrom=grid.points,
        fit_esp_values=esp_values,
    )
    _validate_mk_adapter(
        output_mk_log,
        atom_count=len(elements),
        fit_count=len(grid.points),
    )
    stdout_path = work_dir / "large_embedded_scf.stdout.txt"
    stderr_path = work_dir / "large_embedded_scf.stderr.txt"
    stdout_path.write_text(scf_result.stdout, encoding="utf-8")
    stderr_path.write_text(scf_result.stderr, encoding="utf-8")
    return LargeModelResult(
        input_pdb_path=input_pdb,
        checkpoint_path=checkpoint,
        mk_log_path=output_mk_log,
        grid_xyz_path=grid_path,
        esp_values_path=values_path,
        point_charge_csv_path=point_csv,
        point_charge_xyz_path=point_xyz,
        point_charge_report_path=point_report,
        scf_stdout_path=stdout_path,
        scf_stderr_path=stderr_path,
        atom_count=len(elements),
        energy_hartree=scf_result.energy_hartree,
        grid_point_count=len(grid.points),
        embedding=embedding,
        scf_algorithm=scf_result.scf_algorithm,
        initial_guess=scf_result.initial_guess,
        initial_density_checkpoint_path=(
            scf_result.initial_density_checkpoint_path
        ),
        initial_density_checkpoint_sha256=(
            scf_result.initial_density_checkpoint_sha256
        ),
        initial_density_checkpoint_max_displacement_angstrom=(
            scf_result.initial_density_checkpoint_max_displacement_angstrom
        ),
        initial_density_checkpoint_rms_displacement_angstrom=(
            scf_result.initial_density_checkpoint_rms_displacement_angstrom
        ),
        initial_density_checkpoint_projected=(
            scf_result.initial_density_checkpoint_projected
        ),
        adiis_precondition_cycles=scf_result.adiis_precondition_cycles,
        adiis_precondition_dft_grid_level=(
            scf_result.adiis_precondition_dft_grid_level
        ),
        precondition_cycles_completed=scf_result.precondition_cycles_completed,
        level_shift_mode=scf_result.level_shift_mode,
        level_shift_hartree=scf_result.level_shift_hartree,
        damping_factor=scf_result.damping_factor,
        diis_space=scf_result.diis_space,
        density_fitting=scf_result.density_fitting,
        auxiliary_basis=scf_result.auxiliary_basis,
        spin_square=scf_result.spin_square,
        effective_multiplicity=scf_result.effective_multiplicity,
    )


def extract_mcpb_embedding_charges(
    site: ResolvedMetalSite,
    *,
    prmtop: str | Path,
    inpcrd: str | Path,
    provisional_structure: PdbStructure,
    qm_coordinates: np.ndarray,
    cutoff_angstrom: float | None,
    minimum_distance_angstrom: float,
) -> McpbEmbeddingSelection:
    """Select MM charges while excluding every residue represented in the QM model."""

    try:
        import parmed
    except Exception as exc:
        raise McpbPySCFError(
            "ParmEd is required to extract the MCPB electrostatic environment."
        ) from exc
    try:
        topology = parmed.load_file(str(prmtop), str(inpcrd))
    except Exception as exc:
        raise McpbPySCFError(
            f"Could not load the pre-MCPB Amber topology with ParmEd: {exc}"
        ) from exc
    if topology.coordinates is None:
        raise McpbPySCFError("Pre-MCPB Amber topology contains no coordinates.")
    if len(topology.residues) != len(provisional_structure.residues):
        raise McpbPySCFError(
            "Pre-MCPB topology/residue mapping is inconsistent: "
            f"{len(topology.residues)} topology residues versus "
            f"{len(provisional_structure.residues)} restored PDB residues."
        )
    excluded_keys = {
        (
            ion.residue.id.chain_id,
            ion.residue.id.resid,
            ion.residue.id.icode,
        )
        for ion in site.ions
    }
    excluded_keys.update(
        (
            bond.coordinator_residue.id.chain_id,
            bond.coordinator_residue.id.resid,
            bond.coordinator_residue.id.icode,
        )
        for bond in site.bonds
    )
    excluded_keys.update(
        (residue.id.chain_id, residue.id.resid, residue.id.icode)
        for residue in site.additional_residues
    )
    topology_coordinates = np.asarray(topology.coordinates, dtype=float)
    qm_coords = np.asarray(qm_coordinates, dtype=float)
    charges: list[PointCharge] = []
    distances: list[float] = []
    categories: dict[str, int] = {}
    candidate_count = 0
    too_close = 0
    beyond_cutoff = 0
    excluded_residue_count = 0
    for topology_residue, pdb_residue in zip(
        topology.residues,
        provisional_structure.residues,
        strict=True,
    ):
        topology_names = [str(atom.name) for atom in topology_residue.atoms]
        pdb_names = [atom.name for atom in pdb_residue.atoms]
        if topology_names != pdb_names:
            raise McpbPySCFError(
                "Pre-MCPB topology atom mapping changed for "
                f"{pdb_residue.id.display()}: topology names {topology_names}, "
                f"PDB names {pdb_names}."
            )
        key = (
            pdb_residue.id.chain_id,
            pdb_residue.id.resid,
            pdb_residue.id.icode,
        )
        if key in excluded_keys:
            excluded_residue_count += 1
            continue
        category = _residue_category(pdb_residue)
        for topology_atom in topology_residue.atoms:
            candidate_count += 1
            index = int(topology_atom.idx)
            coordinate = topology_coordinates[index]
            distance = float(
                np.min(np.linalg.norm(qm_coords - coordinate[None, :], axis=1))
            )
            if distance < minimum_distance_angstrom:
                too_close += 1
                continue
            if cutoff_angstrom is not None and distance > cutoff_angstrom:
                beyond_cutoff += 1
                continue
            charges.append(
                PointCharge(
                    x=float(coordinate[0]),
                    y=float(coordinate[1]),
                    z=float(coordinate[2]),
                    charge=float(topology_atom.charge),
                    residue_name=pdb_residue.id.resname,
                    residue_number=pdb_residue.id.resid,
                    atom_name=str(topology_atom.name),
                    category=category,
                )
            )
            distances.append(distance)
            categories[category] = categories.get(category, 0) + 1
    return McpbEmbeddingSelection(
        point_charges=tuple(charges),
        excluded_qm_residue_count=excluded_residue_count,
        candidate_atom_count=candidate_count,
        excluded_by_minimum_distance=too_close,
        excluded_by_cutoff=beyond_cutoff,
        minimum_distance_angstrom=min(distances) if distances else None,
        maximum_distance_angstrom=max(distances) if distances else None,
        net_embedding_charge=float(sum(item.charge for item in charges)),
        categories=categories,
    )


def write_mcpb_gaussian_esp_adapter(
    path: str | Path,
    *,
    atomic_coordinates_angstrom: np.ndarray,
    atomic_esp_values: np.ndarray,
    fit_coordinates_angstrom: np.ndarray,
    fit_esp_values: np.ndarray,
) -> None:
    """Write the exact Gaussian text subset parsed by MCPB.py's ESP reader."""

    atomic_coordinates = np.asarray(atomic_coordinates_angstrom, dtype=float)
    atomic_values = np.asarray(atomic_esp_values, dtype=float)
    fit_coordinates = np.asarray(fit_coordinates_angstrom, dtype=float)
    fit_values = np.asarray(fit_esp_values, dtype=float)
    if atomic_coordinates.shape != (len(atomic_values), 3):
        raise McpbPySCFError("Atomic ESP adapter arrays have inconsistent shapes.")
    if fit_coordinates.shape != (len(fit_values), 3) or not len(fit_values):
        raise McpbPySCFError("Fit ESP adapter arrays have inconsistent or empty shapes.")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            atomic_coordinates,
            atomic_values,
            fit_coordinates,
            fit_values,
        )
    ):
        raise McpbPySCFError("ESP adapter contains non-finite values.")
    lines = [
        " PySCF electrostatically embedded MCPB large-model ESP adapter",
        " Electrostatic Properties Using The SCF Density",
    ]
    for index, coordinate in enumerate(atomic_coordinates, start=1):
        prefix = f"      Atomic Center {index:5d}"
        lines.append(
            f"{prefix:<32}{coordinate[0]:10.6f}{coordinate[1]:10.6f}"
            f"{coordinate[2]:10.6f}"
        )
    for index, coordinate in enumerate(fit_coordinates, start=1):
        prefix = f"     ESP Fit Center {index:5d}"
        lines.append(
            f"{prefix:<32}{coordinate[0]:10.6f}{coordinate[1]:10.6f}"
            f"{coordinate[2]:10.6f}"
        )
    lines.extend(
        [
            " Electrostatic Properties (Atomic Units)",
            " -----------------------------------------------------------------",
            " Center      Type                 Electrostatic Potential",
            " -----------------------------------------------------------------",
            " Values below are evaluated from the polarized PySCF QM density.",
            " MM embedding charges are excluded from the reported ESP.",
        ]
    )
    lines.extend(
        f" Atom {index:6d} Potential {value: .12E}"
        for index, value in enumerate(atomic_values, start=1)
    )
    lines.extend(
        f" Fit  {index:6d} Potential {value: .12E}"
        for index, value in enumerate(fit_values, start=1)
    )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _evaluate_atomic_esp_without_self(result: PySCFResult) -> np.ndarray:
    mol = result.mol
    mf = result.mf
    atom_coordinates = np.asarray(mol.atom_coords(), dtype=float)
    nuclear_charges = np.asarray(mol.atom_charges(), dtype=float)
    density = mf.make_rdm1()
    if isinstance(density, tuple) or (
        hasattr(density, "ndim") and density.ndim == 3
    ):
        total_density = np.asarray(density[0]) + np.asarray(density[1])
    else:
        total_density = np.asarray(density)
    values: list[float] = []
    for index, origin in enumerate(atom_coordinates):
        distances = np.linalg.norm(atom_coordinates - origin[None, :], axis=1)
        mask = np.arange(len(atom_coordinates)) != index
        nuclear = float(np.sum(nuclear_charges[mask] / distances[mask]))
        mol.set_rinv_origin(origin)
        electronic = -float(
            np.einsum("ij,ij->", total_density, mol.intor("int1e_rinv"))
        )
        values.append(nuclear + electronic)
    result_array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result_array)):
        raise McpbPySCFError("Atomic ESP evaluation produced non-finite values.")
    return result_array


def _cartesian_hessian_matrix(raw: np.ndarray, atom_count: int) -> np.ndarray:
    expected_matrix = (atom_count * 3, atom_count * 3)
    if raw.shape == expected_matrix:
        matrix = raw
    elif raw.shape == (atom_count, atom_count, 3, 3):
        matrix = raw.transpose(0, 2, 1, 3).reshape(expected_matrix)
    else:
        raise McpbPySCFError(
            f"PySCF Hessian shape {raw.shape} is incompatible with {atom_count} atoms."
        )
    if not np.all(np.isfinite(matrix)):
        raise McpbPySCFError("PySCF Cartesian Hessian contains non-finite values.")
    return np.asarray(matrix, dtype=float)


def _write_mcpb_fchk(
    path: Path,
    *,
    coordinates_bohr: np.ndarray,
    hessian_hartree_per_bohr2: np.ndarray,
) -> None:
    coordinates = np.asarray(coordinates_bohr, dtype=float).reshape(-1)
    hessian = np.asarray(hessian_hartree_per_bohr2, dtype=float)
    if hessian.shape != (len(coordinates), len(coordinates)):
        raise McpbPySCFError(
            "Formatted-checkpoint coordinate and Hessian dimensions do not agree."
        )
    lower = np.asarray(
        [hessian[row, column] for row in range(len(coordinates)) for column in range(row + 1)],
        dtype=float,
    )
    lines = [
        "mdprep PySCF MCPB formatted-checkpoint adapter",
        "Generated for MCPB.py Seminario parsing",
        f"Current cartesian coordinates    R   N= {len(coordinates):12d}",
        *_format_fchk_values(coordinates),
        f"Cartesian Force Constants        R   N= {len(lower):12d}",
        *_format_fchk_values(lower),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_fchk_values(values: np.ndarray) -> list[str]:
    result: list[str] = []
    for start in range(0, len(values), 5):
        result.append(
            "".join(f"{float(value):16.8E}" for value in values[start : start + 5])
        )
    return result


def _validate_fchk_counts(path: Path, *, atom_count: int) -> None:
    text = path.read_text(encoding="utf-8")
    coordinate_count = 3 * atom_count
    hessian_count = coordinate_count * (coordinate_count + 1) // 2
    if (
        "Current cartesian coordinates" not in text
        or f"{coordinate_count:12d}" not in text
        or "Cartesian Force Constants" not in text
        or f"{hessian_count:12d}" not in text
    ):
        raise McpbPySCFError(
            f"Formatted-checkpoint validation failed for {path}."
        )


def _validate_mk_adapter(path: Path, *, atom_count: int, fit_count: int) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count("      Atomic Center") != atom_count:
        raise McpbPySCFError("MCPB MK adapter atomic-center count is incorrect.")
    if text.count("     ESP Fit Center") != fit_count:
        raise McpbPySCFError("MCPB MK adapter fit-center count is incorrect.")
    if sum(1 for line in text.splitlines() if " Atom " in f" {line} ") != atom_count:
        raise McpbPySCFError("MCPB MK adapter atomic-ESP count is incorrect.")
    if sum(1 for line in text.splitlines() if line.startswith(" Fit  ")) != fit_count:
        raise McpbPySCFError("MCPB MK adapter fit-ESP count is incorrect.")


def _elements_and_coordinates(
    structure: PdbStructure,
) -> tuple[list[str], np.ndarray]:
    elements: list[str] = []
    coordinates: list[list[float]] = []
    for atom in structure.atoms:
        element = atom.element or infer_element(
            atom.name,
            resname=atom.resname,
            record_name=atom.record_name,
            atom_field=(atom.original_line[12:16] if atom.original_line else None),
        )
        if not element:
            raise McpbPySCFError(
                f"Could not infer element for MCPB model atom {atom.atom_identity}."
            )
        elements.append(str(element).capitalize())
        coordinates.append([atom.x, atom.y, atom.z])
    if not elements:
        raise McpbPySCFError(f"MCPB model {structure.path} contains zero atoms.")
    values = np.asarray(coordinates, dtype=float)
    if not np.all(np.isfinite(values)):
        raise McpbPySCFError(f"MCPB model {structure.path} has non-finite coordinates.")
    return elements, values


def _validate_user_optimized_small_model(
    generated_structure: PdbStructure,
    *,
    generated_elements: list[str],
    user_structure: PdbStructure,
    user_elements: list[str],
) -> None:
    """Require a coordinate-only replacement of MCPB.py's small model."""

    generated_count = len(generated_structure.atoms)
    user_count = len(user_structure.atoms)
    if user_count != generated_count:
        raise McpbPySCFError(
            "User-optimized MCPB small-model atom count does not match MCPB.py step 1: "
            f"expected {generated_count}, found {user_count}. Geometry optimization must "
            "not add, remove, or reorder atoms."
        )
    mismatches: list[str] = []
    for index, (generated, user, generated_element, user_element) in enumerate(
        zip(
            generated_structure.atoms,
            user_structure.atoms,
            generated_elements,
            user_elements,
            strict=True,
        ),
        start=1,
    ):
        expected = (
            generated_element,
            generated.name.strip(),
            generated.resname.strip(),
            generated.chain_id,
            generated.resid,
            generated.icode,
        )
        observed = (
            user_element,
            user.name.strip(),
            user.resname.strip(),
            user.chain_id,
            user.resid,
            user.icode,
        )
        if observed != expected:
            mismatches.append(
                f"atom {index}: expected {expected!r}, found {observed!r}"
            )
            if len(mismatches) == 5:
                break
    if mismatches:
        raise McpbPySCFError(
            "User-optimized MCPB small-model identity/order does not match MCPB.py "
            "step 1. Only coordinates may differ. First mismatches: "
            + "; ".join(mismatches)
        )


def _structure_with_coordinates(
    structure: PdbStructure,
    coordinates: np.ndarray,
    *,
    path: Path,
) -> PdbStructure:
    from dataclasses import replace

    atoms = [
        replace(atom, x=float(coord[0]), y=float(coord[1]), z=float(coord[2]))
        for atom, coord in zip(structure.atoms, coordinates, strict=True)
    ]
    by_key: dict[tuple[str, str, int, str | None], list[AtomRecord]] = {}
    for atom in atoms:
        by_key.setdefault(atom.residue_key, []).append(atom)
    residues = [
        ResidueRecord(
            id=residue.id,
            atoms=by_key[(
                residue.id.chain_id,
                residue.id.resname,
                residue.id.resid,
                residue.id.icode,
            )],
            record_names=residue.record_names,
            original_index=residue.original_index,
        )
        for residue in structure.residues
    ]
    return PdbStructure(
        path=path,
        atoms=atoms,
        residues=residues,
        model_count=1,
        used_model=1,
        warnings=list(structure.warnings),
        conect_bonds=set(structure.conect_bonds),
        ter_after_serials=set(structure.ter_after_serials),
    )


def _write_xyz(
    elements: list[str],
    coordinates: np.ndarray,
    path: Path,
    comment: str,
) -> None:
    lines = [str(len(elements)), comment]
    lines.extend(
        f"{element:<2} {coord[0]: .10f} {coord[1]: .10f} {coord[2]: .10f}"
        for element, coord in zip(elements, coordinates, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_embedding_files(
    selection: McpbEmbeddingSelection,
    *,
    csv_path: Path,
    xyz_path: Path,
    report_path: Path,
) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "x",
                "y",
                "z",
                "charge",
                "residue_name",
                "residue_number",
                "atom_name",
                "category",
            ],
        )
        writer.writeheader()
        for item in selection.point_charges:
            writer.writerow(item.to_dict())
    lines = [str(len(selection.point_charges)), "MCPB PySCF MM embedding charges"]
    lines.extend(
        f"X {item.x:.8f} {item.y:.8f} {item.z:.8f} {item.charge:.8f}"
        for item in selection.point_charges
    )
    xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _residue_category(residue: ResidueRecord) -> str:
    name = residue.id.resname.upper()
    if name in WATER_RESIDUES:
        return "water"
    if name in STANDARD_PROTEIN_RESIDUES:
        return "protein"
    return "ligand_or_other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
