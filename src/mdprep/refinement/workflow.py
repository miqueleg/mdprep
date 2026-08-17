"""High-level provisional-build and ASH refinement workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.ligands.workflow import (
    LigandStageResult,
    LigandWorkflowError,
    QmmespProvisionalSystem,
    build_provisional_am1bcc_system,
)
from mdprep.protonation.apply import ProtonationResult
from mdprep.refinement.ash import (
    AshRefinementCache,
    AshRefinementError,
    AshRefinementRun,
    run_ash_refinement,
)
from mdprep.refinement.coordinates import (
    CoordinateTransferError,
    CoordinateTransferResult,
    transfer_optimized_coordinates,
)
from mdprep.refinement.selection import (
    RefinementSelection,
    RefinementSelectionError,
    select_refinement_regions,
)
from mdprep.structure.models import PdbStructure
from mdprep.structure.pdb import PdbParseError, read_pdb
from mdprep.structure.writer import write_pdb


class RefinementWorkflowError(ValueError):
    """Raised when a requested QM/MM refinement cannot complete safely."""


@dataclass(frozen=True)
class RefinementResult:
    provisional_ligands: LigandStageResult
    provisional_system: QmmespProvisionalSystem
    selection: RefinementSelection
    ash_run: AshRefinementRun
    coordinate_transfer: CoordinateTransferResult
    refined_hydrogenated_pdb_path: Path
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structure(self) -> PdbStructure:
        return self.coordinate_transfer.structure

    def to_report_dict(self) -> dict[str, object]:
        active_waters = sum(
            bool(item["water"]) for item in self.selection.active_residues
        )
        return {
            "status": "complete",
            "backend": "ash",
            "qm_method": self.ash_run.qm_method,
            "embedding": self.ash_run.embedding,
            "provisional_ligand_charge_method": "am1bcc",
            "provisional_ligands": self.provisional_ligands.to_report_dict()["ligands"],
            "provisional_system": self.provisional_system.to_dict(),
            "selection": self.selection.to_dict(),
            "active_water_residue_count": active_waters,
            "ash": self.ash_run.to_dict(),
            "coordinate_transfer": self.coordinate_transfer.to_dict(),
            "refined_hydrogenated_pdb_path": str(self.refined_hydrogenated_pdb_path),
            "warnings": list(self.warnings),
        }


def run_refinement_stage(
    *,
    protonated_structure: PdbStructure,
    reference_structure: PdbStructure,
    manifest: ManifestConfig,
    output_dir: str | Path,
    protonation_result: ProtonationResult,
    cached_ash_run: AshRefinementCache | None = None,
) -> RefinementResult:
    """Build provisional AM1-BCC parameters and refine the configured active site."""

    if not manifest.refinement.enabled:
        raise RefinementWorkflowError("Refinement stage requested but refinement.enabled is false")
    assert manifest.refinement.ash is not None  # validated by ManifestConfig
    output = Path(output_dir)
    _remove_stale_refinement_failure(output)
    refinement_dir = output / "refinement"
    provisional_dir = refinement_dir / "provisional"
    ash_dir = refinement_dir / "ash"
    refined_path = output / "intermediate" / "02_qmmm_refined_hydrogenated.pdb"
    try:
        provisional_ligands, provisional_system = build_provisional_am1bcc_system(
            protonated_structure,
            manifest,
            output_dir=provisional_dir,
            protonation_result=protonation_result,
        )
        leap_input_structure = read_pdb(provisional_system.leap_input_path)
        topology_structure = read_pdb(provisional_system.pdb_path)
        selection = select_refinement_regions(
            leap_input_structure=leap_input_structure,
            topology_structure=topology_structure,
            manifest=manifest,
        )
        ash_run = run_ash_refinement(
            prmtop_path=provisional_system.prmtop_path,
            inpcrd_path=provisional_system.inpcrd_path,
            selection=selection,
            config=manifest.refinement.ash,
            work_dir=ash_dir,
            qm_method=manifest.refinement.qm_method,
            embedding=manifest.refinement.embedding,
            mace_polar1=manifest.refinement.mace_polar1,
            cached_run=cached_ash_run,
        )
        coordinate_transfer = transfer_optimized_coordinates(
            reference_structure=reference_structure,
            leap_input_structure=leap_input_structure,
            topology_structure=topology_structure,
            coordinates_angstrom=ash_run.optimized_coordinates_angstrom,
            output_path=refined_path,
        )
        write_pdb(coordinate_transfer.structure, refined_path)
    except (
        LigandWorkflowError,
        PdbParseError,
        RefinementSelectionError,
        AshRefinementError,
        CoordinateTransferError,
        FileNotFoundError,
        OSError,
    ) as exc:
        _write_refinement_failure(output, exc)
        raise RefinementWorkflowError(str(exc)) from exc
    warnings = [
        "The optimized structure is an intermediate model only; final ligand/cofactor "
        "charges and parameters are regenerated with each ligand's requested method.",
    ]
    if manifest.refinement.embedding == "mechanical":
        warnings.append(
            "Mechanical embedding does not polarize the QM method with MM point charges; "
            "QM-MM electrostatics and van der Waals interactions are evaluated by OpenMM."
        )
    if manifest.refinement.qm_method == "mace_polar1":
        warnings.append(
            "MACE-POLAR-1 is a machine-learned high-layer potential rather than an "
            "electronic-structure QM calculation; this run is mechanically embedded ML/MM."
        )
    return RefinementResult(
        provisional_ligands=provisional_ligands,
        provisional_system=provisional_system,
        selection=selection,
        ash_run=ash_run,
        coordinate_transfer=coordinate_transfer,
        refined_hydrogenated_pdb_path=refined_path,
        warnings=tuple(warnings),
    )


def _write_refinement_failure(output_dir: Path, exc: Exception) -> None:
    path = output_dir / "reports" / "refinement_failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "errors": [str(exc)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _remove_stale_refinement_failure(output_dir: Path) -> None:
    """Ensure a successful overwrite cannot retain a failure from an older run."""
    path = output_dir / "reports" / "refinement_failure.json"
    if path.exists():
        path.unlink()
