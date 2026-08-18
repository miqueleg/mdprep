"""PySCF-based ligand RESP/QMMESP charge derivation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdprep.ambertools.mol2 import write_mol2_with_charges
from mdprep.ambertools.resp import AmberRespError, AmberRespFitResult, run_amber_resp_fit
from mdprep.charges.esp_grid import EspGridError, generate_merz_kollman_grid, write_grid_xyz
from mdprep.config.models import LigandConfig
from mdprep.ligands.extract import ExtractedLigand
from mdprep.qm.point_charges import PointChargeSelection, write_point_charge_files
from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM, PySCFEspError, evaluate_ligand_esp, write_esp_values
from mdprep.qm.pyscf_runner import PySCFResult, PySCFRunnerError, run_pyscf_scf
from mdprep.structure.models import AtomRecord


class LigandPySCFChargeError(ValueError):
    """Raised when PySCF ligand charge derivation fails."""


QMMESP_CONFIRMATION = (
    "The target-ligand PySCF single point was run as electrostatically embedded QM/MM "
    "using pyscf.qmmm.mm_charge. Fixed MM point charges from the provisional Amber "
    "system polarized the target-ligand QM density. The RESP fitting target contains "
    "only the polarized target-ligand QM electrostatic potential; environment point "
    "charges were not fitted and were not written to the ligand mol2."
)


GAS_RESP_CONFIRMATION = "Gas-phase ligand ESP fit; no MM point charges were used."


@dataclass(frozen=True)
class LigandPySCFChargeResult:
    method: str
    qm_dir: Path
    charged_mol2_path: Path
    fitted_charges_csv_path: Path
    fit_report_path: Path
    pyscf_result: dict[str, object]
    fit_result: dict[str, object]
    grid_point_count: int
    embedding_summary: dict[str, object] | None
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "qm_dir": str(self.qm_dir),
            "charged_mol2_path": str(self.charged_mol2_path),
            "fitted_charges_csv_path": str(self.fitted_charges_csv_path),
            "fit_report_path": str(self.fit_report_path),
            "pyscf_result": self.pyscf_result,
            "fit_result": self.fit_result,
            "grid_point_count": self.grid_point_count,
            "embedding_summary": self.embedding_summary,
            "warnings": self.warnings,
        }


def derive_pyscf_charges(
    *,
    extracted: ExtractedLigand,
    provisional_mol2_path: str | Path,
    output_mol2_path: str | Path,
    output_dir: str | Path,
    method_name: str,
    point_charges: PointChargeSelection | None = None,
) -> LigandPySCFChargeResult:
    ligand = extracted.config
    config = ligand.qmmesp
    if config is None:
        raise LigandPySCFChargeError(f"Ligand {ligand.id} requires a qmmesp block for {method_name}.")
    if method_name == "qmmesp_pyscf" and point_charges is None:
        raise LigandPySCFChargeError("qmmesp_pyscf requires an explicit provisional-system MM environment selection.")
    if method_name == "qmmesp_pyscf" and point_charges is not None and point_charges.total_after_cutoff == 0:
        raise LigandPySCFChargeError(
            "qmmesp_pyscf selected zero MM point charges; refusing to run a gas-phase SCF "
            "under a QMMESP label. Expand the environment selection or embedding cutoff."
        )
    if method_name == "gas_resp_pyscf" and point_charges is not None:
        raise LigandPySCFChargeError("gas_resp_pyscf cannot accept MM embedding point charges.")
    qm_dir = Path(output_dir) / "ligands" / ligand.id / "qm" / method_name
    qm_dir.mkdir(parents=True, exist_ok=True)
    atoms = extracted.atoms
    elements = [_atom_element(atom) for atom in atoms]
    coords = np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    scf_charge = ligand.net_charge if config.scf_charge is None else config.scf_charge
    scf_spin = ligand.multiplicity - 1 if config.scf_spin is None else config.scf_spin
    embedded_qmmm = point_charges is not None
    point_charge_count = 0 if point_charges is None else point_charges.total_after_cutoff
    pyscf_input = {
        "ligand_id": ligand.id,
        "elements": elements,
        "coordinates_angstrom": coords.tolist(),
        "charge": scf_charge,
        "spin": scf_spin,
        "multiplicity": scf_spin + 1,
        "method": config.method,
        "basis": config.basis,
        "max_cycle": config.max_cycle,
        "conv_tol": config.conv_tol,
        "num_threads": config.num_threads,
        "max_memory_mb": config.max_memory_mb,
        "checkpoint_path": str(qm_dir / "pyscf.chk"),
        "calculation_type": (
            "electrostatically_embedded_qm_mm_single_point"
            if embedded_qmmm
            else "gas_phase_qm_single_point"
        ),
        "qm_region": "target_ligand_only",
        "qm_atom_count": len(atoms),
        "mm_region": "selected_non_target_atoms_from_provisional_amber_system" if embedded_qmmm else None,
        "mm_point_charge_count": point_charge_count,
        "electrostatic_embedding": embedded_qmmm,
        "embedding_operator": "pyscf.qmmm.mm_charge" if embedded_qmmm else None,
        "mm_point_charge_potential_in_scf_hamiltonian": embedded_qmmm,
        "mm_bonded_and_lennard_jones_terms_in_scf": False,
        "direct_mm_potential_in_resp_target": False,
        "esp_grid": config.grid.model_dump(mode="json"),
        "resp_fitting": config.resp_fitting.model_dump(mode="json"),
    }
    (qm_dir / "pyscf_input.json").write_text(json.dumps(pyscf_input, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if point_charges is not None:
        write_point_charge_files(
            point_charges,
            csv_path=qm_dir / "mm_point_charges.csv",
            xyz_path=qm_dir / "mm_point_charges.xyz",
            summary_path=qm_dir / "embedding_summary.json",
        )
    try:
        grid = generate_merz_kollman_grid(
            elements=elements,
            coordinates=coords,
            vdw_scale_factors=config.grid.vdw_scale_factors,
            point_density_per_square_angstrom=config.grid.point_density_per_square_angstrom,
            exclude_inside_vdw_scale=config.grid.exclude_inside_vdw_scale,
            max_points=config.grid.max_points,
        )
        mm_charges = None if point_charges is None else point_charges.charge_array
        mm_coords = None if point_charges is None else point_charges.coordinate_array
        pyscf_result = run_pyscf_scf(
            elements=elements,
            coordinates=coords,
            charge=scf_charge,
            spin=scf_spin,
            method=config.method,
            basis=config.basis,
            max_cycle=config.max_cycle,
            conv_tol=config.conv_tol,
            mm_charges=mm_charges,
            mm_coordinates=mm_coords,
            work_dir=qm_dir,
            checkpoint_path=qm_dir / "pyscf.chk",
            num_threads=config.num_threads,
            max_memory_mb=config.max_memory_mb,
        )
        (qm_dir / "pyscf_stdout.txt").write_text(pyscf_result.stdout, encoding="utf-8")
        (qm_dir / "pyscf_stderr.txt").write_text(pyscf_result.stderr, encoding="utf-8")
        (qm_dir / "pyscf_result.json").write_text(
            json.dumps(pyscf_result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        esp = evaluate_ligand_esp(
            mol=pyscf_result.mol,
            mf=pyscf_result.mf,
            grid_coordinates_angstrom=grid.points,
        )
        write_grid_xyz(grid, qm_dir / "esp_grid.xyz")
        write_esp_values(esp, str(qm_dir / "esp_values.dat"))
        fit = run_amber_resp_fit(
            provisional_mol2_path=provisional_mol2_path,
            atom_coordinates_bohr=coords * BOHR_PER_ANGSTROM,
            grid_coordinates_bohr=grid.points * BOHR_PER_ANGSTROM,
            esp_values_au=esp,
            total_charge=ligand.net_charge,
            multiplicity=ligand.multiplicity,
            atom_types=ligand.atom_types,
            work_dir=qm_dir / "resp",
        )
    except (EspGridError, PySCFRunnerError, PySCFEspError, AmberRespError) as exc:
        raise LigandPySCFChargeError(str(exc)) from exc

    charged_mol2 = Path(output_mol2_path)
    write_mol2_with_charges(provisional_mol2_path, [float(charge) for charge in fit.charges], charged_mol2)
    charges_csv = qm_dir / "fitted_charges.csv"
    _write_fitted_charges(charges_csv, atoms, fit)
    fit_report = {
        **fit.to_dict(),
        "grid_point_count": int(len(grid.points)),
        "fitted_charge_center_count": len(atoms),
        "fitted_charge_centers": [atom.name for atom in atoms],
        "calculation_type": pyscf_input["calculation_type"],
        "electrostatic_embedding_applied": embedded_qmmm,
        "embedding_operator": pyscf_input["embedding_operator"],
        "mm_point_charge_count": point_charge_count,
        "mm_point_charge_potential_in_scf_hamiltonian": embedded_qmmm,
        "mm_bonded_and_lennard_jones_terms_in_scf": False,
        "external_mm_potential_included_in_fit": False,
        "esp_grid": config.grid.model_dump(mode="json"),
        "confirmation": QMMESP_CONFIRMATION if point_charges is not None else GAS_RESP_CONFIRMATION,
    }
    fit_report_path = qm_dir / "fit_report.json"
    fit_report_path.write_text(json.dumps(fit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    embedding_summary = None
    if point_charges is not None:
        embedding_summary = point_charges.to_dict()
    return LigandPySCFChargeResult(
        method=method_name,
        qm_dir=qm_dir,
        charged_mol2_path=charged_mol2,
        fitted_charges_csv_path=charges_csv,
        fit_report_path=fit_report_path,
        pyscf_result=pyscf_result.to_dict(),
        fit_result=fit_report,
        grid_point_count=int(len(grid.points)),
        embedding_summary=embedding_summary,
        warnings=list(fit.warnings) + list(pyscf_result.warnings),
    )


def _atom_element(atom: AtomRecord) -> str:
    if atom.element:
        return atom.element
    stripped = atom.name.strip()
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    return stripped[:1].upper()


def _write_fitted_charges(path: Path, atoms: list[AtomRecord], fit: AmberRespFitResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["atom_index", "atom_name", "charge"])
        writer.writeheader()
        for index, (atom, charge) in enumerate(zip(atoms, fit.charges, strict=True), start=1):
            writer.writerow({"atom_index": index, "atom_name": atom.name, "charge": float(charge)})
