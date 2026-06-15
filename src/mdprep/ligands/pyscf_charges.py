"""PySCF-based ligand RESP/QMMESP charge derivation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mdprep.ambertools.mol2 import write_mol2_with_charges
from mdprep.charges.esp_grid import EspGridError, generate_connolly_grid, write_grid_xyz
from mdprep.charges.resp_fit import RespFitError, RespFitResult, fit_resp_charges
from mdprep.config.models import LigandConfig
from mdprep.ligands.extract import ExtractedLigand
from mdprep.qm.point_charges import PointChargeSelection, write_point_charge_files
from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM, PySCFEspError, evaluate_ligand_esp, write_esp_values
from mdprep.qm.pyscf_runner import PySCFResult, PySCFRunnerError, run_pyscf_scf
from mdprep.structure.models import AtomRecord


class LigandPySCFChargeError(ValueError):
    """Raised when PySCF ligand charge derivation fails."""


QMMESP_CONFIRMATION = (
    "MM point charges were used only for electrostatic embedding/polarization of the "
    "target ligand QM density. The ESP fitting target contains only the polarized target "
    "ligand QM electrostatic potential. Environment point charges were not fitted and "
    "were not written to the ligand mol2."
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
    if config.resp_fitting.backend not in {"native", "auto"}:
        raise LigandPySCFChargeError("Only the native RESP/ESP fitting backend is implemented in mdprep v0.1.")
    qm_dir = Path(output_dir) / "ligands" / ligand.id / "qm" / method_name
    qm_dir.mkdir(parents=True, exist_ok=True)
    atoms = extracted.atoms
    elements = [_atom_element(atom) for atom in atoms]
    coords = np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    scf_charge = ligand.net_charge if config.scf_charge is None else config.scf_charge
    scf_spin = ligand.multiplicity - 1 if config.scf_spin is None else config.scf_spin
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
        "point_charge_count": 0 if point_charges is None else point_charges.total_after_cutoff,
    }
    (qm_dir / "pyscf_input.json").write_text(json.dumps(pyscf_input, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        grid = generate_connolly_grid(
            elements=elements,
            coordinates=coords,
            vdw_scale_factors=config.grid.vdw_scale_factors,
            points_per_atom_per_shell=config.grid.points_per_atom_per_shell,
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
        )
        esp = evaluate_ligand_esp(
            mol=pyscf_result.mol,
            mf=pyscf_result.mf,
            grid_coordinates_angstrom=grid.points,
        )
        fit = fit_resp_charges(
            atom_coordinates=coords * BOHR_PER_ANGSTROM,
            grid_coordinates=grid.points * BOHR_PER_ANGSTROM,
            esp_values=esp,
            total_charge=float(ligand.net_charge),
            restraint=config.resp_fitting.restraint,
            restraint_a=config.resp_fitting.restraint_a,
            restraint_b=config.resp_fitting.restraint_b,
            max_iter=config.resp_fitting.max_iter,
            convergence=config.resp_fitting.convergence,
        )
    except (EspGridError, PySCFRunnerError, PySCFEspError, RespFitError) as exc:
        raise LigandPySCFChargeError(str(exc)) from exc

    write_grid_xyz(grid, qm_dir / "esp_grid.xyz")
    write_esp_values(esp, str(qm_dir / "esp_values.dat"))
    (qm_dir / "pyscf_stdout.txt").write_text(pyscf_result.stdout, encoding="utf-8")
    (qm_dir / "pyscf_stderr.txt").write_text(pyscf_result.stderr, encoding="utf-8")
    (qm_dir / "pyscf_result.json").write_text(json.dumps(pyscf_result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    charged_mol2 = Path(output_mol2_path)
    write_mol2_with_charges(provisional_mol2_path, [float(charge) for charge in fit.charges], charged_mol2)
    charges_csv = qm_dir / "fitted_charges.csv"
    _write_fitted_charges(charges_csv, atoms, fit)
    fit_report = {
        **fit.to_dict(),
        "grid_point_count": int(len(grid.points)),
        "fitted_charge_center_count": len(atoms),
        "fitted_charge_centers": [atom.name for atom in atoms],
        "external_mm_potential_included_in_fit": False,
        "confirmation": QMMESP_CONFIRMATION if point_charges is not None else GAS_RESP_CONFIRMATION,
    }
    fit_report_path = qm_dir / "fit_report.json"
    fit_report_path.write_text(json.dumps(fit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    embedding_summary = None
    if point_charges is not None:
        write_point_charge_files(
            point_charges,
            csv_path=qm_dir / "mm_point_charges.csv",
            xyz_path=qm_dir / "mm_point_charges.xyz",
            summary_path=qm_dir / "embedding_summary.json",
        )
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


def _write_fitted_charges(path: Path, atoms: list[AtomRecord], fit: RespFitResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["atom_index", "atom_name", "charge"])
        writer.writeheader()
        for index, (atom, charge) in enumerate(zip(atoms, fit.charges, strict=True), start=1):
            writer.writerow({"atom_index": index, "atom_name": atom.name, "charge": float(charge)})
