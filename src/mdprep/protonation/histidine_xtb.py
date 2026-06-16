"""xTB HID/HIE histidine tautomer selection."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
import json
from pathlib import Path

from mdprep.config.models import ManifestConfig
from mdprep.protonation.histidine_geometry import (
    HistidineGeometryError,
    build_tautomer_cluster_model,
    heavy_atom_distance,
    is_hydrogen_like,
    write_xcontrol_fix_file,
    write_xyz,
)
from mdprep.protonation.xtb_parser import XtbEnergyComparison, compare_hid_hie_energies, parse_xtb_energy_file
from mdprep.protonation.xtb_runner import XtbExecutionError, run_xtb
from mdprep.structure.classify import is_likely_ligand_or_cofactor, is_water_residue
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


class HistidineXtbError(ValueError):
    """Raised when HID/HIE xTB selection cannot be completed."""


@dataclass(frozen=True)
class HistidineXtbSelection:
    residue: ResidueRecord
    hid_energy_hartree: float
    hie_energy_hartree: float
    delta_kcal_mol: float
    selected_state: str
    close_call: bool
    warnings: list[str]
    mode: str
    model: str
    executable: str
    cluster_charge: int
    output_dir: Path
    temporary_water_hydrogens_added: int
    waters_modified_for_xtb_only: list[dict[str, object]]
    final_pdb_modified_by_temporary_water_hydrogens: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "chain": self.residue.id.chain_id,
            "resid": self.residue.id.resid,
            "icode": self.residue.id.icode,
            "hid_energy_hartree": self.hid_energy_hartree,
            "hie_energy_hartree": self.hie_energy_hartree,
            "delta_kcal_mol": self.delta_kcal_mol,
            "selected_state": self.selected_state,
            "close_call": self.close_call,
            "warnings": self.warnings,
            "mode": self.mode,
            "model": self.model,
            "executable": self.executable,
            "cluster_charge": self.cluster_charge,
            "output_dir": str(self.output_dir),
            "temporary_water_hydrogens_added": self.temporary_water_hydrogens_added,
            "waters_modified_for_xtb_only": self.waters_modified_for_xtb_only,
            "final_pdb_modified_by_temporary_water_hydrogens": self.final_pdb_modified_by_temporary_water_hydrogens,
        }


def select_histidine_tautomer(
    structure: PdbStructure,
    histidine: ResidueRecord,
    manifest: ManifestConfig,
    *,
    work_dir: str | Path,
    planned_states: dict[int, str],
) -> HistidineXtbSelection:
    config = manifest.protonation.histidine.xtb
    hist_dir = Path(work_dir) / _histidine_dir_name(histidine)
    hist_dir.mkdir(parents=True, exist_ok=True)
    cluster_residues = build_histidine_cluster(
        structure,
        histidine,
        cutoff_angstrom=config.cutoff_angstrom,
    )
    cluster_charge = estimate_cluster_charge(cluster_residues, manifest, planned_states)
    hid_xyz = hist_dir / "HID.xyz"
    hie_xyz = hist_dir / "HIE.xyz"
    try:
        cluster_kwargs = {
            "add_missing_water_hydrogens": config.add_missing_water_hydrogens,
            "water_oh_distance_angstrom": config.water_oh_distance_angstrom,
            "water_hoh_angle_degrees": config.water_hoh_angle_degrees,
        }
        hid_model = build_tautomer_cluster_model(
            cluster_residues,
            histidine,
            tautomer="HID",
            **cluster_kwargs,
        )
        hie_model = build_tautomer_cluster_model(
            cluster_residues,
            histidine,
            tautomer="HIE",
            **cluster_kwargs,
        )
        write_xyz(hid_model.atoms, hid_xyz, comment="HID")
        write_xyz(hie_model.atoms, hie_xyz, comment="HIE")
        hid_input = hist_dir / "HID_xtb.inp"
        hie_input = hist_dir / "HIE_xtb.inp"
        write_xcontrol_fix_file(hid_model.fixed_atom_indices, hid_input)
        write_xcontrol_fix_file(hie_model.fixed_atom_indices, hie_input)
    except HistidineGeometryError as exc:
        raise HistidineXtbError(str(exc)) from exc

    hid_stdout = hist_dir / "HID_xtb_stdout.txt"
    hie_stdout = hist_dir / "HIE_xtb_stdout.txt"
    hid_stderr = hist_dir / "HID_xtb_stderr.txt"
    hie_stderr = hist_dir / "HIE_xtb_stderr.txt"
    try:
        hid_run = run_xtb(
            config=config,
            xyz_path=hid_xyz,
            work_dir=hist_dir,
            cluster_charge=cluster_charge,
            stdout_path=hid_stdout,
            stderr_path=hid_stderr,
            input_path=hid_input if config.mode == "opt" and hid_model.fixed_atom_indices else None,
        )
        hie_run = run_xtb(
            config=config,
            xyz_path=hie_xyz,
            work_dir=hist_dir,
            cluster_charge=cluster_charge,
            stdout_path=hie_stdout,
            stderr_path=hie_stderr,
            input_path=hie_input if config.mode == "opt" and hie_model.fixed_atom_indices else None,
        )
        hid_energy = parse_xtb_energy_file(hid_stdout)
        hie_energy = parse_xtb_energy_file(hie_stdout)
    except (XtbExecutionError, ValueError) as exc:
        raise HistidineXtbError(str(exc)) from exc

    comparison = compare_hid_hie_energies(
        hid_energy_hartree=hid_energy,
        hie_energy_hartree=hie_energy,
        close_call_kcal_mol=config.energy_close_call_kcal_mol,
    )
    temporary_water_records = [record.to_dict() for record in hid_model.temporary_water_hydrogens]
    model_warnings = _unique_strings(chain(hid_model.warnings, hie_model.warnings))
    selection = HistidineXtbSelection(
        residue=histidine,
        hid_energy_hartree=comparison.hid_energy_hartree,
        hie_energy_hartree=comparison.hie_energy_hartree,
        delta_kcal_mol=comparison.delta_kcal_mol,
        selected_state=comparison.selected_state,
        close_call=comparison.close_call,
        warnings=comparison.warnings + model_warnings,
        mode=config.mode,
        model=config.model,
        executable=hid_run.command_result.command[0],
        cluster_charge=cluster_charge,
        output_dir=hist_dir,
        temporary_water_hydrogens_added=sum(
            int(record["hydrogens_added"]) for record in temporary_water_records
        ),
        waters_modified_for_xtb_only=temporary_water_records,
        final_pdb_modified_by_temporary_water_hydrogens=False,
    )
    cluster_summary = {
        "HID": hid_model.to_dict(),
        "HIE": hie_model.to_dict(),
        "model": "CA-truncated hydrogen-preserving cluster with fixed CA/capping atoms",
        "temporary_water_hydrogens_for_xtb_only": {
            "hydrogens_added": selection.temporary_water_hydrogens_added,
            "waters_modified": selection.waters_modified_for_xtb_only,
            "final_pdb_modified": False,
        },
    }
    (hist_dir / "cluster_model.json").write_text(
        json.dumps(cluster_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (hist_dir / "energies.json").write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selection


def build_histidine_cluster(
    structure: PdbStructure,
    histidine: ResidueRecord,
    *,
    cutoff_angstrom: float,
) -> list[ResidueRecord]:
    hist_heavy = [atom for atom in histidine.atoms if not is_hydrogen_like(atom)]
    if not hist_heavy:
        raise HistidineXtbError(f"Histidine {histidine.id.display()} has no heavy atoms.")
    cluster: list[ResidueRecord] = []
    for residue in structure.residues:
        if residue is histidine:
            cluster.append(residue)
            continue
        for atom in residue.atoms:
            if is_hydrogen_like(atom):
                continue
            if any(heavy_atom_distance(atom, hist_atom) <= cutoff_angstrom for hist_atom in hist_heavy):
                cluster.append(residue)
                break
    return cluster


def estimate_cluster_charge(
    cluster_residues: list[ResidueRecord],
    manifest: ManifestConfig,
    planned_states: dict[int, str],
) -> int:
    charge = 0
    ligand_charges = _ligand_charge_by_residue(cluster_residues, manifest)
    for residue in cluster_residues:
        if id(residue) in ligand_charges:
            charge += ligand_charges[id(residue)]
            continue
        resname = planned_states.get(id(residue), residue.id.resname)
        if is_water_residue(residue):
            continue
        if is_likely_ligand_or_cofactor(residue):
            raise HistidineXtbError(
                f"Unknown heterogen {residue.id.display()} encountered in xTB cluster; configure it under ligands."
            )
        charge += {
            "ASP": -1,
            "GLU": -1,
            "CYM": -1,
            "ASH": 0,
            "GLH": 0,
            "CYS": 0,
            "CYX": 0,
            "HID": 0,
            "HIE": 0,
            "HIS": 0,
            "LYS": 1,
            "ARG": 1,
            "HIP": 1,
            "LYN": 0,
        }.get(resname, 0)
    return int(charge)


def _ligand_charge_by_residue(
    residues: list[ResidueRecord],
    manifest: ManifestConfig,
) -> dict[int, int]:
    result: dict[int, int] = {}
    structure = PdbStructure(
        path=Path("cluster"),
        atoms=[atom for residue in residues for atom in residue.atoms],
        residues=residues,
        model_count=1,
    )
    for ligand in manifest.ligands:
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError:
            continue
        result[id(residue)] = ligand.net_charge
    return result


def _histidine_dir_name(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "blank"
    return f"{chain}_HIS{residue.id.resid}{residue.id.icode or ''}"


def _unique_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
