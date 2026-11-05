"""A bonded site minimised with the provisional force field, to be fitted there.

MCPB.py's Seminario fit reads bond and angle force constants off the Hessian,
and that is only meaningful at a stationary point: elsewhere the Hessian
holds the curvature of a slope, and a bond can come out with a negative force
constant, which MCPB.py writes as zero. A crystal site with only its
hydrogens relaxed is usually close enough. A predicted one often is not --
an AlphaFold iron site fitted as a singlet had three Fe-O bonds with no
restoring force.

``geometry_source: ff_minimized_site`` takes the fit somewhere better without
a quantum optimisation: the provisional system mdprep builds before MCPB.py,
with the metal as a nonbonded ion, is minimised over every residue within a
radius of the metal and held everywhere else. The Hessian and the ESP are
taken there, and the parameters are then built onto the original geometry --
the structure that is prepared is the one that came in.

The minimised site has to still be the site, or its parameters describe
another one. Preparation stops when a donor leaves the first shell or another
atom joins it, when a bond to the metal changes by more than a set amount, or
when the site's heavy atoms move too far; each limit is in
``mcpb.pyscf.site_minimization``, and the report records every distance.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, replace
from math import dist, sqrt
from pathlib import Path
from typing import Sequence

from mdprep.config.models import McpbSiteMinimizationConfig
from mdprep.metals.c4 import C4PostprocessError, _prmtop_flag_strings, apply_12_6_4_c4
from mdprep.metals.coordination import ResolvedMetalSite
from mdprep.refinement.openmm_minimizer import (
    OpenmmRefinementError,
    run_openmm_site_minimization,
)
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, build_residues

# Atoms that can bind a metal: a new one in the first shell is a new bond.
DONOR_ELEMENTS = frozenset({"N", "O", "S", "F", "CL", "BR", "I"})


class SiteMinimizationError(ValueError):
    """Raised when a site cannot be minimised, or moved too far to be fitted."""


@dataclass(frozen=True)
class SiteMinimizationResult:
    # The structure with its site minimised: what MCPB.py cuts its models from.
    structure: PdbStructure
    prmtop_path: Path
    # The provisional system's coordinates with the site minimised, for the
    # large model's electrostatic embedding.
    inpcrd_path: Path
    report_path: Path
    report: dict[str, object]


def minimize_site_for_fit(
    site: ResolvedMetalSite,
    *,
    structure: PdbStructure,
    prmtop: str | Path,
    inpcrd: str | Path,
    config: McpbSiteMinimizationConfig,
    cutoff_angstrom: float,
    c4_types: Sequence[str] = (),
    water_model: str = "TIP3P",
    work_dir: str | Path,
) -> SiteMinimizationResult:
    """Minimise the site with the provisional nonbonded model, and check it.

    ``structure`` is the provisional system's, atom for atom: the pre-MCPB
    tleap build restored to the input's identities. ``c4_types`` are the
    12-6-4 ion types to give their C4 term first, without which a 12-6-4 ion
    is a 12-6 one with the wrong radius.
    """

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    system = Path(prmtop)
    c4_report: dict[str, object] | None = None
    if c4_types:
        system = work / "system.site_minimization.prmtop"
        shutil.copyfile(prmtop, system)
        try:
            c4_report = apply_12_6_4_c4(
                system,
                atom_types=list(c4_types),
                water_model=water_model,
                work_dir=work / "c4",
            ).to_dict()
        except C4PostprocessError as exc:
            raise SiteMinimizationError(
                f"Could not add the 12-6-4 C4 term to the system the site of "
                f"{site.config.id!r} is minimised in: {exc}"
            ) from exc

    names = _prmtop_flag_strings(system.read_text(encoding="utf-8", errors="replace"), "ATOM_NAME")
    if names != [atom.name.strip() for atom in structure.atoms]:
        raise SiteMinimizationError(
            "The provisional system does not hold the structure's atoms in the "
            f"structure's order ({len(names)} atoms against {len(structure.atoms)}), so "
            "its minimised coordinates cannot be put back on the structure."
        )

    index_of = {atom.atom_identity: index for index, atom in enumerate(structure.atoms)}
    metals = [index_of[ion.atom.atom_identity] for ion in site.ions]
    near = {
        atom.residue_key
        for atom in structure.atoms
        if any(dist(_xyz(atom), _xyz(structure.atoms[metal])) <= config.radius_angstrom for metal in metals)
    }
    movable = [index for index, atom in enumerate(structure.atoms) if atom.residue_key in near]
    try:
        run = run_openmm_site_minimization(
            prmtop_path=system,
            inpcrd_path=inpcrd,
            movable=movable,
            tolerance_kj_mol_per_nm=config.tolerance_kj_mol_per_nm,
            max_iterations=config.max_iterations,
            nonbonded_cutoff_angstrom=config.nonbonded_cutoff_angstrom,
        )
    except OpenmmRefinementError as exc:
        raise SiteMinimizationError(
            f"Minimising the site of {site.config.id!r} failed: {exc}"
        ) from exc

    moved = set(movable)
    atoms = [
        replace(atom, x=x, y=y, z=z, original_line="") if index in moved else atom
        for index, (atom, (x, y, z)) in enumerate(zip(structure.atoms, run.coordinates_angstrom))
    ]
    minimized = replace(structure, atoms=atoms, residues=build_residues(atoms))

    report = _site_changes(site, structure, minimized, index_of, config, cutoff_angstrom)
    report.update(
        {
            "site_id": site.config.id,
            "radius_angstrom": config.radius_angstrom,
            "residues_minimized": len(near),
            "minimization": run.to_dict(),
            "limits": config.model_dump(mode="json"),
            "c4": c4_report,
        }
    )
    report_path = work / "site_minimization.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["problems"]:
        raise SiteMinimizationError(
            f"Minimising the site of {site.config.id!r} with the provisional nonbonded "
            "force field changed it too much to fit it there: "
            + "; ".join(report["problems"])  # type: ignore[arg-type]
            + ". Fit it at its own geometry (geometry_source: qmmm_refinement), or "
            "loosen mcpb.pyscf.site_minimization if the change is wanted. Every "
            f"distance is in {report_path}."
        )

    inpcrd_path = work / "system.site_minimized.inpcrd"
    _write_inpcrd(Path(prmtop), Path(inpcrd), run.coordinates_angstrom, inpcrd_path)
    return SiteMinimizationResult(
        structure=minimized,
        prmtop_path=system,
        inpcrd_path=inpcrd_path,
        report_path=report_path,
        report=report,
    )


def _site_changes(
    site: ResolvedMetalSite,
    before: PdbStructure,
    after: PdbStructure,
    index_of: dict[tuple, int],
    config: McpbSiteMinimizationConfig,
    cutoff_angstrom: float,
) -> dict[str, object]:
    """How far the site moved, bond by bond, and whether it is still the site."""

    problems: list[str] = []
    bonds = []
    for bond in site.bonds:
        metal = index_of[bond.ion.atom.atom_identity]
        donor = index_of[bond.coordinator.atom_identity]
        start = dist(_xyz(before.atoms[metal]), _xyz(before.atoms[donor]))
        end = dist(_xyz(after.atoms[metal]), _xyz(after.atoms[donor]))
        label = f"{_label(bond.ion.atom)}-{_label(bond.coordinator)}"
        bonds.append({"bond": label, "before_angstrom": round(start, 4), "after_angstrom": round(end, 4)})
        if end > cutoff_angstrom:
            problems.append(f"{label} is {end:.2f} A, outside the {cutoff_angstrom:.2f} A bond cutoff")
        elif abs(end - start) > config.max_bond_change_angstrom:
            problems.append(f"{label} went from {start:.2f} to {end:.2f} A")

    bonded = {bond.coordinator.atom_identity for bond in site.bonds}
    metal_residues = {ion.atom.residue_key for ion in site.ions}
    for ion in site.ions:
        metal = after.atoms[index_of[ion.atom.atom_identity]]
        for atom in after.atoms:
            if (
                atom.atom_identity in bonded
                or atom.residue_key in metal_residues
                or (atom.element or "").strip().upper() not in DONOR_ELEMENTS
            ):
                continue
            distance = dist(_xyz(metal), _xyz(atom))
            if distance <= cutoff_angstrom:
                problems.append(
                    f"{_label(atom)} joined the first shell of {_label(ion.atom)} at {distance:.2f} A"
                )

    residues: set[tuple] = set(metal_residues)
    residues |= {bond.coordinator.residue_key for bond in site.bonds}
    residues |= {_key(residue.id) for residue in site.additional_residues}
    shifts = [
        (dist(_xyz(start), _xyz(end)), end)
        for start, end in zip(before.atoms, after.atoms)
        if start.residue_key in residues and not is_hydrogen(start)
    ]
    rmsd = sqrt(sum(shift * shift for shift, _ in shifts) / len(shifts)) if shifts else 0.0
    largest, where = max(shifts, key=lambda item: item[0]) if shifts else (0.0, None)
    if rmsd > config.max_rmsd_angstrom:
        problems.append(f"the site's heavy atoms moved {rmsd:.2f} A RMS")
    if largest > config.max_shift_angstrom:
        problems.append(f"{_label(where)} moved {largest:.2f} A")
    return {
        "bonds": bonds,
        "site_heavy_atom_rmsd_angstrom": round(rmsd, 4),
        "site_largest_shift_angstrom": round(largest, 4),
        "site_largest_shift_atom": _label(where) if where is not None else None,
        "problems": problems,
    }


def put_back_original_coordinates(path: str | Path, original: PdbStructure) -> int:
    """Rewrite a PDB of the fitted structure with the original coordinates.

    The fit was taken at the minimised site; the structure that is prepared
    is the one that came in. Atoms are matched by serial, which the fit's
    structure shares with the original, and must be of the same element --
    not the same name, since MCPB.py renames a metal's atom. Returns how many
    atoms were moved back.
    """

    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.writer import write_pdb

    fitted = read_pdb(path)
    by_serial = {atom.serial: atom for atom in original.atoms}
    atoms: list[AtomRecord] = []
    moved = 0
    for atom in fitted.atoms:
        source = by_serial.get(atom.serial)
        if source is None or (source.element or "").strip().upper() != (
            atom.element or ""
        ).strip().upper():
            raise SiteMinimizationError(
                f"Could not find atom {atom.serial} ({atom.name}) of the fitted structure "
                "in the original, to put its original coordinates back."
            )
        if _xyz(source) != _xyz(atom):
            moved += 1
        atoms.append(replace(atom, x=source.x, y=source.y, z=source.z, original_line=""))
    write_pdb(replace(fitted, atoms=atoms, residues=build_residues(atoms)), path)
    return moved


def _write_inpcrd(
    prmtop: Path,
    inpcrd: Path,
    coordinates: Sequence[tuple[float, float, float]],
    destination: Path,
) -> None:
    try:
        import parmed
    except ImportError as exc:  # AmberTools ships it; the metal stage needs it anyway
        raise SiteMinimizationError("Writing the minimised coordinates requires ParmEd.") from exc
    system = parmed.load_file(str(prmtop), str(inpcrd))
    system.coordinates = [list(xyz) for xyz in coordinates]
    system.save(str(destination), format="rst7", overwrite=True)


def _xyz(atom: AtomRecord) -> tuple[float, float, float]:
    return (atom.x, atom.y, atom.z)


def _key(residue: ResidueId) -> tuple:
    return (residue.chain_id, residue.resname, residue.resid, residue.icode)


def _label(atom: AtomRecord | None) -> str:
    if atom is None:
        return "?"
    return f"{atom.chain_id or '<blank>'}:{atom.resname}{atom.resid}{atom.icode or ''}@{atom.name.strip()}"
