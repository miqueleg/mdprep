"""Temporary PDBFixer hydrogenation for xTB histidine-tautomer clusters.

The generated hydrogens are environment atoms for HID/HIE ranking only.  They
are never transferred to the protonation-stage output structure; the normal
Amber/tleap hydrogenation remains authoritative for the prepared system.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from math import dist
from pathlib import Path
import random

from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


class TemporaryHydrogenationError(ValueError):
    """Raised when a safe temporary hydrogenated environment cannot be built."""


@dataclass(frozen=True)
class TemporaryHydrogenationResult:
    backend: str
    backend_version: str | None
    platform: str
    ph: float
    random_seed: int
    input_path: Path
    raw_output_path: Path
    restored_output_path: Path
    structure: PdbStructure
    added_hydrogen_count: int
    maximum_original_atom_displacement_angstrom: float
    final_prepared_pdb_modified: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "platform": self.platform,
            "ph": self.ph,
            "random_seed": self.random_seed,
            "input_path": str(self.input_path),
            "raw_output_path": str(self.raw_output_path),
            "restored_output_path": str(self.restored_output_path),
            "added_hydrogen_count": self.added_hydrogen_count,
            "maximum_original_atom_displacement_angstrom": (
                self.maximum_original_atom_displacement_angstrom
            ),
            "final_prepared_pdb_modified": self.final_prepared_pdb_modified,
            "interpretation": (
                "PDBFixer hydrogens complete the local xTB HID/HIE environment only. "
                "They are not copied into the protonation-stage or final prepared PDB."
            ),
        }


def add_temporary_protein_hydrogens(
    structure: PdbStructure,
    *,
    ph: float,
    work_dir: str | Path,
    random_seed: int = 20260722,
) -> TemporaryHydrogenationResult:
    """Add a complete temporary H environment without changing input atoms."""

    try:
        from openmm import Platform
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception as exc:
        raise TemporaryHydrogenationError(
            "protonation.histidine.xtb.add_missing_protein_hydrogens was requested, "
            "but PDBFixer and OpenMM are unavailable. Install PDBFixer or set the "
            "option to false and provide a fully hydrogenated input model."
        ) from exc

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "histidine_xtb_environment_input.pdb"
    raw_path = work / "histidine_xtb_environment.pdbfixer_raw.pdb"
    restored_path = work / "histidine_xtb_environment.hydrogenated.pdb"
    write_pdb(structure, input_path)

    try:
        platform_name = "Reference"
        platform = Platform.getPlatformByName(platform_name)
        fixer = PDBFixer(filename=str(input_path), platform=platform)
        # OpenMM Modeller places hydrogens with Python's global random
        # generator before minimizing them.  Restore the caller's RNG state
        # after making this scientific input reproducible.
        random_state = random.getstate()
        random.seed(random_seed)
        try:
            fixer.addMissingHydrogens(float(ph))
        finally:
            random.setstate(random_state)
        with raw_path.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    except Exception as exc:
        raise TemporaryHydrogenationError(
            f"PDBFixer could not build the temporary xTB hydrogen environment: {exc}"
        ) from exc

    raw = read_pdb(raw_path)
    restored, maximum_displacement = _restore_and_validate(
        structure,
        raw,
        output_path=restored_path,
    )
    write_pdb(restored, restored_path)
    added = len(restored.atoms) - len(structure.atoms)
    if added <= 0:
        raise TemporaryHydrogenationError(
            "PDBFixer did not add any temporary hydrogens to a dehydrogenated "
            "histidine-ranking environment."
        )
    return TemporaryHydrogenationResult(
        backend="pdbfixer",
        backend_version=_package_version("pdbfixer"),
        platform=platform_name,
        ph=float(ph),
        random_seed=random_seed,
        input_path=input_path,
        raw_output_path=raw_path,
        restored_output_path=restored_path,
        structure=restored,
        added_hydrogen_count=added,
        maximum_original_atom_displacement_angstrom=maximum_displacement,
    )


def _restore_and_validate(
    reference: PdbStructure,
    observed: PdbStructure,
    *,
    output_path: Path,
) -> tuple[PdbStructure, float]:
    if len(observed.residues) != len(reference.residues):
        raise TemporaryHydrogenationError(
            "Temporary PDBFixer hydrogenation changed the residue count: "
            f"{len(reference.residues)} -> {len(observed.residues)}."
        )
    atoms: list[AtomRecord] = []
    maximum_displacement = 0.0
    serial = 0
    for expected, actual in zip(reference.residues, observed.residues, strict=True):
        expected_key = (expected.id.chain_id, expected.id.resid, expected.id.icode)
        actual_key = (actual.id.chain_id, actual.id.resid, actual.id.icode)
        if expected_key != actual_key or expected.id.resname != actual.id.resname:
            raise TemporaryHydrogenationError(
                "Temporary PDBFixer hydrogenation changed residue identity/order: "
                f"expected {expected.id.display()}, found {actual.id.display()}."
            )
        expected_by_name = {atom.name: atom for atom in expected.atoms}
        if len(expected_by_name) != len(expected.atoms):
            raise TemporaryHydrogenationError(
                f"Input residue {expected.id.display()} has duplicate atom names."
            )
        actual_names = {atom.name for atom in actual.atoms}
        missing = sorted(set(expected_by_name) - actual_names)
        if missing:
            raise TemporaryHydrogenationError(
                f"Temporary PDBFixer hydrogenation removed atoms from "
                f"{expected.id.display()}: {missing}"
            )
        residue_record_name = "HETATM" if expected.record_names == {"HETATM"} else "ATOM"
        for atom in actual.atoms:
            serial += 1
            original = expected_by_name.get(atom.name)
            if original is not None:
                displacement = dist(
                    (original.x, original.y, original.z),
                    (atom.x, atom.y, atom.z),
                )
                maximum_displacement = max(maximum_displacement, displacement)
                if displacement > 0.002:
                    raise TemporaryHydrogenationError(
                        "Temporary PDBFixer hydrogenation moved an original atom by "
                        f"{displacement:.6f} A: {original.atom_identity}."
                    )
            elif (atom.element or "").strip().upper() != "H":
                raise TemporaryHydrogenationError(
                    "Temporary PDBFixer hydrogenation added a non-hydrogen atom "
                    f"to {expected.id.display()}: {atom.name}."
                )
            atoms.append(
                replace(
                    atom,
                    serial=serial,
                    resname=expected.id.resname,
                    chain_id=expected.id.chain_id,
                    resid=expected.id.resid,
                    icode=expected.id.icode,
                    record_name=(original.record_name if original else residue_record_name),
                    occupancy=(original.occupancy if original else atom.occupancy),
                    bfactor=(original.bfactor if original else atom.bfactor),
                    element=(original.element if original else atom.element),
                )
            )
    return (
        PdbStructure(
            path=output_path,
            atoms=atoms,
            residues=_build_residues(atoms),
            model_count=1,
            used_model=1,
            warnings=list(reference.warnings),
        ),
        maximum_displacement,
    )


def _build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    return [
        ResidueRecord(
            id=ResidueId(chain_id=key[0], resname=key[1], resid=key[2], icode=key[3]),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, (key, residue_atoms) in enumerate(grouped.items())
    ]


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
