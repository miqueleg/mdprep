"""Temporary PDBFixer hydrogenation for xTB histidine-tautomer clusters.

The generated hydrogens are environment atoms for HID/HIE ranking only.  They
are never transferred to the protonation-stage output structure; the normal
Amber/tleap hydrogenation remains authoritative for the prepared system.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from math import dist
from pathlib import Path
import random

from typing import Mapping

from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
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
    # What was requested of OpenMM and what it actually built, per residue.
    requested_variants: tuple[tuple[str, str], ...] = ()
    # Residues handed back exactly as they arrived, because they already
    # carried hydrogens OpenMM's templates cannot account for.
    preserved_residues: tuple[str, ...] = ()
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
            "residues_left_hydrogenated_as_found": list(self.preserved_residues),
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
    assigned_states: Mapping[tuple[str, int, str | None], str] | None = None,
) -> TemporaryHydrogenationResult:
    """Add a complete temporary H environment without changing input atoms.

    ``assigned_states`` carries the protonation PropKa and the manual overrides
    decided, keyed by (chain, residue number, insertion code). Without it the
    hydrogens are placed from a pH alone, by a pKa table that is not the one
    the rest of the run used -- so an aspartate assigned ASH could be built
    ionised, and the declared charge of any cluster containing it would be
    wrong by a proton. Those states are passed through as explicit variants and
    the result is checked against them.
    """

    try:
        from openmm import Platform
        from openmm.app import Modeller, PDBFile
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
        modeller = Modeller(fixer.topology, fixer.positions)
        variants, requested = _requested_variants(modeller.topology, assigned_states)
        # Residues somebody else already hydrogenated are held back here, so
        # that Modeller copies them over instead of building a second set.
        held_back = _hold_back_hydrogenated_residues(modeller.topology, variants)
        # OpenMM Modeller places hydrogens with Python's global random
        # generator before minimizing them.  Restore the caller's RNG state
        # after making this scientific input reproducible.
        random_state = random.getstate()
        random.seed(random_seed)
        try:
            # Modeller, unlike PDBFixer, accepts an explicit variant per
            # residue and reports back what it actually built.
            actual = modeller.addHydrogens(
                pH=float(ph), variants=variants, platform=platform
            )
        finally:
            random.setstate(random_state)
        preserved = _release_held_back_residues(modeller.topology, held_back)
        _verify_variants(modeller.topology, variants, actual)
        with raw_path.open("w", encoding="utf-8") as handle:
            PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)
    except TemporaryHydrogenationError:
        # Already says what went wrong, in this module's own terms.
        raise
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
        requested_variants=requested,
        preserved_residues=preserved,
    )


# OpenMM names the protonation variants of the standard residues exactly as
# Amber does, so an assigned state can be handed straight through.
KNOWN_VARIANTS = {
    "ASH", "ASP", "GLH", "GLU", "LYN", "LYS",
    "HID", "HIE", "HIP", "CYS", "CYM", "CYX",
}


def _requested_variants(
    topology: object,
    assigned_states: Mapping[tuple[str, int, str | None], str] | None,
) -> tuple[list[str | None], tuple[tuple[str, str], ...]]:
    """One entry per residue: the assigned state, or None to let OpenMM pick."""

    variants: list[str | None] = []
    requested: list[tuple[str, str]] = []
    states = assigned_states or {}
    for residue in topology.residues():  # type: ignore[attr-defined]
        key = (
            residue.chain.id or "",
            _integer_or_none(residue.id),
            (residue.insertionCode or "").strip() or None,
        )
        state = states.get(key)
        if state is None or state.upper() not in KNOWN_VARIANTS:
            variants.append(None)
            continue
        variants.append(state.upper())
        requested.append((f"{key[0]}:{residue.name}{key[1]}", state.upper()))
    return variants, tuple(requested)


def _integer_or_none(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


# A residue name OpenMM has no hydrogen definitions for.  A residue wearing it
# is copied over untouched instead of being hydrogenated, and it wears it only
# between the two calls below.
_HELD_BACK_RESIDUE_NAME = "XMD"


def _is_topology_hydrogen(atom: object) -> bool:
    return getattr(getattr(atom, "element", None), "symbol", None) == "H"


def _hold_back_hydrogenated_residues(
    topology: object,
    variants: list[str | None],
) -> dict[int, tuple[str, tuple[str, ...]]]:
    """Hide residues OpenMM would hydrogenate a second time.

    Modeller recognises the hydrogens a residue already has through the
    topology's bonds, and those bonds come from its own residue templates.  A
    hydrogen placed by something else -- a ligand built from the manifest
    SMILES, say -- carries a name no template lists, so no bond is made for it,
    Modeller sees a bare heavy atom and builds a full second set of hydrogens
    beside the first.  A free lysine substrate came out of that with HB2/HB3
    and HB/HBA both present.

    Renaming such a residue to a name OpenMM does not know makes it copy the
    residue over verbatim.  Residues with a requested protonation state are
    left alone: for those Modeller discards the hydrogens it does not
    recognise before building the state that was asked for, which is the whole
    point of passing the states in, and cannot duplicate anything.
    """

    bonded_hydrogens: set[int] = set()
    for first, second in topology.bonds():  # type: ignore[attr-defined]
        for atom, partner in ((first, second), (second, first)):
            if _is_topology_hydrogen(atom) and not _is_topology_hydrogen(partner):
                bonded_hydrogens.add(atom.index)

    held_back: dict[int, tuple[str, tuple[str, ...]]] = {}
    for residue in list(topology.residues()):  # type: ignore[attr-defined]
        if variants[residue.index] is not None:
            continue
        atoms = list(residue.atoms())
        if not any(
            _is_topology_hydrogen(atom) and atom.index not in bonded_hydrogens
            for atom in atoms
        ):
            continue
        held_back[residue.index] = (
            residue.name,
            tuple(atom.name for atom in atoms),
        )
        residue.name = _HELD_BACK_RESIDUE_NAME
    return held_back


def _release_held_back_residues(
    topology: object,
    held_back: dict[int, tuple[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    """Give the held-back residues their names back, having checked them.

    The check is what makes the rename safe to rely on: if a future OpenMM
    learned hydrogen definitions for the placeholder name, the residue would
    come back with atoms it did not go in with and this fails rather than
    writing the duplicates it was meant to prevent.
    """

    preserved: list[str] = []
    for residue in list(topology.residues()):  # type: ignore[attr-defined]
        if residue.index not in held_back:
            continue
        name, atom_names = held_back[residue.index]
        residue.name = name
        observed = tuple(atom.name for atom in residue.atoms())
        display = f"{residue.chain.id or ''}:{name}{residue.id}"
        if observed != atom_names:
            raise TemporaryHydrogenationError(
                f"Temporary hydrogenation was asked to leave {display} as it "
                f"arrived, already hydrogenated, but OpenMM changed its atoms: "
                f"{list(atom_names)} -> {list(observed)}."
            )
        preserved.append(display)
    return tuple(preserved)


def _verify_variants(
    topology: object,
    requested: list[str | None],
    actual: list[str | None] | None,
) -> None:
    """Every state that was asked for must be the one that was built.

    Without this the hydrogens and the declared charge can disagree, which is
    invisible until a quantum program refuses the electron count.
    """

    if actual is None:
        raise TemporaryHydrogenationError(
            "OpenMM did not report which protonation variants it built, so the "
            "temporary hydrogens cannot be checked against the assigned states."
        )
    residues = list(topology.residues())  # type: ignore[attr-defined]
    mismatches = [
        f"{residue.chain.id or ''}:{residue.name}{residue.id} "
        f"requested {want}, built {got}"
        for residue, want, got in zip(residues, requested, actual)
        if want is not None and got is not None and got.upper() != want.upper()
    ]
    if mismatches:
        raise TemporaryHydrogenationError(
            "Temporary hydrogenation did not build the assigned protonation "
            "states: " + "; ".join(mismatches[:8])
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
            residues=build_residues(atoms),
            model_count=1,
            used_model=1,
            warnings=list(reference.warnings),
        ),
        maximum_displacement,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
