"""Explicit structure repair with strict identity and geometry validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from math import dist
from pathlib import Path

import numpy as np

from mdprep.config.models import ManifestConfig, ResidueSelector
from mdprep.structure.classify import is_standard_protein_residue
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


class StructureRepairError(ValueError):
    """Raised when requested coordinate repair is unavailable or ambiguous."""


@dataclass(frozen=True)
class StructureRepairResult:
    backend: str
    backend_version: str
    input_path: Path
    sequence_source_path: Path | None
    raw_output_path: Path
    output_path: Path
    structure: PdbStructure
    missing_residues_detected: tuple[dict[str, object], ...]
    missing_heavy_atoms_detected: tuple[dict[str, object], ...]
    added_atom_count: int
    peptide_bond_checks: tuple[dict[str, object], ...]
    random_seed: int
    platform: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "input_path": str(self.input_path),
            "sequence_source_path": (
                str(self.sequence_source_path)
                if self.sequence_source_path is not None
                else None
            ),
            "raw_output_path": str(self.raw_output_path),
            "output_path": str(self.output_path),
            "missing_residues_detected": list(self.missing_residues_detected),
            "missing_heavy_atoms_detected": list(
                self.missing_heavy_atoms_detected
            ),
            "added_atom_count": self.added_atom_count,
            "peptide_bond_checks": list(self.peptide_bond_checks),
            "random_seed": self.random_seed,
            "platform": self.platform,
            "warnings": list(self.warnings),
        }


def repair_structure(
    manifest: ManifestConfig,
    *,
    output_path: str | Path,
) -> StructureRepairResult | None:
    config = manifest.structure.repair
    if config.backend == "none":
        return None
    if config.backend != "pdbfixer":
        raise StructureRepairError(
            f"Unsupported structure repair backend: {config.backend}"
        )
    try:
        from openmm import Platform
        from openmm.app import PDBFile
        from pdbfixer import PDBFixer
    except Exception as exc:
        raise StructureRepairError(
            "structure.repair.backend: pdbfixer was requested, but PDBFixer "
            "and OpenMM are unavailable. Install PDBFixer or disable repair."
        ) from exc

    input_path = Path(manifest.project.input_structure)
    sequence_path = (
        Path(config.sequence_source) if config.sequence_source is not None else None
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input structure not found: {input_path}")
    if sequence_path is not None and not sequence_path.is_file():
        raise FileNotFoundError(
            f"Structure repair sequence source not found: {sequence_path}"
        )

    try:
        platform = Platform.getPlatformByName(config.platform)
        fixer = PDBFixer(filename=str(input_path), platform=platform)
        if sequence_path is not None:
            sequence_fixer = PDBFixer(filename=str(sequence_path), platform=platform)
            fixer.sequences = sequence_fixer.sequences
        fixer.findMissingResidues()
        detected_residues = _detected_missing_residues(fixer)
        _validate_expected_missing_residues(
            detected_residues,
            expected=config.missing_residues,
        )
        fixer.findMissingAtoms()
        detected_atoms = _detected_missing_atoms(fixer)
        if detected_atoms and not config.add_missing_heavy_atoms:
            formatted = ", ".join(
                f"{item['chain']}:{item['resname']}{item['resid']} "
                f"({','.join(item['atom_names'])})"
                for item in detected_atoms
            )
            raise StructureRepairError(
                "Missing protein heavy atoms were detected but "
                "structure.repair.add_missing_heavy_atoms is false: "
                + formatted
            )
        fixer.addMissingAtoms(seed=config.random_seed)
    except StructureRepairError:
        raise
    except Exception as exc:
        raise StructureRepairError(f"PDBFixer structure repair failed: {exc}") from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(output.stem + ".pdbfixer_raw.pdb")
    with raw_output.open("w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)

    original = read_pdb(
        input_path,
        altloc_policy=manifest.structure.altloc_policy,
    )
    repaired_raw = read_pdb(raw_output)
    repaired = _restore_existing_atom_metadata(
        original,
        repaired_raw,
        output_path=output,
    )
    checks = _validate_repaired_peptide_bonds(
        repaired,
        expected=config.missing_residues,
        minimum=config.peptide_bond_min_angstrom,
        maximum=config.peptide_bond_max_angstrom,
        require_trans=config.require_trans_peptide_bonds,
        minimum_abs_omega=config.trans_peptide_min_abs_degrees,
    )
    write_pdb(repaired, output)
    warnings = (
        "PDBFixer-generated missing coordinates are deterministic for the "
        "recorded seed/platform but remain modeled coordinates; minimize the "
        "final Amber system before production MD.",
    )
    return StructureRepairResult(
        backend="pdbfixer",
        backend_version=_package_version("pdbfixer"),
        input_path=input_path,
        sequence_source_path=sequence_path,
        raw_output_path=raw_output,
        output_path=output,
        structure=repaired,
        missing_residues_detected=tuple(detected_residues),
        missing_heavy_atoms_detected=tuple(detected_atoms),
        added_atom_count=len(repaired.atoms) - len(original.atoms),
        peptide_bond_checks=tuple(checks),
        random_seed=config.random_seed,
        platform=config.platform,
        warnings=warnings,
    )


def _detected_missing_residues(fixer: object) -> list[dict[str, object]]:
    topology = fixer.topology  # type: ignore[attr-defined]
    chains = list(topology.chains())
    result: list[dict[str, object]] = []
    for key, names in sorted(fixer.missingResidues.items()):  # type: ignore[attr-defined]
        if not names:
            continue
        chain_index, insertion_index = key
        if chain_index >= len(chains):
            raise StructureRepairError(
                f"PDBFixer reported invalid missing-residue chain index {chain_index}."
            )
        chain = chains[chain_index]
        residues = list(chain.residues())
        previous_id = (
            _integer_residue_id(residues[insertion_index - 1].id)
            if insertion_index > 0
            else None
        )
        next_id = (
            _integer_residue_id(residues[insertion_index].id)
            if insertion_index < len(residues)
            else None
        )
        inferred = _infer_missing_residue_numbers(
            previous_id,
            next_id,
            count=len(names),
        )
        for offset, name in enumerate(names):
            result.append(
                {
                    "chain": chain.id,
                    "resname": str(name),
                    "resid": inferred[offset],
                    "icode": None,
                    "chain_index": chain_index,
                    "insertion_index": insertion_index + offset,
                }
            )
    return result


def _infer_missing_residue_numbers(
    previous_id: int | None,
    next_id: int | None,
    *,
    count: int,
) -> list[int]:
    if previous_id is not None and next_id is not None:
        expected = list(range(previous_id + 1, next_id))
        if len(expected) != count:
            raise StructureRepairError(
                "PDBFixer missing-residue placement is ambiguous: residue IDs "
                f"{previous_id} and {next_id} delimit {len(expected)} position(s), "
                f"but the sequence contains {count} missing residue(s)."
            )
        return expected
    if previous_id is not None:
        return list(range(previous_id + 1, previous_id + count + 1))
    if next_id is not None:
        return list(range(next_id - count, next_id))
    raise StructureRepairError(
        "PDBFixer reported missing residues in an empty chain; residue numbering "
        "cannot be inferred safely."
    )


def _integer_residue_id(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise StructureRepairError(
            f"Structure repair requires integer PDB residue IDs; found {value!r}."
        ) from exc


def _validate_expected_missing_residues(
    detected: list[dict[str, object]],
    *,
    expected: list[ResidueSelector],
) -> None:
    detected_keys = {
        (
            str(item["chain"]),
            str(item["resname"]),
            int(item["resid"]),
            item["icode"],
        )
        for item in detected
    }
    expected_keys = {
        (item.chain, item.resname, item.resid, item.icode) for item in expected
    }
    if detected_keys != expected_keys:
        missing_from_manifest = sorted(detected_keys - expected_keys)
        requested_but_absent = sorted(expected_keys - detected_keys)
        raise StructureRepairError(
            "PDBFixer missing-residue detection does not exactly match "
            "structure.repair.missing_residues. "
            f"Unapproved detections: {missing_from_manifest or 'none'}; "
            f"requested but not detected: {requested_but_absent or 'none'}."
        )


def _detected_missing_atoms(fixer: object) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for residue, atoms in fixer.missingAtoms.items():  # type: ignore[attr-defined]
        if not atoms:
            continue
        values.append(
            {
                "chain": residue.chain.id,
                "resname": residue.name,
                "resid": int(residue.id),
                "icode": residue.insertionCode or None,
                "atom_names": sorted(atom.name for atom in atoms),
                "source": "missing_heavy_atoms",
            }
        )
    for residue, atom_names in fixer.missingTerminals.items():  # type: ignore[attr-defined]
        if not atom_names:
            continue
        values.append(
            {
                "chain": residue.chain.id,
                "resname": residue.name,
                "resid": int(residue.id),
                "icode": residue.insertionCode or None,
                "atom_names": sorted(str(name) for name in atom_names),
                "source": "missing_terminal_atoms",
            }
        )
    return sorted(
        values,
        key=lambda item: (
            str(item["chain"]),
            int(item["resid"]),
            str(item["icode"] or ""),
        ),
    )


def _restore_existing_atom_metadata(
    original: PdbStructure,
    repaired: PdbStructure,
    *,
    output_path: Path,
) -> PdbStructure:
    original_by_identity = {atom.atom_identity: atom for atom in original.atoms}
    repaired_by_identity = {atom.atom_identity: atom for atom in repaired.atoms}
    lost = sorted(set(original_by_identity) - set(repaired_by_identity))
    if lost:
        raise StructureRepairError(
            "PDBFixer removed or renamed existing atoms, which mdprep forbids: "
            + ", ".join(str(item) for item in lost[:20])
        )

    restored_atoms: list[AtomRecord] = []
    new_serial_by_identity: dict[tuple[str, str, int, str | None, str], int] = {}
    for repaired_atom in repaired.atoms:
        if repaired_atom.serial is None:
            raise StructureRepairError(
                f"PDBFixer output atom {repaired_atom.atom_identity} has no serial."
            )
        original_atom = original_by_identity.get(repaired_atom.atom_identity)
        if original_atom is not None:
            displacement = dist(
                (original_atom.x, original_atom.y, original_atom.z),
                (repaired_atom.x, repaired_atom.y, repaired_atom.z),
            )
            if displacement > 0.002:
                raise StructureRepairError(
                    "PDBFixer moved an existing atom by more than 0.002 A: "
                    f"{original_atom.atom_identity} moved {displacement:.6f} A."
                )
            restored_atom = replace(
                repaired_atom,
                occupancy=original_atom.occupancy,
                bfactor=original_atom.bfactor,
                element=original_atom.element,
                record_name=original_atom.record_name,
                original_line=original_atom.original_line,
            )
        else:
            restored_atom = repaired_atom
        restored_atoms.append(restored_atom)
        new_serial_by_identity[restored_atom.atom_identity] = repaired_atom.serial

    old_identity_by_serial = {
        atom.serial: atom.atom_identity
        for atom in original.atoms
        if atom.serial is not None
    }
    bonds = set(repaired.conect_bonds)
    for left, right in original.conect_bonds:
        left_identity = old_identity_by_serial.get(left)
        right_identity = old_identity_by_serial.get(right)
        if left_identity is None or right_identity is None:
            continue
        new_left = new_serial_by_identity.get(left_identity)
        new_right = new_serial_by_identity.get(right_identity)
        if new_left is not None and new_right is not None:
            bonds.add((min(new_left, new_right), max(new_left, new_right)))

    repaired_by_serial = {
        atom.serial: atom
        for atom in restored_atoms
        if atom.serial is not None
    }
    serials_by_residue: dict[tuple[str, str, int, str | None], list[int]] = {}
    for atom in restored_atoms:
        if atom.serial is not None:
            serials_by_residue.setdefault(atom.residue_key, []).append(atom.serial)
    ter_after = set(repaired.ter_after_serials)
    for old_serial in original.ter_after_serials:
        identity = old_identity_by_serial.get(old_serial)
        if identity is None:
            continue
        original_atom = original_by_identity.get(identity)
        if original_atom is None:
            continue
        repaired_residue_serials = serials_by_residue.get(original_atom.residue_key, [])
        if not repaired_residue_serials:
            continue
        # A terminal atom such as OXT may have been appended after the atom
        # that preceded the input TER record. Move the boundary to the final
        # atom of that repaired residue instead of splitting the residue.
        ter_after = {
            serial
            for serial in ter_after
            if repaired_by_serial.get(serial) is None
            or repaired_by_serial[serial].residue_key != original_atom.residue_key
        }
        ter_after.add(repaired_residue_serials[-1])

    return PdbStructure(
        path=output_path,
        atoms=restored_atoms,
        residues=_build_residues_like(repaired.residues, restored_atoms),
        model_count=1,
        used_model=1,
        warnings=list(original.warnings),
        conect_bonds=bonds,
        ter_after_serials=ter_after,
    )


def _build_residues_like(
    repaired_residues: list[ResidueRecord],
    atoms: list[AtomRecord],
) -> list[ResidueRecord]:
    atoms_by_key: dict[tuple[str, str, int, str | None], list[AtomRecord]] = {}
    for atom in atoms:
        atoms_by_key.setdefault(atom.residue_key, []).append(atom)
    result: list[ResidueRecord] = []
    for index, residue in enumerate(repaired_residues):
        residue_atoms = atoms_by_key.get(
            (
                residue.id.chain_id,
                residue.id.resname,
                residue.id.resid,
                residue.id.icode,
            ),
            [],
        )
        if not residue_atoms:
            continue
        result.append(
            ResidueRecord(
                id=residue.id,
                atoms=residue_atoms,
                record_names={atom.record_name for atom in residue_atoms},
                original_index=index,
            )
        )
    return result


def _validate_repaired_peptide_bonds(
    structure: PdbStructure,
    *,
    expected: list[ResidueSelector],
    minimum: float,
    maximum: float,
    require_trans: bool,
    minimum_abs_omega: float,
) -> list[dict[str, object]]:
    polymer = [
        residue for residue in structure.residues if is_standard_protein_residue(residue)
    ]
    checks: list[dict[str, object]] = []
    for selector in expected:
        matches = [
            residue
            for residue in polymer
            if residue.id.chain_id == selector.chain
            and residue.id.resname == selector.resname
            and residue.id.resid == selector.resid
            and residue.id.icode == selector.icode
        ]
        if len(matches) != 1:
            raise StructureRepairError(
                f"Repaired residue {selector.chain}:{selector.resname}{selector.resid} "
                f"resolved {len(matches)} times."
            )
        residue = matches[0]
        index = polymer.index(residue)
        neighbors: list[tuple[ResidueRecord, ResidueRecord]] = []
        if index > 0 and polymer[index - 1].id.chain_id == residue.id.chain_id:
            neighbors.append((polymer[index - 1], residue))
        if (
            index + 1 < len(polymer)
            and polymer[index + 1].id.chain_id == residue.id.chain_id
        ):
            neighbors.append((residue, polymer[index + 1]))
        for left, right in neighbors:
            c_atom = _required_atom(left, "C")
            n_atom = _required_atom(right, "N")
            length = dist(
                (c_atom.x, c_atom.y, c_atom.z),
                (n_atom.x, n_atom.y, n_atom.z),
            )
            if not minimum <= length <= maximum:
                raise StructureRepairError(
                    f"Repaired peptide bond {left.id.display()} C -- "
                    f"{right.id.display()} N is {length:.4f} A; required "
                    f"{minimum:.3f}-{maximum:.3f} A."
                )
            omega = _dihedral_degrees(
                _required_atom(left, "CA"),
                c_atom,
                n_atom,
                _required_atom(right, "CA"),
            )
            if require_trans and abs(omega) < minimum_abs_omega:
                raise StructureRepairError(
                    f"Repaired peptide bond {left.id.display()}--"
                    f"{right.id.display()} has omega {omega:.2f} degrees; "
                    "an unapproved cis/non-trans peptide was generated."
                )
            checks.append(
                {
                    "left_residue": left.id.to_dict(),
                    "right_residue": right.id.to_dict(),
                    "c_n_distance_angstrom": length,
                    "omega_degrees": omega,
                    "trans_required": require_trans,
                    "ok": True,
                }
            )
    return checks


def _required_atom(residue: ResidueRecord, name: str) -> AtomRecord:
    matches = [atom for atom in residue.atoms if atom.name == name]
    if len(matches) != 1:
        raise StructureRepairError(
            f"Residue {residue.id.display()} requires exactly one {name} atom "
            f"for peptide validation; found {len(matches)}."
        )
    return matches[0]


def _dihedral_degrees(
    atom0: AtomRecord,
    atom1: AtomRecord,
    atom2: AtomRecord,
    atom3: AtomRecord,
) -> float:
    points = [
        np.asarray((atom.x, atom.y, atom.z), dtype=float)
        for atom in (atom0, atom1, atom2, atom3)
    ]
    b0 = -(points[1] - points[0])
    b1 = points[2] - points[1]
    b2 = points[3] - points[2]
    norm = np.linalg.norm(b1)
    if norm == 0:
        raise StructureRepairError("Cannot evaluate peptide omega for zero-length bond.")
    b1 /= norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
