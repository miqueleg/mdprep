"""Explicit structure repair with strict identity and geometry validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from math import dist
from pathlib import Path
from typing import Callable

import numpy as np

from mdprep.config.models import ManifestConfig, ResidueSelector
from mdprep.structure.classify import is_standard_protein_residue
from mdprep.structure.heterogens import (
    KEPT_WHEN_SIDE_CHAIN_REBUILT,
    side_chain_closes_onto_backbone,
)
from mdprep.structure.hydrogens import (
    input_hydrogen_keeper,
    is_hydrogen,
    strip_input_hydrogens,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.pdb_fields import expected_split_metal_numbering
from mdprep.structure.reader import is_mmcif_path, read_structure
from mdprep.structure.writer import write_pdb


# PDBFixer round-trips coordinates through OpenMM in nanometres, and a PDB
# stores three decimal places in angstrom, so an atom it did not touch still
# comes back displaced by a few thousandths of an angstrom. The tolerance has
# to clear that floor while staying far below any real movement: a bond is
# 1.5 A, and the smallest displacement that means anything is two orders of
# magnitude larger than this.
EXISTING_ATOM_TOLERANCE_ANGSTROM = 0.01


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

    missing_residues_left_absent: tuple[dict[str, object], ...] = ()
    # Modified residues PDBFixer replaced by their parents, each with the
    # name it had and the one it got.
    modified_residues_replaced: tuple[dict[str, object], ...] = ()

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
            "missing_residues_left_absent": list(self.missing_residues_left_absent),
            "missing_heavy_atoms_detected": list(
                self.missing_heavy_atoms_detected
            ),
            "added_atom_count": self.added_atom_count,
            "modified_residues_replaced": list(self.modified_residues_replaced),
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

    # Repair runs on the chains the build will keep, with alternate locations
    # already resolved. Otherwise PDBFixer rules on residues about to be thrown
    # away, and picks conformers the manifest did not choose.
    selected_chains = [chain.strip() for chain in manifest.structure.keep_chains]
    repair_input = Path(output_path).with_name(
        Path(output_path).stem + ".repair_input.pdb"
    )
    repair_input.parent.mkdir(parents=True, exist_ok=True)
    _write_repair_input(
        input_path,
        repair_input,
        chains=selected_chains,
        altloc_policy=manifest.structure.altloc_policy,
        keep_hydrogens=input_hydrogen_keeper(manifest),
    )

    try:
        platform = Platform.getPlatformByName(config.platform)
        with expected_split_metal_numbering():
            fixer = PDBFixer(filename=str(repair_input), platform=platform)
        if sequence_path is not None:
            sequence_fixer = PDBFixer(filename=str(sequence_path), platform=platform)
            fixer.sequences = sequence_fixer.sequences
        fixer.findMissingResidues()
        detected_residues = _detected_missing_residues(fixer)
        left_absent = _validate_expected_missing_residues(
            detected_residues,
            expected=config.missing_residues,
            policy=config.unresolved_residues,
        )
        if left_absent:
            # PDBFixer builds whatever remains in missingResidues, so the ones
            # being declined have to be removed before addMissingAtoms runs.
            approved = {
                (item.chain, item.resname, item.resid, item.icode)
                for item in config.missing_residues
            }
            # Keep only the entries every one of whose residues was approved.
            approved_keys = {
                item["_key"]
                for item in detected_residues
                if (
                    str(item["chain"]),
                    str(item["resname"]),
                    int(item["resid"]),
                    item["icode"],
                )
                in approved
            }
            fixer.missingResidues = {
                key: value
                for key, value in fixer.missingResidues.items()
                if key in approved_keys
            }
        replaced: list[dict[str, object]] = []
        if config.replace_modified_residues:
            # After the missing residues, which PDBFixer finds by matching
            # residue names against SEQRES -- where a selenomethionine is
            # still MSE. Before the missing atoms, so that what the parent has
            # and the modified form lacked, a methionine's SD for a
            # selenomethionine's SE, is built like any other missing atom.
            fixer.nonstandardResidues = _listed_modified_residues(
                fixer, config.replace_modified_residues
            )
            replaced = [
                {
                    "chain": residue.chain.id,
                    "resname": residue.name,
                    "resid": int(residue.id),
                    "icode": _insertion_code(residue),
                    "replacement": replacement,
                }
                for residue, replacement in fixer.nonstandardResidues
            ]
            fixer.replaceNonstandardResidues()
            cleared = _clear_side_chains_bonded_to_the_backbone(fixer, replaced)
            for item in replaced:
                item["side_chain_rebuilt"] = cleared.get(
                    (str(item["chain"]), int(item["resid"]), item["icode"]), []
                )
        replaced_keys = {
            (str(item["chain"]), int(item["resid"]), item["icode"]) for item in replaced
        }
        fixer.findMissingAtoms()
        detected_atoms = _detected_missing_atoms(fixer)
        if not config.add_missing_heavy_atoms:
            # A chain's terminal OXT is tleap's to build, as it is when nothing
            # is repaired: replacing a modified residue is no reason for
            # PDBFixer to build atoms nobody asked it for.
            fixer.missingTerminals = {}
        # A replaced residue is rebuilt to its parent's atoms by design.
        unasked = [
            item
            for item in detected_atoms
            if (str(item["chain"]), int(item["resid"]), item["icode"]) not in replaced_keys
            and item["source"] != "missing_terminal_atoms"
        ]
        if unasked and not config.add_missing_heavy_atoms:
            formatted = ", ".join(
                f"{item['chain']}:{item['resname']}{item['resid']} "
                f"({','.join(item['atom_names'])})"
                for item in unasked
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
        repair_input,
        altloc_policy=manifest.structure.altloc_policy,
    )
    repaired_raw = read_pdb(raw_output)
    repaired = _restore_existing_atom_metadata(
        original,
        repaired_raw,
        output_path=output,
        replaced={
            (str(item["chain"]), str(item["resname"]), int(item["resid"]), item["icode"])
            for item in replaced
        },
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
        *(
            f"{item['chain']}:{item['resname']}{item['resid']}{item['icode'] or ''} "
            f"was replaced by {item['replacement']} with its side chain rebuilt "
            f"beyond CB ({', '.join(item['side_chain_rebuilt'])}): it was bonded "
            f"to the residue's own backbone, which {item['replacement']}'s is not, "
            "so the atoms the two share by name were not where the parent has "
            "them. The new side chain is PDBFixer's template rotamer, "
            + _closest_rebuilt_contact(repaired, item)
            + "; minimize before anything else."
            for item in replaced
            if item["side_chain_rebuilt"]
        ),
    )
    return StructureRepairResult(
        backend="pdbfixer",
        backend_version=_package_version("pdbfixer"),
        input_path=input_path,
        sequence_source_path=sequence_path,
        raw_output_path=raw_output,
        output_path=output,
        structure=repaired,
        missing_residues_detected=tuple(
            {key: value for key, value in item.items() if key != "_key"}
            for item in detected_residues
        ),
        missing_residues_left_absent=tuple(left_absent),
        modified_residues_replaced=tuple(replaced),
        missing_heavy_atoms_detected=tuple(detected_atoms),
        added_atom_count=len(repaired.atoms) - len(original.atoms),
        peptide_bond_checks=tuple(checks),
        random_seed=config.random_seed,
        platform=config.platform,
        warnings=warnings,
    )


def _listed_modified_residues(fixer: object, listed: list[object]) -> list[tuple[object, str]]:
    """PDBFixer's residues for each listed modified residue, with its parent.

    Exactly the listed ones. PDBFixer's own findNonstandardResidues would
    replace any residue its table names, chain or not, and nobody asked for
    those.
    """

    by_key = {
        (
            residue.chain.id,
            residue.name,
            int(residue.id),
            _insertion_code(residue),
        ): residue
        for residue in fixer.topology.residues()  # type: ignore[attr-defined]
    }
    found: list[tuple[object, str]] = []
    for item in listed:
        selector = item.selector  # type: ignore[attr-defined]
        key = (selector.chain, selector.resname, selector.resid, selector.icode or None)
        residue = by_key.get(key)
        if residue is None:
            raise StructureRepairError(
                "structure.repair.replace_modified_residues names "
                f"{selector.chain}:{selector.resname}{selector.resid}{selector.icode or ''}, "
                "which is not in the chains being built."
            )
        found.append((residue, item.parent))  # type: ignore[attr-defined]
    return found


def _clear_side_chains_bonded_to_the_backbone(
    fixer: object, replaced: list[dict[str, object]]
) -> dict[tuple[str, int, str | None], list[str]]:
    """Delete, for PDBFixer to rebuild, a replaced side chain its parent cannot have.

    Everything beyond CB goes, and the parent's template supplies the side
    chain when the missing atoms are built. Returns the deleted atoms' names
    by residue.
    """

    from openmm import unit
    from openmm.app import Modeller

    parents = {
        (str(item["chain"]), int(item["resid"]), item["icode"]): str(item["replacement"])
        for item in replaced
    }
    positions = fixer.positions.value_in_unit(unit.angstrom)  # type: ignore[attr-defined]
    cleared: dict[tuple[str, int, str | None], list[str]] = {}
    doomed: list[object] = []
    for residue in fixer.topology.residues():  # type: ignore[attr-defined]
        key = (residue.chain.id, int(residue.id), _insertion_code(residue))
        if key not in parents:
            continue
        atoms = [
            atom
            for atom in residue.atoms()
            if atom.element is not None and atom.element.symbol != "H"
        ]
        if side_chain_closes_onto_backbone(
            ((atom.name, positions[atom.index]) for atom in atoms), parents[key]
        ):
            side_chain = [
                atom for atom in atoms if atom.name not in KEPT_WHEN_SIDE_CHAIN_REBUILT
            ]
            cleared[key] = sorted(atom.name for atom in side_chain)
            doomed.extend(side_chain)
    if doomed:
        modeller = Modeller(fixer.topology, fixer.positions)  # type: ignore[attr-defined]
        modeller.delete(doomed)
        fixer.topology = modeller.topology  # type: ignore[attr-defined]
        fixer.positions = modeller.positions  # type: ignore[attr-defined]
    return cleared


def _closest_rebuilt_contact(structure: PdbStructure, item: dict[str, object]) -> str:
    """How close a rebuilt side chain comes to the atoms around it, in words."""

    chain, resid, icode = str(item["chain"]), int(item["resid"]), item["icode"]

    def heavy(atom: AtomRecord) -> bool:
        return not is_hydrogen(atom)

    def own(atom: AtomRecord) -> bool:
        return (atom.chain_id, atom.resid, atom.icode or None) == (chain, resid, icode)

    rebuilt = [
        atom
        for atom in structure.atoms
        if own(atom) and heavy(atom) and atom.name not in KEPT_WHEN_SIDE_CHAIN_REBUILT
    ]
    others = [atom for atom in structure.atoms if not own(atom) and heavy(atom)]
    if not rebuilt or not others:
        return "with nothing around it"
    near = np.array([(atom.x, atom.y, atom.z) for atom in others])
    closest: tuple[float, AtomRecord, AtomRecord] | None = None
    for atom in rebuilt:
        distances = np.linalg.norm(near - (atom.x, atom.y, atom.z), axis=1)
        index = int(distances.argmin())
        if closest is None or distances[index] < closest[0]:
            closest = (float(distances[index]), atom, others[index])
    distance, mine, other = closest  # type: ignore[misc]
    return (
        f"whose {mine.name} is {distance:.2f} A from "
        f"{other.chain_id}:{other.resname}{other.resid}{other.icode or ''} {other.name}"
    )


def _write_repair_input(
    source: Path,
    destination: Path,
    *,
    chains: list[str],
    altloc_policy: str,
    keep_hydrogens: Callable[[ResidueRecord], bool] | None = None,
) -> None:
    """Prepare the file PDBFixer reads: chosen chains, chosen conformers.

    PDBFixer applies its own alternate-location rule, which is not mdprep's.
    On 6NIE that silently repaired a different conformer of Arg177 than the one
    the manifest's policy selected -- the two differ by several angstrom at the
    guanidinium. Resolving altlocs here leaves PDBFixer nothing to disagree
    about.

    Filtering is done on the original lines rather than by rewriting the
    structure, so SEQRES and the rest of the header survive; PDBFixer needs
    them to know which residues were never resolved.

    An mmCIF has no such lines to keep, so it is converted instead. PDBFixer
    then sees only the residues that are present, which is what the default
    `unresolved_residues: leave_absent` asks for anyway; it can still complete
    a side chain, because that comes from its residue templates rather than
    from the header.

    With ``keep_hydrogens``, the input's hydrogens go too, except in the
    residues it names: PDBFixer is then repairing the structure the rest of
    mdprep will build, not the one some other program protonated.
    """

    structure = read_structure(source, altloc_policy=altloc_policy)  # type: ignore[arg-type]
    if keep_hydrogens is not None:
        structure, _ = strip_input_hydrogens(structure, keep=keep_hydrogens)
    if is_mmcif_path(source):
        selected_chains = set(chains)
        kept_atoms = [
            atom
            for atom in structure.atoms
            if not selected_chains or atom.chain_id in selected_chains
        ]
        write_pdb(
            PdbStructure(
                path=destination,
                atoms=kept_atoms,
                residues=build_residues(kept_atoms),
                model_count=structure.model_count,
                used_model=structure.used_model,
                warnings=list(structure.warnings),
                conect_bonds=structure.conect_bonds,
                ter_after_serials=structure.ter_after_serials,
            ),
            destination,
        )
        return

    # Each atom the structure kept, by the line it was read from. Serials
    # would do if they were unique, but reduce writes every hydrogen it adds
    # with serial 0, the reader renumbers a file like that, and its lines no
    # longer carry the serials the structure does. An atom the reader
    # rewrote -- a residue name too wide for its column -- has no line of
    # its own any more, and is found by its serial instead.
    kept_lines = Counter(
        atom.original_line for atom in structure.atoms if atom.original_line
    )
    rewritten_serials = {
        atom.serial
        for atom in structure.atoms
        if not atom.original_line and atom.serial is not None
    }
    selected_chains = set(chains)
    kept: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")):
            if selected_chains and len(line) > 21 and line[21] not in selected_chains:
                continue
            if kept_lines[line] > 0:
                kept_lines[line] -= 1
            elif _parse_serial_for_filter(line) not in rewritten_serials:
                continue
            # The chosen conformer loses its altloc marker, so nothing
            # downstream has a choice left to make.
            line = line[:16] + " " + line[17:]
        elif line.startswith(("ANISOU", "TER")):
            if selected_chains and len(line) > 21 and line[21] not in selected_chains:
                continue
        kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _parse_serial_for_filter(line: str) -> int | None:
    from mdprep.structure.pdb import _parse_serial

    return _parse_serial(line)


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
                    "_key": key,
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
    policy: str = "fail",
) -> list[dict[str, object]]:
    """Reconcile what PDBFixer found with what the manifest approved.

    Returns the residues that will be left absent, which the caller reports.
    """
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
    requested_but_absent = sorted(expected_keys - detected_keys)
    if requested_but_absent:
        # Asking to build something PDBFixer cannot see is always an error,
        # whatever the policy: the manifest and the structure disagree.
        raise StructureRepairError(
            "structure.repair.missing_residues names residues PDBFixer did not "
            f"detect as missing: {requested_but_absent}."
        )
    unapproved = sorted(detected_keys - expected_keys)
    if not unapproved:
        return []
    if policy == "fail":
        raise StructureRepairError(
            "PDBFixer missing-residue detection does not exactly match "
            "structure.repair.missing_residues. "
            f"Unapproved detections: {unapproved}. Approve each one to build it, "
            "or set structure.repair.unresolved_residues: leave_absent to "
            "deliberately leave them out."
        )
    return [
        {"chain": chain, "resname": resname, "resid": resid, "icode": icode}
        for chain, resname, resid, icode in unapproved
    ]


def _insertion_code(residue: object) -> str | None:
    """A PDBFixer residue's insertion code, None when blank as everywhere else.

    PDBFixer keeps the column as it read it, a space when there is none, and a
    key built with the space matches nothing mdprep built.
    """

    return (residue.insertionCode or "").strip() or None  # type: ignore[attr-defined]


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
                "icode": _insertion_code(residue),
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
                "icode": _insertion_code(residue),
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
    replaced: set[tuple[str, str, int, str | None]] = frozenset(),
) -> PdbStructure:
    # A residue replaced by its parent changes its name and some of its
    # atoms on purpose; every other atom must come back as it went in.
    original_by_identity = {
        atom.atom_identity: atom for atom in original.atoms if atom.residue_key not in replaced
    }
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
            if displacement > EXISTING_ATOM_TOLERANCE_ANGSTROM:
                raise StructureRepairError(
                    "PDBFixer moved an existing atom by more than "
                    f"{EXISTING_ATOM_TOLERANCE_ANGSTROM} A: "
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
