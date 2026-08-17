"""Ligand residue extraction from prepared structures."""

from __future__ import annotations

import re
from collections import Counter
import json
from dataclasses import dataclass
from pathlib import Path
from dataclasses import replace

from mdprep.config.models import LigandConfig, ManifestConfig
from mdprep.ligands.identity import ligand_identity_dict
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.pdb import infer_element
from mdprep.structure.selectors import SelectorError, resolve_residue_selector
from mdprep.structure.writer import write_pdb


class LigandExtractionError(ValueError):
    """Raised when configured ligands cannot be extracted safely."""


@dataclass(frozen=True)
class ExtractedLigand:
    config: LigandConfig
    residue: ResidueRecord
    pdb_path: Path
    identity_path: Path
    warnings: list[str]

    @property
    def atoms(self):
        return self.residue.atoms

    def identity_dict(self) -> dict[str, object]:
        return ligand_identity_dict(
            ligand_id=self.config.id,
            selector=self.config.selector.model_dump(mode="json"),
            residue=self.residue,
            net_charge=self.config.net_charge,
            multiplicity=self.config.multiplicity,
            expected_formula=self.config.expected_formula,
            charge_method=self.config.charge_method,
            atom_types=self.config.atom_types,
        )


def extract_configured_ligands(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    output_dir: str | Path,
) -> list[ExtractedLigand]:
    extracted: list[ExtractedLigand] = []
    for ligand in manifest.ligands:
        extracted.append(extract_ligand(structure, ligand, output_dir=output_dir))
    return extracted


def extract_ligand(
    structure: PdbStructure,
    ligand: LigandConfig,
    *,
    output_dir: str | Path,
) -> ExtractedLigand:
    try:
        residue = resolve_residue_selector(structure, ligand.selector.model_dump())
    except SelectorError as exc:
        raise LigandExtractionError(
            f"Ligand {ligand.id} selector did not resolve exactly one residue: {exc}"
        ) from exc
    if not residue.atoms:
        raise LigandExtractionError(f"Ligand {ligand.id} selector resolved a residue with zero atoms.")

    warnings: list[str] = []
    fixed_atoms = []
    for atom in residue.atoms:
        if atom.element is None:
            atom_field = atom.original_line[12:16] if len(atom.original_line) >= 16 else None
            inferred = infer_element(
                atom.name,
                resname=atom.resname,
                record_name=atom.record_name,
                atom_field=atom_field,
            )
            warnings.append(f"Inferred missing element for ligand {ligand.id} atom {atom.name}: {inferred}")
            fixed_atoms.append(atom.__class__(**{**atom.__dict__, "element": inferred}))
        else:
            fixed_atoms.append(atom)
    fixed_atoms, rename_map = _uniquify_duplicate_atom_names(fixed_atoms)
    if rename_map:
        renamed = ", ".join(f"{old}->{new}" for old, new in rename_map)
        warnings.append(
            "Ligand atom names were not unique in the input PDB; generated deterministic "
            f"Amber-safe names for this ligand: {renamed}."
        )
    if fixed_atoms != residue.atoms:
        residue = ResidueRecord(
            id=residue.id,
            atoms=fixed_atoms,
            record_names=residue.record_names,
            original_index=residue.original_index,
        )
    warnings.extend(_validate_ligand_formula(ligand, residue))

    ligand_dir = Path(output_dir) / "ligands" / ligand.id / "input"
    ligand_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = ligand_dir / f"{ligand.id}.pdb"
    identity_path = ligand_dir / "identity.json"
    ligand_structure = PdbStructure(
        path=pdb_path,
        atoms=list(residue.atoms),
        residues=[residue],
        model_count=1,
    )
    write_pdb(ligand_structure, pdb_path)
    conect_lines = _ligand_conect_lines(structure.path, residue.atoms)
    if conect_lines:
        _insert_conect_before_end(pdb_path, conect_lines)
        warnings.append(
            f"Preserved {len(conect_lines)} ligand CONECT record(s) for {ligand.id} in the extracted PDB."
        )
    elif _requires_pdb_chemistry_perception(ligand):
        warnings.append(
            "No ligand CONECT records were found in the input PDB. AmberTools will infer ligand "
            "connectivity from coordinates; for chemically sensitive or complex substrates, provide "
            "a curated user_mol2/user_frcmod so atom types and bonded terms are not inferred from PDB geometry."
        )
    extracted = ExtractedLigand(
        config=ligand,
        residue=residue,
        pdb_path=pdb_path,
        identity_path=identity_path,
        warnings=warnings,
    )
    identity_path.write_text(
        json.dumps(extracted.identity_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return extracted


def _uniquify_duplicate_atom_names(atoms: list) -> tuple[list, list[tuple[str, str]]]:
    counts: dict[str, int] = {}
    for atom in atoms:
        counts[atom.name] = counts.get(atom.name, 0) + 1
    duplicate_names = {name for name, count in counts.items() if count > 1}
    if not duplicate_names:
        return atoms, []

    used = {atom.name for atom in atoms if atom.name not in duplicate_names}
    element_counters: dict[str, int] = {}
    renamed_atoms = []
    rename_map: list[tuple[str, str]] = []
    for atom in atoms:
        if atom.name not in duplicate_names:
            renamed_atoms.append(atom)
            continue
        element = (atom.element or infer_element(atom.name, resname=atom.resname, record_name=atom.record_name) or "X").upper()
        new_name = _next_unique_ligand_atom_name(element, used, element_counters)
        used.add(new_name)
        renamed_atoms.append(replace(atom, name=new_name))
        rename_map.append((atom.name, new_name))
    return renamed_atoms, rename_map


def _next_unique_ligand_atom_name(
    element: str,
    used: set[str],
    counters: dict[str, int],
) -> str:
    symbol = "".join(char for char in element.upper() if char.isalnum()) or "X"
    if len(symbol) > 2:
        symbol = symbol[:2]
    while True:
        counters[symbol] = counters.get(symbol, 0) + 1
        suffix = _base36(counters[symbol])
        candidate = f"{symbol}{suffix}"
        if len(candidate) > 4:
            candidate = f"{symbol[:1]}{suffix}"[-4:]
        if candidate not in used:
            return candidate


def _base36(value: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value <= 0:
        return "0"
    digits = []
    current = value
    while current:
        current, remainder = divmod(current, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def _requires_pdb_chemistry_perception(ligand: LigandConfig) -> bool:
    if ligand.charge_method in {"am1bcc", "mcpb_resp_pyscf"}:
        return True
    if ligand.charge_method in {"gas_resp_pyscf", "qmmesp_pyscf"}:
        return ligand.user_mol2 is None
    return False


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]*)")


def _validate_ligand_formula(
    ligand: LigandConfig,
    residue: ResidueRecord,
) -> list[str]:
    """Require explicit ligand hydrogens and validate an optional exact formula.

    PDB coordinates do not encode bond orders or protonation. A molecule that
    contains no hydrogen is therefore rejected unless the user explicitly
    declares a hydrogen-free expected formula. An exact formula is the only
    reliable way to detect a partially hydrogenated ligand, so production
    examples should always set expected_formula.
    """

    observed = Counter((atom.element or "").strip().capitalize() for atom in residue.atoms)
    observed.pop("", None)
    expected = _parse_formula(ligand.expected_formula) if ligand.expected_formula else None

    if observed.get("H", 0) == 0 and (expected is None or expected.get("H", 0) > 0):
        warning = (
            f"WARNING [LIGAND_HYDROGENS_MISSING] Ligand {ligand.id} "
            f"({residue.id.chain_id}:{residue.id.resname}:{residue.id.resid}) contains no "
            "explicit hydrogen atoms."
        )
        error = (
            f"ERROR [LIGAND_CHEMISTRY_UNRESOLVED] Ligand {ligand.id} cannot be sent to "
            "AM1-BCC or QM charge derivation without an explicit, validated protonation "
            "state. Add ligand hydrogens before running mdprep and set expected_formula; "
            "mdprep will not guess ligand protonation."
        )
        raise LigandExtractionError(f"{warning}\n{error}")

    warnings: list[str] = []
    if expected is None:
        warnings.append(
            f"WARNING [LIGAND_FORMULA_UNVERIFIED] Ligand {ligand.id} contains explicit "
            "hydrogens, but expected_formula was not supplied; mdprep cannot prove that "
            "the ligand is completely protonated."
        )
        return warnings

    if observed == expected:
        return warnings

    observed_heavy = Counter({key: value for key, value in observed.items() if key != "H"})
    expected_heavy = Counter({key: value for key, value in expected.items() if key != "H"})
    if observed_heavy == expected_heavy and observed.get("H", 0) < expected.get("H", 0):
        missing = expected.get("H", 0) - observed.get("H", 0)
        warning = (
            f"WARNING [LIGAND_HYDROGENS_MISSING] Ligand {ligand.id} has "
            f"{observed.get('H', 0)} explicit H atom(s), but expected_formula "
            f"{ligand.expected_formula} requires {expected.get('H', 0)}; {missing} hydrogen "
            "atom(s) are missing."
        )
        error = (
            f"ERROR [LIGAND_CHEMISTRY_UNRESOLVED] Ligand {ligand.id} is only partially "
            "hydrogenated. Add the missing ligand hydrogens before running mdprep; "
            "automatic ligand protonation is intentionally unsupported."
        )
        raise LigandExtractionError(f"{warning}\n{error}")

    raise LigandExtractionError(
        f"ERROR [LIGAND_FORMULA_MISMATCH] Ligand {ligand.id} has observed formula "
        f"{_format_formula(observed)}, but expected_formula is {ligand.expected_formula}. "
        "Correct the input coordinates or the manifest; mdprep will not alter ligand "
        "composition silently."
    )


def _parse_formula(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    position = 0
    for match in _FORMULA_TOKEN.finditer(formula):
        if match.start() != position:
            raise LigandExtractionError(f"Invalid expected_formula: {formula}")
        element, count_text = match.groups()
        if element in counts:
            raise LigandExtractionError(
                f"Invalid expected_formula {formula}: element {element} appears more than once"
            )
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise LigandExtractionError(
                f"Invalid expected_formula {formula}: element counts must be positive"
            )
        counts[element] = count
        position = match.end()
    if position != len(formula) or not counts:
        raise LigandExtractionError(f"Invalid expected_formula: {formula}")
    return counts


def _format_formula(counts: Counter[str]) -> str:
    ordered = []
    for element in ("C", "H"):
        if element in counts:
            ordered.append(element + (str(counts[element]) if counts[element] != 1 else ""))
    for element in sorted(key for key in counts if key not in {"C", "H"}):
        ordered.append(element + (str(counts[element]) if counts[element] != 1 else ""))
    return "".join(ordered)


def _ligand_conect_lines(input_path: Path, atoms: list) -> list[str]:
    serials = {atom.serial for atom in atoms if atom.serial is not None}
    if not serials or not input_path.exists():
        return []

    lines: list[str] = []
    seen: set[str] = set()
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("CONECT"):
            continue
        numbers = _parse_conect_numbers(line)
        if not numbers:
            continue
        source, *targets = numbers
        if source not in serials:
            continue
        selected_targets = [target for target in targets if target in serials]
        for offset in range(0, len(selected_targets), 4):
            conect = _format_conect(source, selected_targets[offset : offset + 4])
            if conect not in seen:
                seen.add(conect)
                lines.append(conect)
    return lines


def _parse_conect_numbers(line: str) -> list[int]:
    numbers: list[int] = []
    for field in line[6:].split():
        try:
            numbers.append(int(field))
        except ValueError:
            continue
    return numbers


def _format_conect(source: int, targets: list[int]) -> str:
    return "CONECT" + f"{source:5d}" + "".join(f"{target:5d}" for target in targets) + "\n"


def _insert_conect_before_end(path: Path, conect_lines: list[str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    insert_at = len(lines)
    if lines and lines[-1].strip() == "END":
        insert_at = len(lines) - 1
    lines[insert_at:insert_at] = conect_lines
    path.write_text("".join(lines), encoding="utf-8")
