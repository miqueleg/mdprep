"""Lightweight fixed-width PDB parser."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, Literal

from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord


AltlocPolicy = Literal["highest_occupancy", "first", "fail"]
VALID_ALTLOC_POLICIES = {"highest_occupancy", "first", "fail"}


class PdbParseError(ValueError):
    """Raised when a PDB file cannot be parsed under the requested policy."""


def read_pdb(
    path: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
) -> PdbStructure:
    pdb_path = Path(path)
    if pdb_path.suffix.lower() in {".cif", ".mmcif"}:
        raise PdbParseError("mmCIF input is not supported in mdprep Task 2; provide a PDB file.")
    if altloc_policy not in VALID_ALTLOC_POLICIES:
        raise PdbParseError(
            f"Invalid altloc policy {altloc_policy!r}; expected one of {sorted(VALID_ALTLOC_POLICIES)}"
        )
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input structure not found: {pdb_path}")

    lines = pdb_path.read_text(encoding="utf-8").splitlines()
    model_count = sum(1 for line in lines if line.startswith("MODEL"))
    warnings: list[str] = []
    if model_count == 0:
        model_count = 1
        model_lines = lines
    else:
        model_lines = _first_model_lines(lines)
        if model_count > 1:
            warnings.append(f"Input contains {model_count} MODEL records; using MODEL 1 only.")

    raw_atoms = [_parse_atom_line(line) for line in model_lines if line.startswith(("ATOM", "HETATM"))]
    atoms = _apply_altloc_policy(raw_atoms, altloc_policy)
    residues = _build_residues(atoms)
    return PdbStructure(
        path=pdb_path,
        atoms=atoms,
        residues=residues,
        model_count=model_count,
        used_model=1,
        warnings=warnings,
    )


def _first_model_lines(lines: Iterable[str]) -> list[str]:
    in_first_model = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("MODEL"):
            if not in_first_model and not collected:
                in_first_model = True
                continue
            if in_first_model:
                continue
        if line.startswith("ENDMDL") and in_first_model:
            break
        if in_first_model:
            collected.append(line)
    return collected


def _parse_atom_line(line: str) -> AtomRecord:
    record_name = line[0:6].strip()
    if record_name not in {"ATOM", "HETATM"}:
        raise PdbParseError(f"Unsupported atom record {record_name!r}")

    serial = _parse_int(line[6:11])
    name = line[12:16].strip()
    altloc = _blank_to_none(line[16:17])
    resname = line[17:20].strip()
    chain_id = line[21:22]
    if chain_id == " ":
        chain_id = ""
    resid_text = line[22:26].strip()
    if not resid_text:
        raise PdbParseError(f"Missing residue number in line: {line}")
    resid = int(resid_text)
    icode = _blank_to_none(line[26:27])
    x = _parse_float_required(line[30:38], "x", line)
    y = _parse_float_required(line[38:46], "y", line)
    z = _parse_float_required(line[46:54], "z", line)
    occupancy = _parse_float(line[54:60])
    bfactor = _parse_float(line[60:66])
    element = _blank_to_none(line[76:78] if len(line) >= 78 else "")
    if element is None:
        element = infer_element(
            name,
            resname=resname,
            record_name=record_name,
            atom_field=line[12:16],
        )

    return AtomRecord(
        serial=serial,
        name=name,
        altloc=altloc,
        resname=resname,
        chain_id=chain_id,
        resid=resid,
        icode=icode,
        x=x,
        y=y,
        z=z,
        occupancy=occupancy,
        bfactor=bfactor,
        element=element,
        record_name=record_name,  # type: ignore[arg-type]
        original_line=line,
    )


STANDARD_ELEMENT_BY_PROTEIN_ATOM = {
    "C": "C",
    "CA": "C",
    "CB": "C",
    "CG": "C",
    "CG1": "C",
    "CG2": "C",
    "CD": "C",
    "CD1": "C",
    "CD2": "C",
    "CE": "C",
    "CE1": "C",
    "CE2": "C",
    "CE3": "C",
    "CH2": "C",
    "CZ": "C",
    "CZ2": "C",
    "CZ3": "C",
    "N": "N",
    "ND1": "N",
    "ND2": "N",
    "NE": "N",
    "NE1": "N",
    "NE2": "N",
    "NH1": "N",
    "NH2": "N",
    "NZ": "N",
    "O": "O",
    "OD1": "O",
    "OD2": "O",
    "OE1": "O",
    "OE2": "O",
    "OG": "O",
    "OG1": "O",
    "OH": "O",
    "OXT": "O",
    "S": "S",
    "SD": "S",
    "SG": "S",
}

STANDARD_PROTEIN_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "LYN",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

PERIODIC_ELEMENTS = {
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
}


def infer_element(
    atom_name: str,
    *,
    resname: str | None = None,
    record_name: str | None = None,
    atom_field: str | None = None,
) -> str | None:
    stripped = atom_name.strip()
    if not stripped:
        return None
    while stripped and stripped[0].isdigit():
        stripped = stripped[1:]
    if not stripped:
        return None
    upper = stripped.upper()
    if resname in STANDARD_PROTEIN_RESNAMES:
        if upper.startswith("H"):
            return "H"
        mapped = STANDARD_ELEMENT_BY_PROTEIN_ATOM.get(upper)
        if mapped is not None:
            return mapped
        return upper[0]
    if atom_field:
        aligned = atom_field.rstrip()
        if aligned.startswith(" ") and upper[0] in PERIODIC_ELEMENTS:
            return _canonical_element(upper[0])
        if not aligned.startswith(" ") and len(upper) >= 2 and upper[:2] in PERIODIC_ELEMENTS:
            return _canonical_element(upper[:2])
    if len(upper) >= 2 and upper[:2] in PERIODIC_ELEMENTS:
        return _canonical_element(upper[:2])
    if upper[0] in PERIODIC_ELEMENTS:
        return _canonical_element(upper[0])
    return upper[0]


def _canonical_element(symbol: str) -> str:
    upper = symbol.upper()
    return upper[0] + upper[1:].lower()


def _apply_altloc_policy(atoms: list[AtomRecord], policy: AltlocPolicy) -> list[AtomRecord]:
    grouped: dict[tuple[str, str, int, str | None, str], list[tuple[int, AtomRecord]]] = defaultdict(list)
    for index, atom in enumerate(atoms):
        grouped[atom.atom_identity].append((index, atom))

    selected_indices: set[int] = set()
    for identity, entries in grouped.items():
        alternates = [(index, atom) for index, atom in entries if atom.altloc is not None]
        if not alternates:
            selected_indices.update(index for index, _ in entries)
            continue
        if policy == "fail":
            display = _identity_display(identity)
            raise PdbParseError(f"Unresolved alternate locations found for atom {display}")
        selected_index, _ = _select_altloc(entries, policy)
        selected_indices.add(selected_index)

    return [atom for index, atom in enumerate(atoms) if index in selected_indices]


def _select_altloc(
    entries: list[tuple[int, AtomRecord]],
    policy: AltlocPolicy,
) -> tuple[int, AtomRecord]:
    blank_entries = [(index, atom) for index, atom in entries if atom.altloc is None]
    if policy == "first":
        return blank_entries[0] if blank_entries else entries[0]

    if blank_entries:
        blank_index, blank_atom = blank_entries[0]
        blank_occ = blank_atom.occupancy if blank_atom.occupancy is not None else 0.0
        best_alt_index, best_alt = max(
            [(index, atom) for index, atom in entries if atom.altloc is not None],
            key=lambda item: (item[1].occupancy if item[1].occupancy is not None else 0.0, -item[0]),
        )
        best_alt_occ = best_alt.occupancy if best_alt.occupancy is not None else 0.0
        if best_alt_occ > blank_occ:
            return best_alt_index, best_alt
        return blank_index, blank_atom

    return max(
        entries,
        key=lambda item: (item[1].occupancy if item[1].occupancy is not None else 0.0, -item[0]),
    )


def _build_residues(atoms: list[AtomRecord]) -> list[ResidueRecord]:
    grouped: "OrderedDict[tuple[str, str, int, str | None], list[AtomRecord]]" = OrderedDict()
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)

    residues: list[ResidueRecord] = []
    for index, ((chain_id, resname, resid, icode), residue_atoms) in enumerate(grouped.items()):
        residues.append(
            ResidueRecord(
                id=ResidueId(chain_id=chain_id, resname=resname, resid=resid, icode=icode),
                atoms=residue_atoms,
                record_names={atom.record_name for atom in residue_atoms},
                original_index=index,
            )
        )
    return residues


def _identity_display(identity: tuple[str, str, int, str | None, str]) -> str:
    chain_id, resname, resid, icode, atom_name = identity
    chain = chain_id if chain_id else "<blank>"
    return f"{chain}:{resname}{resid}{icode or ''}@{atom_name}"


def _parse_int(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if stripped else None


def _parse_float(text: str) -> float | None:
    stripped = text.strip()
    return float(stripped) if stripped else None


def _parse_float_required(text: str, field_name: str, line: str) -> float:
    stripped = text.strip()
    if not stripped:
        raise PdbParseError(f"Missing {field_name} coordinate in line: {line}")
    return float(stripped)


def _blank_to_none(text: str) -> str | None:
    stripped = text.strip()
    return stripped if stripped else None
