"""Lightweight fixed-width PDB parser."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import replace
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

    atom_line_indices = [
        index
        for index, line in enumerate(model_lines)
        if line.startswith(("ATOM", "HETATM"))
    ]
    raw_atoms = [_parse_atom_line(model_lines[index]) for index in atom_line_indices]

    # Writers that exhaust the five-column serial field for systems above
    # 99,999 atoms either overflow into column 12 (tleap/ambpdb), switch to
    # hybrid-36, or wrap around. The first two are decoded by _parse_serial;
    # a genuine wrap leaves duplicates that are renumbered in file order so
    # that serial-keyed bookkeeping downstream stays well defined.
    raw_serials = [atom.serial for atom in raw_atoms if atom.serial is not None]
    if len(set(raw_serials)) != len(raw_serials):
        if any(line.startswith("CONECT") for line in lines):
            raise PdbParseError(
                "PDB atom serials are not unique and the file contains CONECT "
                "records, so the connectivity they encode cannot be resolved. "
                "Provide a PDB whose serials are unique or written in hybrid-36."
            )
        raw_atoms = [
            replace(atom, serial=position + 1)
            for position, atom in enumerate(raw_atoms)
        ]
        warnings.append(
            f"Input contains {len(raw_serials) - len(set(raw_serials))} duplicate "
            "atom serials (the five-column PDB serial field wrapped); atoms were "
            "renumbered sequentially in file order."
        )

    atoms = _apply_altloc_policy(raw_atoms, altloc_policy)
    residues = _build_residues(atoms)
    present_serials = {atom.serial for atom in atoms if atom.serial is not None}
    retained = {id(atom) for atom in atoms}
    serial_by_line_index = {
        line_index: atom.serial
        for line_index, atom in zip(atom_line_indices, raw_atoms)
        if id(atom) in retained and atom.serial is not None
    }
    ter_after_serials = _parse_ter_after_serials(model_lines, serial_by_line_index)
    # CONECT records conventionally follow ENDMDL, so inspect the complete
    # file while filtering endpoints to the selected first model/altloc.
    conect_bonds = _parse_conect_bonds(lines, present_serials)
    return PdbStructure(
        path=pdb_path,
        atoms=atoms,
        residues=residues,
        model_count=model_count,
        used_model=1,
        warnings=warnings,
        conect_bonds=conect_bonds,
        ter_after_serials=ter_after_serials,
    )


def _parse_ter_after_serials(
    lines: list[str],
    serial_by_line_index: dict[int, int],
) -> set[int]:
    result: set[int] = set()
    last_serial: int | None = None
    for index, line in enumerate(lines):
        if line.startswith(("ATOM", "HETATM")):
            serial = serial_by_line_index.get(index)
            if serial is not None:
                last_serial = serial
        elif line.startswith("TER") and last_serial is not None:
            result.add(last_serial)
    return result


def _parse_conect_bonds(
    lines: list[str],
    present_serials: set[int],
) -> set[tuple[int, int]]:
    bonds: set[tuple[int, int]] = set()
    for line in lines:
        if not line.startswith("CONECT"):
            continue
        values: list[int] = []
        for field in line[6:].split():
            try:
                values.append(int(field))
            except ValueError:
                continue
        if len(values) < 2:
            continue
        source, *targets = values
        if source not in present_serials:
            continue
        for target in targets:
            if target not in present_serials or target == source:
                continue
            bonds.add((min(source, target), max(source, target)))
    return bonds


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

    serial = _parse_serial(line)
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
    residue_upper = (resname or "").strip().upper()
    # Some AmberTools/MCPB.py PDB writers emit a one-atom ion as ATOM rather
    # than HETATM and align its atom field like a one-letter element (for
    # example " FE "). A coincident elemental residue/atom symbol is the
    # unambiguous ion identity in either record form.
    if (
        record_name in {"ATOM", "HETATM"}
        and residue_upper == upper
        and upper in PERIODIC_ELEMENTS
    ):
        return _canonical_element(upper)
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


HYBRID36_DIGITS_UPPER = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HYBRID36_DIGITS_LOWER = "0123456789abcdefghijklmnopqrstuvwxyz"
SERIAL_FIELD_WIDTH = 5
# Hybrid-36 reserves the decimal range for 1..99999 and then continues with
# "A0000"; see Brookhaven/cctbx hybrid-36 (Grosse-Kunstleve et al., 2006).
HYBRID36_UPPER_OFFSET = 10 ** SERIAL_FIELD_WIDTH - 10 * 36 ** (SERIAL_FIELD_WIDTH - 1)
HYBRID36_LOWER_OFFSET = HYBRID36_UPPER_OFFSET + 26 * 36 ** (SERIAL_FIELD_WIDTH - 1)


def _decode_base36(text: str, digits: str) -> int:
    value = 0
    for character in text:
        value = value * 36 + digits.index(character)
    return value


def _parse_serial(line: str) -> int | None:
    """Decode the atom serial of an ATOM/HETATM line.

    The PDB serial field is five columns wide, which only reaches 99,999 atoms.
    Writers resolve the overflow in three incompatible ways, all accepted here:

    * tleap/ambpdb widen the field into column 12, which is otherwise blank;
    * cctbx-derived tools switch to hybrid-36 ("A0000" follows "99999");
    * some writers emit "*****", which carries no serial at all.
    """

    field = line[6 : 6 + SERIAL_FIELD_WIDTH]
    # tleap and ambpdb push the sixth digit into column 12, which the PDB
    # format leaves blank between the serial and the atom-name field. Only an
    # overflow can land there, so require a value past the five-column limit
    # rather than trusting column 12 on a file whose columns are already off.
    if len(line) > 11 and line[11].isdigit() and field.strip().isdigit():
        widened = line[6:12].strip()
        if widened.isdigit() and int(widened) >= 10 ** SERIAL_FIELD_WIDTH:
            field = widened
    stripped = field.strip()
    if not stripped or set(stripped) == {"*"}:
        return None
    if stripped.lstrip("-").isdigit():
        return int(stripped)
    if all(character in HYBRID36_DIGITS_UPPER for character in stripped):
        return _decode_base36(stripped, HYBRID36_DIGITS_UPPER) + HYBRID36_UPPER_OFFSET
    if all(character in HYBRID36_DIGITS_LOWER for character in stripped):
        return _decode_base36(stripped, HYBRID36_DIGITS_LOWER) + HYBRID36_LOWER_OFFSET
    raise PdbParseError(f"Unparseable atom serial {field!r} in line: {line}")


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
