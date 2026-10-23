"""Bonds and modifications a structure file declares about itself.

Coordinates do not say which atoms are bonded across residues, and distance
alone cannot settle it: a predicted complex has ligand atoms 1.8 A from a
protein atom that are a clash, not a bond. A deposition says so explicitly --
a PDB file in LINK records, an mmCIF in ``_struct_conn`` -- and which polymer
residues are modified versions of a standard one, in MODRES or
``_pdbx_struct_mod_residue``.

mdprep needs both. A glycan is bonded to an asparagine, a covalent
intermediate to an aspartate, a myeloperoxidase heme to the protein in three
places: built as free molecules, each would sit at bond length from the
protein and nothing would say so. A selenomethionine inside a chain is a
residue of the chain, not a ligand.

Links are read against the atoms as the file wrote them and kept by serial,
so they follow an atom through whatever the reader renames afterwards -- a
chain id too wide for a PDB column, a metal moved out of its cofactor.
"""

from __future__ import annotations

from dataclasses import dataclass

from mdprep.structure.models import AtomRecord

Identity = tuple[str, str, int, str | None, str]
ResidueKey = tuple[str, str, int, str | None]

# Elements a bond to which is coordination, not a covalent link.
METAL_ELEMENTS = frozenset(
    {
        "LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
        "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD",
        "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
        "AL", "GA", "IN", "TL", "SN", "PB", "BI",
        "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU",
        "U",
    }
)


@dataclass(frozen=True)
class StructureLink:
    """One declared bond between two residues."""

    first: Identity
    second: Identity
    # covalent, metal, disulfide, hydrogen or other.
    kind: str

    def residues(self) -> tuple[ResidueKey, ResidueKey]:
        return (self.first[:4], self.second[:4])  # type: ignore[return-value]


@dataclass(frozen=True)
class RawLink:
    first: Identity
    second: Identity
    # What the file says the bond is, when it says: mmCIF does, PDB does not.
    declared_kind: str | None = None


def parse_pdb_links(lines: list[str]) -> list[RawLink]:
    """LINK records, by the fixed columns of the PDB format."""

    links: list[RawLink] = []
    for line in lines:
        if not line.startswith("LINK  "):
            continue
        padded = line.rstrip("\n").ljust(80)
        # A bond to a symmetry mate joins this molecule to a copy of itself
        # the file does not contain.
        if any(padded[field].strip() not in {"", "1555"} for field in (slice(59, 65), slice(66, 72))):
            continue
        try:
            first = _pdb_identity(padded, name=(12, 16), resname=(17, 20), chain=21, resid=(22, 26), icode=26)
            second = _pdb_identity(padded, name=(42, 46), resname=(47, 50), chain=51, resid=(52, 56), icode=56)
        except ValueError:
            continue
        links.append(RawLink(first=first, second=second))
    return links


def parse_pdb_modres(lines: list[str]) -> dict[ResidueKey, str]:
    """MODRES records: each modified residue and its standard parent."""

    parents: dict[ResidueKey, str] = {}
    for line in lines:
        if not line.startswith("MODRES"):
            continue
        padded = line.rstrip("\n").ljust(80)
        try:
            resid = int(padded[18:22])
        except ValueError:
            continue
        resname = padded[12:15].strip()
        parent = padded[24:27].strip()
        if resname and parent:
            parents[(padded[16], resname, resid, padded[22].strip() or None)] = parent
    return parents


def parse_mmcif_links(text: str) -> list[RawLink]:
    """``_struct_conn`` rows, by author identifiers where present."""

    links: list[RawLink] = []
    for row in read_category_rows(text, "_struct_conn."):
        if any(
            (_cif_value(row, key) or "1_555") != "1_555"
            for key in ("ptnr1_symmetry", "ptnr2_symmetry")
        ):
            continue
        try:
            first = _cif_identity(row, "ptnr1", "pdbx_ptnr1_PDB_ins_code")
            second = _cif_identity(row, "ptnr2", "pdbx_ptnr2_PDB_ins_code")
        except (KeyError, ValueError):
            continue
        kind = (row.get("conn_type_id") or "").lower()
        links.append(RawLink(first=first, second=second, declared_kind=kind or None))
    return links


def parse_mmcif_modres(text: str) -> dict[ResidueKey, str]:
    """``_pdbx_struct_mod_residue`` rows: each modified residue and its parent."""

    parents: dict[ResidueKey, str] = {}
    for row in read_category_rows(text, "_pdbx_struct_mod_residue."):
        chain = _cif_value(row, "auth_asym_id", "label_asym_id")
        resname = _cif_value(row, "auth_comp_id", "label_comp_id")
        resid = _cif_value(row, "auth_seq_id", "label_seq_id")
        parent = _cif_value(row, "parent_comp_id")
        if not (chain and resname and resid and parent):
            continue
        try:
            number = int(resid)
        except ValueError:
            continue
        parents[(chain, resname, number, _cif_value(row, "PDB_ins_code"))] = parent
    return parents


def resolve_links(
    raw_links: list[RawLink],
    atoms_as_written: list[AtomRecord],
    atoms_as_read: list[AtomRecord],
) -> tuple[StructureLink, ...]:
    """Each link against the atoms the reader kept, under their final names.

    An endpoint that names no atom -- an alternate location the altloc
    policy discarded, a residue outside the first model -- drops the link.
    """

    final = _final_atoms(atoms_as_written, atoms_as_read)
    resolved: list[StructureLink] = []
    for raw in raw_links:
        first = final.get(raw.first)
        second = final.get(raw.second)
        if first is None or second is None or first.residue_key == second.residue_key:
            continue
        link = StructureLink(
            first=first.atom_identity,
            second=second.atom_identity,
            kind=_kind(raw.declared_kind, first, second),
        )
        if link not in resolved:
            resolved.append(link)
    return tuple(resolved)


def resolve_residue_keys(
    parents: dict[ResidueKey, str],
    atoms_as_written: list[AtomRecord],
    atoms_as_read: list[AtomRecord],
) -> dict[ResidueKey, str]:
    """The same mapping, keyed by the residues as the reader named them."""

    final = _final_atoms(atoms_as_written, atoms_as_read)
    resolved: dict[ResidueKey, str] = {}
    for atom in atoms_as_written:
        parent = parents.get(atom.residue_key)
        kept = final.get(atom.atom_identity)
        if parent is not None and kept is not None:
            resolved[kept.residue_key] = parent
    return resolved


def _final_atoms(
    atoms_as_written: list[AtomRecord], atoms_as_read: list[AtomRecord]
) -> dict[Identity, AtomRecord]:
    """Each written identity to the atom the reader kept for it, by serial.

    An identity can name several written atoms -- one per alternate
    location -- of which the altloc policy kept at most one.
    """

    final_by_serial = {atom.serial: atom for atom in atoms_as_read if atom.serial is not None}
    found: dict[Identity, AtomRecord] = {}
    for atom in atoms_as_written:
        kept = final_by_serial.get(atom.serial) if atom.serial is not None else None
        if kept is not None:
            found.setdefault(atom.atom_identity, kept)
    return found


def read_category_rows(text: str, prefix: str) -> list[dict[str, str]]:
    """Every row of one mmCIF category, in loop form or as single key-values.

    A category with one row is written as ``_cat.key value`` lines rather
    than as a loop, which is how a deposition with a single link writes it.
    """

    from mdprep.structure.mmcif import split_cif_row

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "loop_":
            tags: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
                tags.append(lines[cursor].strip())
                cursor += 1
            if not tags or not tags[0].startswith(prefix):
                index = cursor
                continue
            columns = [tag.split(".", 1)[1] for tag in tags]
            rows: list[dict[str, str]] = []
            values: list[str] = []
            while cursor < len(lines):
                line = lines[cursor]
                body = line.strip()
                if not body or body.startswith("#") or body == "loop_" or body.startswith(("_", "data_")):
                    break
                values.extend(split_cif_row(line))
                while len(values) >= len(columns):
                    rows.append(dict(zip(columns, values[: len(columns)])))
                    values = values[len(columns):]
                cursor += 1
            return rows
        if stripped.startswith(prefix):
            row: dict[str, str] = {}
            cursor = index
            while cursor < len(lines) and lines[cursor].strip().startswith(prefix):
                parts = split_cif_row(lines[cursor].strip())
                if len(parts) >= 2:
                    row[parts[0].split(".", 1)[1]] = parts[1]
                cursor += 1
            return [row]
        index += 1
    return []


def _kind(declared: str | None, first: AtomRecord, second: AtomRecord) -> str:
    if declared:
        if declared.startswith("covale"):
            return "covalent"
        if declared.startswith("metalc"):
            return "metal"
        if declared.startswith("disulf"):
            return "disulfide"
        if declared.startswith("hydrog"):
            return "hydrogen"
        return "other"
    elements = {(atom.element or "").strip().upper() for atom in (first, second)}
    if elements & METAL_ELEMENTS:
        return "metal"
    if elements == {"S"} and {first.resname, second.resname} <= {"CYS", "CYX"}:
        return "disulfide"
    return "covalent"


def _pdb_identity(
    line: str,
    *,
    name: tuple[int, int],
    resname: tuple[int, int],
    chain: int,
    resid: tuple[int, int],
    icode: int,
) -> Identity:
    return (
        line[chain],
        line[resname[0]:resname[1]].strip(),
        int(line[resid[0]:resid[1]]),
        line[icode].strip() or None,
        line[name[0]:name[1]].strip(),
    )


def _cif_identity(row: dict[str, str], partner: str, icode_key: str) -> Identity:
    chain = _cif_value(row, f"{partner}_auth_asym_id", f"{partner}_label_asym_id")
    resname = _cif_value(row, f"{partner}_auth_comp_id", f"{partner}_label_comp_id")
    resid = _cif_value(row, f"{partner}_auth_seq_id", f"{partner}_label_seq_id")
    atom = _cif_value(row, f"{partner}_label_atom_id", f"{partner}_auth_atom_id")
    if not (chain and resname and resid and atom):
        raise KeyError(partner)
    return (chain, resname, int(resid), _cif_value(row, icode_key), atom)


def _cif_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value not in {".", "?"}:
            return value.strip("'\"")
    return None
