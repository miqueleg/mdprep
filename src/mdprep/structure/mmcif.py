"""Read the atom_site table of an mmCIF file, without a CIF library.

Structure prediction writes mmCIF -- Boltz emits ModelCIF, and the RCSB
serves mmCIF for everything -- so a preparation tool that only reads PDB
turns a predicted structure into a conversion chore before it starts.

Parsed by hand rather than with gemmi, for one reason: the web builder runs
mdprep under Pyodide, which has pydantic and PyYAML and nothing else. A CIF
reader that pulls in a compiled dependency would work everywhere except the
place a student is most likely to use it. The atom_site loop is a whitespace
separated table, which is little enough to read directly.

Coordinates come from atom_site. Besides it, only what a deposition declares
about bonds is read -- ``_struct_conn`` and ``_pdbx_struct_mod_residue``, see
``structure/links.py`` -- because a bond cannot be told from a clash by
distance. The chemical component dictionary a file carries is not consulted,
so a wrong one cannot mislead it.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path

from mdprep.structure.links import (
    parse_mmcif_links,
    parse_mmcif_modres,
    resolve_links,
    resolve_residue_keys,
)
from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
from mdprep.structure.pdb_fields import fit_atoms_to_pdb_fields
from mdprep.structure.pdb import (
    VALID_ALTLOC_POLICIES,
    canonical_element,
    monatomic_ion_element,
    AltlocPolicy,
    PdbParseError,
    _apply_altloc_policy,
    label_unlabeled_alternates,
)

# CIF spells an absent value these two ways, and neither means the string.
CIF_NULL = {".", "?"}


class MmcifParseError(PdbParseError):
    """Raised when an mmCIF file cannot be read as a structure."""


def split_cif_row(line: str) -> list[str]:
    """Split one data row, honouring the quoting CIF allows.

    An atom name containing a prime -- O5' in a nucleotide -- is quoted, and
    splitting on whitespace alone would lose it.
    """

    tokens: list[str] = []
    rest = line.strip()
    while rest:
        if rest[0] in "'\"":
            quote = rest[0]
            end = rest.find(quote, 1)
            while end != -1 and end + 1 < len(rest) and not rest[end + 1].isspace():
                # A quote only closes the value when whitespace follows it.
                end = rest.find(quote, end + 1)
            if end == -1:
                raise MmcifParseError(f"Unterminated quote in mmCIF row: {line.strip()!r}")
            tokens.append(rest[1:end])
            rest = rest[end + 1 :].lstrip()
        else:
            head, _, tail = rest.partition(" ")
            tokens.append(head)
            rest = tail.lstrip()
    return tokens


def _value(row: dict[str, str], *keys: str) -> str | None:
    """First of ``keys`` the row actually carries a value for."""

    for key in keys:
        item = row.get(key)
        if item is not None and item not in CIF_NULL and item != "":
            return item
    return None


def read_atom_site_rows(text: str) -> list[dict[str, str]]:
    """Every atom_site row, as a mapping from unprefixed tag to raw value."""

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        tags: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_"):
            tags.append(lines[cursor].strip())
            cursor += 1
        if not tags or not tags[0].startswith("_atom_site."):
            index = cursor
            continue
        columns = [tag.split(".", 1)[1] for tag in tags]
        rows: list[dict[str, str]] = []
        while cursor < len(lines):
            line = lines[cursor]
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "loop_":
                break
            if stripped.startswith("_") or stripped.startswith("data_"):
                break
            values = split_cif_row(line)
            if len(values) != len(columns):
                raise MmcifParseError(
                    f"mmCIF atom_site row {len(rows) + 1} has {len(values)} values "
                    f"for {len(columns)} columns: {stripped!r}"
                )
            rows.append(dict(zip(columns, values)))
            cursor += 1
        return rows
    raise MmcifParseError(
        "No _atom_site loop found. mdprep reads coordinates from that table, "
        "so a file without one carries no structure it can prepare."
    )


def read_mmcif(
    path: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
) -> PdbStructure:
    """Read an mmCIF file into the same structure a PDB file would give."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Input structure not found: {source}")
    if altloc_policy not in VALID_ALTLOC_POLICIES:
        raise MmcifParseError(f"Invalid altloc policy: {altloc_policy}")

    text = source.read_text(encoding="utf-8", errors="replace")
    rows = read_atom_site_rows(text)
    if not rows:
        raise MmcifParseError(f"mmCIF contains no atoms: {source}")

    models = list(dict.fromkeys(row.get("pdbx_PDB_model_num", "1") for row in rows))
    warnings: list[str] = []
    if len(models) > 1:
        warnings.append(
            f"Input contains {len(models)} models; using the first ({models[0]}) only."
        )

    atoms: list[AtomRecord] = []
    for number, row in enumerate(rows, start=1):
        if row.get("pdbx_PDB_model_num", "1") != models[0]:
            continue
        record = _value(row, "group_PDB") or "ATOM"
        if record not in {"ATOM", "HETATM"}:
            raise MmcifParseError(
                f"Unsupported mmCIF record {record!r} in atom_site row {number}"
            )
        # Author identifiers where present, which is what a PDB file carries
        # and what every selector in a manifest is written against. Prediction
        # output often has only the label ones.
        name = _value(row, "auth_atom_id", "label_atom_id")
        resname = _value(row, "auth_comp_id", "label_comp_id")
        # label_seq_id is null for every non-polymer, so auth_seq_id first.
        resid = _value(row, "auth_seq_id", "label_seq_id")
        chain = _value(row, "auth_asym_id", "label_asym_id") or ""
        element = _value(row, "type_symbol")
        missing = [
            label
            for label, item in (
                ("atom name", name),
                ("residue name", resname),
                ("residue number", resid),
                ("element", element),
            )
            if item is None
        ]
        if missing:
            raise MmcifParseError(
                f"mmCIF atom_site row {number} has no {', '.join(missing)}. "
                "mdprep needs each of those to identify an atom."
            )
        try:
            x, y, z = (float(row[key]) for key in ("Cartn_x", "Cartn_y", "Cartn_z"))
        except (KeyError, ValueError) as exc:
            raise MmcifParseError(
                f"mmCIF atom_site row {number} has no usable coordinates: {exc}"
            ) from exc
        if not all(isfinite(value) for value in (x, y, z)):
            raise MmcifParseError(
                f"mmCIF atom_site row {number} has non-finite coordinates"
            )
        try:
            serial = int(_value(row, "id") or number)
            residue_number = int(resid)
        except ValueError as exc:
            raise MmcifParseError(
                f"mmCIF atom_site row {number} has a non-numeric id or residue "
                f"number: {exc}"
            ) from exc
        # Same rule the PDB reader applies: a one-atom residue named after
        # its element settles its own identity, and both readers have to
        # agree or the same structure in two formats behaves differently.
        # mmCIF states the element, but not its spelling: a file written
        # from PDB columns carries "FE" where another carries "Fe". The two
        # formats have to hand back the same structure, so the symbol is
        # canonicalised here the way the PDB reader does it.
        element = canonical_element(element) if element else element
        ion = monatomic_ion_element(name, resname=resname, record_name=record)
        atoms.append(
            AtomRecord(
                serial=serial,
                name=name,
                altloc=_value(row, "label_alt_id", "auth_alt_id"),
                resname=resname,
                chain_id=chain,
                resid=residue_number,
                icode=_value(row, "pdbx_PDB_ins_code"),
                x=x,
                y=y,
                z=z,
                occupancy=_as_float(_value(row, "occupancy")),
                bfactor=_as_float(_value(row, "B_iso_or_equiv")),
                element=ion or element,
                record_name=record,  # type: ignore[arg-type]
                original_line="",
            )
        )

    atoms, unlabeled = label_unlabeled_alternates(atoms)
    warnings.extend(unlabeled)
    as_written = atoms
    atoms = _apply_altloc_policy(atoms, altloc_policy)
    if not atoms:
        raise MmcifParseError(f"mmCIF contains no usable atoms: {source}")
    # mmCIF puts no width on a name and a PDB record does, and everything
    # downstream of here is fixed-column. Fit the names now, while this is
    # still the only code that has seen them.
    atoms, field_warnings, assigned_resnames = fit_atoms_to_pdb_fields(atoms)
    warnings.extend(field_warnings)
    return PdbStructure(
        path=source,
        atoms=atoms,
        residues=build_residues(atoms),
        model_count=len(models),
        warnings=warnings,
        assigned_resnames=assigned_resnames,
        links=resolve_links(parse_mmcif_links(text), as_written, atoms),
        modified_residue_parents=resolve_residue_keys(
            parse_mmcif_modres(text), as_written, atoms
        ),
    )


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
