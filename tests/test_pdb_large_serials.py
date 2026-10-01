"""Regression tests for PDB atom serials beyond the five-column field.

Systems above 99,999 atoms exhaust the PDB serial field. tleap/ambpdb widen it
into column 12, cctbx-derived writers switch to hybrid-36, and some writers
wrap around. All three used to make read_pdb reject its own final output with
"PDB atom serials must be unique when present".
"""

from pathlib import Path

import pytest

from mdprep.structure.pdb import PdbParseError, read_pdb
from mdprep.structure.writer import format_serial_field, write_pdb


def _atom_line(serial_field: str, name: str, resname: str, resid: int, z: float) -> str:
    return (
        f"ATOM  {serial_field}{name:^4}{' '}{resname:>3} A{resid:4d}    "
        f"{1.0:8.3f}{2.0:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          {'O':>2}"
    )


def _write_overflow_pdb(path: Path, atom_count: int, *, first_serial: int = 99998) -> None:
    lines = []
    for offset in range(atom_count):
        serial = first_serial + offset
        lines.append(
            _atom_line(format_serial_field(serial), "O", "WAT", offset + 1, float(offset))
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_tleap_six_digit_serial_overflow_is_parsed_and_unique(tmp_path):
    """tleap pushes the sixth digit into column 12; it is not a duplicate."""
    pdb = tmp_path / "big.pdb"
    _write_overflow_pdb(pdb, 6)

    structure = read_pdb(pdb)

    assert [atom.serial for atom in structure.atoms] == [
        99998, 99999, 100000, 100001, 100002, 100003
    ]
    assert structure.warnings == []
    # The widened serial must not shift the atom-name or residue-name fields.
    assert {atom.name for atom in structure.atoms} == {"O"}
    assert {atom.resname for atom in structure.atoms} == {"WAT"}


def test_six_digit_serials_round_trip_through_the_writer(tmp_path):
    source = tmp_path / "big.pdb"
    _write_overflow_pdb(source, 6)
    structure = read_pdb(source)

    output = tmp_path / "roundtrip.pdb"
    write_pdb(structure, output)
    reparsed = read_pdb(output)

    assert [atom.serial for atom in reparsed.atoms] == [
        atom.serial for atom in structure.atoms
    ]
    assert [atom.name for atom in reparsed.atoms] == [atom.name for atom in structure.atoms]
    assert [atom.resname for atom in reparsed.atoms] == [
        atom.resname for atom in structure.atoms
    ]


@pytest.mark.parametrize(
    ("field", "expected"),
    [("A0000", 100000), ("A0001", 100001), ("ZZZZZ", 43770015), ("a0000", 43770016)],
)
def test_hybrid36_serials_are_decoded(tmp_path, field, expected):
    pdb = tmp_path / "hybrid36.pdb"
    pdb.write_text(_atom_line(field + " ", "O", "WAT", 1, 0.0) + "\nEND\n", encoding="utf-8")

    structure = read_pdb(pdb)

    assert structure.atoms[0].serial == expected
    assert structure.atoms[0].name == "O"


@pytest.mark.parametrize("serial", [1, 99999, 100000, 163469, 999999, 1_000_000, 43770015])
def test_serial_field_is_six_columns_and_round_trips(serial):
    field = format_serial_field(serial)
    assert len(field) == 6

    pdb_line = _atom_line(field, "O", "WAT", 1, 0.0)
    # Columns 13-16 must still hold the atom name and 18-20 the residue name.
    assert pdb_line[12:16].strip() == "O"
    assert pdb_line[17:20].strip() == "WAT"


def test_overflowed_serials_do_not_break_final_validation(tmp_path):
    """The reported failure: a >99,999-atom final PDB was rejected as unparseable."""
    pdb = tmp_path / "final.pdb"
    _write_overflow_pdb(pdb, 2000, first_serial=99_000)

    structure = read_pdb(pdb)

    assert len(structure.atoms) == 2000
    serials = [atom.serial for atom in structure.atoms]
    assert len(set(serials)) == len(serials)
    assert serials[-1] == 100_999


def test_wrapped_duplicate_serials_are_renumbered_with_a_warning(tmp_path):
    """A writer that genuinely wraps modulo 100,000 must not abort the run."""
    pdb = tmp_path / "wrapped.pdb"
    lines = [
        _atom_line(f"{99999:5d} ", "O", "WAT", 1, 0.0),
        _atom_line(f"{0:5d} ", "O", "WAT", 2, 1.0),
        _atom_line(f"{1:5d} ", "O", "WAT", 3, 2.0),
        _atom_line(f"{1:5d} ", "O", "WAT", 4, 3.0),
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    structure = read_pdb(pdb)

    assert [atom.serial for atom in structure.atoms] == [1, 2, 3, 4]
    assert any("renumbered" in warning for warning in structure.warnings)


def test_duplicate_serials_with_conect_records_fail_clearly(tmp_path):
    """Renumbering would silently destroy the connectivity CONECT encodes."""
    pdb = tmp_path / "ambiguous.pdb"
    lines = [
        _atom_line(f"{1:5d} ", "O", "WAT", 1, 0.0),
        _atom_line(f"{1:5d} ", "O", "WAT", 2, 1.0),
        "CONECT    1    2",
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(PdbParseError, match="CONECT"):
        read_pdb(pdb)


def test_ter_markers_survive_serial_overflow(tmp_path):
    pdb = tmp_path / "ter.pdb"
    lines = [
        _atom_line(format_serial_field(99999), "O", "WAT", 1, 0.0),
        "TER",
        _atom_line(format_serial_field(100000), "O", "WAT", 2, 1.0),
        _atom_line(format_serial_field(100001), "O", "WAT", 3, 2.0),
        "TER",
        "END",
    ]
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")

    structure = read_pdb(pdb)

    assert structure.ter_after_serials == {99999, 100001}
