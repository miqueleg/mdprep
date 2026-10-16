"""A PDB written by mdprep must read back as the structure that was written.

Every fixed-column defect found in this codebase so far was the same shape: a
field that did not fit the columns the format gives it, written anyway, and
read back as something else. The residue name that pushed every later column
one place right. The residue number past 9,999 that came back unparseable.
The iron whose element column said fluorine.

Each was found by a user running into it, one at a time, and each was fixed
with a test for that one case. This is the loop that catches the class: build
structures that probe the edges of every field, write them, read them back,
and insist on getting the same thing. A field that cannot be represented must
raise, because the one outcome that must never happen is a file that looks
fine and means something else.

Deterministic by construction -- the cases are enumerated rather than
sampled, so a failure here names the field that broke rather than a seed.
"""

from itertools import product
from pathlib import Path

import pytest

from mdprep.structure.models import AtomRecord, PdbStructure
from mdprep.structure.pdb import read_pdb
from mdprep.structure.pdb_fields import PdbFieldError
from mdprep.structure.writer import write_pdb


# Values at and around the limit of each fixed-column field. A PDB record
# gives a residue name three columns, a chain one, an atom name four and a
# residue number four; serials get five, continuing into hybrid-36.
RESNAMES = ["A", "AB", "LIG", "HOH", "FE2"]
CHAIN_IDS = ["A", "z", "0", " "]
RESIDS = [-99, 0, 1, 999, 9999, 10000, 10001, 46655, 46656]
ATOM_NAMES = ["C", "CA", "C1", "FE", "HD11", "1HB"]
ELEMENTS = ["C", "N", "O", "FE", "CL", "ZN"]
SERIALS = [1, 99999, 100000, 999999, 1000000]
# Inside the eight columns a coordinate gets; the sign costs a digit.
COORDINATES = [0.0, -0.001, 999.999, -999.999]


def atom(**overrides) -> AtomRecord:
    fields = dict(
        serial=1,
        name="CA",
        altloc="",
        resname="ALA",
        chain_id="A",
        resid=1,
        icode="",
        x=1.0,
        y=2.0,
        z=3.0,
        occupancy=1.0,
        bfactor=0.0,
        element="C",
        record_name="ATOM",
        original_line="",
    )
    fields.update(overrides)
    return AtomRecord(**fields)


def round_trip(atoms: list[AtomRecord], tmp_path: Path) -> list[AtomRecord]:
    """Write these atoms and read them back, or let the writer refuse."""

    output = tmp_path / "probe.pdb"
    write_pdb(
        PdbStructure(path=output, atoms=atoms, residues=[], model_count=1), output
    )
    return read_pdb(output).atoms


def assert_same(written: AtomRecord, read_back: AtomRecord, field: str) -> None:
    before = getattr(written, field)
    after = getattr(read_back, field)
    if field in {"chain_id", "altloc", "icode"}:
        before = (before or "").strip()
        after = (after or "").strip()
    if field in {"resname", "name", "element"}:
        before = (before or "").strip().upper()
        after = (after or "").strip().upper()
    assert after == before, (
        f"{field} came back as {after!r} after being written as {before!r}. "
        "A PDB that reads back as something else is worse than one that "
        "refuses to be written."
    )


@pytest.mark.parametrize("resid", RESIDS)
def test_a_residue_number_survives_the_round_trip(resid, tmp_path):
    """Four columns do not hold a solvated box's residue count."""

    [read_back] = round_trip([atom(resid=resid)], tmp_path)

    assert_same(atom(resid=resid), read_back, "resid")


@pytest.mark.parametrize("serial", SERIALS)
def test_an_atom_serial_survives_the_round_trip(serial, tmp_path):
    [read_back] = round_trip([atom(serial=serial)], tmp_path)

    assert read_back.serial == serial


@pytest.mark.parametrize("resname,chain_id", list(product(RESNAMES, CHAIN_IDS)))
def test_a_residue_name_and_chain_survive_together(resname, chain_id, tmp_path):
    """These two share a boundary: a name one character too wide lands in the
    chain column, and the chain then lands in the residue number."""

    written = atom(resname=resname, chain_id=chain_id)

    [read_back] = round_trip([written], tmp_path)

    assert_same(written, read_back, "resname")
    assert_same(written, read_back, "chain_id")
    assert_same(written, read_back, "resid")


@pytest.mark.parametrize("name,element", list(product(ATOM_NAMES, ELEMENTS)))
def test_an_atom_name_and_its_element_survive_together(name, element, tmp_path):
    """The atom-name field is read by column alignment to tell a two-letter
    element from a one-letter element with a digit, so the two cannot be
    checked apart."""

    written = atom(name=name, element=element, resname="LIG", record_name="HETATM")

    [read_back] = round_trip([written], tmp_path)

    assert_same(written, read_back, "name")
    assert_same(written, read_back, "element")


@pytest.mark.parametrize("value", COORDINATES)
def test_coordinates_survive_the_round_trip(value, tmp_path):
    """Three decimal places, and a field that overflows takes a digit off the
    front rather than complaining."""

    written = atom(x=value, y=-value, z=value)

    [read_back] = round_trip([written], tmp_path)

    for axis in ("x", "y", "z"):
        assert getattr(read_back, axis) == pytest.approx(
            getattr(written, axis), abs=5e-4
        ), f"{axis} came back as {getattr(read_back, axis)}"


def test_a_whole_structure_of_edge_cases_survives_at_once(tmp_path):
    """One at a time is not the same test: fields interact through shared
    column boundaries, and a solvated system has all of these at once."""

    atoms = []
    serial = 1
    for resid, resname, chain_id in product(RESIDS, RESNAMES, ["A", "B"]):
        atoms.append(
            atom(
                serial=serial,
                resid=resid,
                resname=resname,
                chain_id=chain_id,
                record_name="HETATM" if resname in {"LIG", "FE2"} else "ATOM",
                name="FE" if resname == "FE2" else "CA",
                element="FE" if resname == "FE2" else "C",
            )
        )
        serial += 1

    read_back = round_trip(atoms, tmp_path)

    assert len(read_back) == len(atoms)
    for written, got in zip(atoms, read_back):
        for field in ("resid", "resname", "chain_id", "name", "element", "serial"):
            assert_same(written, got, field) if field not in {"serial"} else None
        assert got.serial == written.serial


@pytest.mark.parametrize(
    "overrides",
    [
        {"resname": "LIG2"},
        {"chain_id": "AKG"},
        {"resid": -1000},
        {"x": -1234.567},
        {"z": 12345.678},
    ],
    ids=[
        "resname too wide",
        "chain too wide",
        "resid too low",
        "coordinate too negative",
        "coordinate too large",
    ],
)
def test_what_cannot_be_represented_is_refused_not_mangled(overrides, tmp_path):
    """The failure mode this whole file exists to prevent."""

    with pytest.raises(PdbFieldError):
        round_trip([atom(**overrides)], tmp_path)
