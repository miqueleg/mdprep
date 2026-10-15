from pathlib import Path

from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


DATA = Path("tests/data")


def test_write_pdb_round_trips_core_records(tmp_path):
    structure = read_pdb(DATA / "protein_with_waters.pdb")
    output = tmp_path / "roundtrip.pdb"

    write_pdb(structure, output)
    reparsed = read_pdb(output)

    assert len(reparsed.atoms) == len(structure.atoms)
    assert [atom.name for atom in reparsed.atoms] == [atom.name for atom in structure.atoms]
    assert [residue.id for residue in reparsed.residues] == [residue.id for residue in structure.residues]
    for original, rewritten in zip(structure.atoms, reparsed.atoms):
        assert rewritten.x == original.x
        assert rewritten.y == original.y
        assert rewritten.z == original.z

    assert output.read_text(encoding="utf-8").rstrip().endswith("END")


def test_write_pdb_round_trips_ter_and_conect(tmp_path):
    source = tmp_path / "connected.pdb"
    source.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  \n"
        "ATOM      2  C   ALA A   1       1.300   0.000   0.000  1.00  0.00           C  \n"
        "TER\n"
        "HETATM    3  C1  LIG A 501       3.000   0.000   0.000  1.00  0.00           C  \n"
        "HETATM    4  O1  LIG A 501       4.200   0.000   0.000  1.00  0.00           O  \n"
        "CONECT    3    4\n"
        "CONECT    4    3\n"
        "END\n",
        encoding="utf-8",
    )
    structure = read_pdb(source)
    output = tmp_path / "roundtrip_connected.pdb"

    assert structure.ter_after_serials == {2}
    assert structure.conect_bonds == {(3, 4)}

    write_pdb(structure, output)
    reparsed = read_pdb(output)

    assert reparsed.ter_after_serials == {2}
    assert reparsed.conect_bonds == {(3, 4)}
    text = output.read_text(encoding="utf-8")
    assert text.count("\nTER\n") == 1
    assert "CONECT    3    4" in text
    assert "CONECT    4    3" in text


# --------------------------------------------------------------------------
# Fields that do not fit
#
# Every width in an ATOM record is a minimum, not a maximum, so a value one
# character too wide shifts the rest of the line right instead of being
# refused. The file still looks like a PDB file and reads back as nonsense
# somewhere else entirely, which is how a four-character ligand name from a
# prediction turned into RDKit complaining about a residue number.
# --------------------------------------------------------------------------

import pytest

from mdprep.structure.models import AtomRecord
from mdprep.structure.pdb_fields import PdbFieldError
from mdprep.structure.writer import format_atom_record


def _atom(**overrides):
    fields = dict(
        serial=2251,
        name="C1",
        altloc="",
        resname="LIG",
        chain_id="A",
        resid=1,
        icode="",
        x=1.0,
        y=2.0,
        z=3.0,
        occupancy=1.0,
        bfactor=0.0,
        element="C",
        record_name="HETATM",
        original_line="",
    )
    fields.update(overrides)
    return AtomRecord(**fields)


@pytest.mark.parametrize(
    "overrides",
    [
        {"resname": "LIG2"},
        {"chain_id": "AKG"},
        {"resid": 12345},
        {"resid": -1000},
    ],
    ids=["wide resname", "wide chain", "resid too high", "resid too low"],
)
def test_a_field_that_would_shift_the_columns_is_refused(overrides):
    with pytest.raises(PdbFieldError):
        format_atom_record(_atom(**overrides))


def test_the_columns_a_reader_depends_on_are_where_they_should_be():
    line = format_atom_record(_atom())

    assert line[17:20] == "LIG"   # resName, columns 18-20
    assert line[21] == "A"        # chainID, column 22
    assert line[22:26] == "   1"  # resSeq, columns 23-26
    assert line[30:38] == "   1.000"  # x, columns 31-38
