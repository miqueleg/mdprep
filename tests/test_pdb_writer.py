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
        {"resid": -1000},
    ],
    ids=["wide resname", "wide chain", "resid too low"],
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


def test_an_ion_keeps_its_element_when_the_writer_gets_the_column_wrong():
    """tleap writes Fe(II)'s element field as " F", with the "2+" spilling
    into the charge columns. Read literally that is fluorine, and the iron
    then failed to match itself across tleap -- the provisional system said
    fluorine where the input said iron, and the refinement mapping gave up.
    """

    import tempfile

    line = (
        "ATOM   3957  FE  FE2   246     -13.875  64.274  26.944"
        "  1.00  0.00           F2+\n"
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "ion.pdb"
        path.write_text(line + "END\n", encoding="utf-8")
        structure = read_pdb(path)

    assert structure.atoms[0].element.upper() == "FE"


def test_the_ion_rule_does_not_touch_an_ordinary_alpha_carbon():
    """CA is calcium only when its residue says so too."""

    from mdprep.structure.pdb import monatomic_ion_element

    assert monatomic_ion_element("CA", resname="CA", record_name="HETATM") == "Ca"
    assert monatomic_ion_element("CA", resname="ALA", record_name="ATOM") is None
    assert monatomic_ion_element("FE", resname="FE2", record_name="ATOM") == "Fe"
    assert monatomic_ion_element("C1", resname="LIG", record_name="HETATM") is None


def test_a_solvated_box_numbers_its_residues_past_the_four_columns(tmp_path):
    """A solvated system has more than 9,999 residues long before it has
    99,999 atoms, so the residue field overflows first. tleap continues in
    hybrid-36 and so does mdprep, or the final parse of a finished system
    fails on "A000" after all the work is done.
    """

    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.writer import format_resid_field, write_pdb
    from mdprep.structure.models import PdbStructure

    assert format_resid_field(9999) == "9999"
    assert format_resid_field(10000) == "A000"

    atoms = [
        _atom(serial=index, resid=resid, resname="WAT", name="O", element="O")
        for index, resid in enumerate((9999, 10000, 10001, 50000), start=1)
    ]
    output = tmp_path / "big.pdb"
    write_pdb(PdbStructure(path=output, atoms=atoms, residues=[], model_count=1), output)

    assert [atom.resid for atom in read_pdb(output).atoms] == [
        9999,
        10000,
        10001,
        50000,
    ]


# --------------------------------------------------------------------------
# Chain breaks
#
# tleap bonds consecutive residues of one chain unless a TER separates them.
# SyrB2 (PDB 2FCT) is missing residues 57-60; without a TER the prepared
# system had a 6.08 A peptide bond from 56 to 61, and the first picoseconds
# of heating turned it into NaN.
# --------------------------------------------------------------------------


def _backbone(serial, resid, x0):
    """N, CA, C of an alanine laid along x, starting at ``x0``."""

    return [
        _atom(serial=serial, name="N", resname="ALA", resid=resid, x=x0, y=0.0, z=0.0,
              element="N", record_name="ATOM"),
        _atom(serial=serial + 1, name="CA", resname="ALA", resid=resid, x=x0 + 1.46,
              y=0.0, z=0.0, element="C", record_name="ATOM"),
        _atom(serial=serial + 2, name="C", resname="ALA", resid=resid, x=x0 + 2.0,
              y=0.0, z=0.0, element="C", record_name="ATOM"),
    ]


def _chain(tmp_path, *, second_resid, gap):
    """Residue 1, then ``second_resid`` with its N ``gap`` A from 1's C."""

    from mdprep.structure.models import PdbStructure
    from mdprep.structure.writer import write_pdb

    atoms = _backbone(1, 1, 0.0) + _backbone(4, second_resid, 2.0 + gap)
    output = tmp_path / "chain.pdb"
    write_pdb(PdbStructure(path=output, atoms=atoms, residues=[], model_count=1), output)
    return output


def test_a_gap_in_space_gets_a_ter_so_tleap_does_not_bond_across_it(tmp_path):
    output = _chain(tmp_path, second_resid=61, gap=6.08)

    assert read_pdb(output).ter_after_serials >= {3}, "TER after residue 1's last atom"


def test_a_numbering_jump_between_bonded_residues_is_not_a_break(tmp_path):
    """Insertions and renumbered constructs skip numbers while the chain is
    continuous. Only the measured C-N distance decides."""

    output = _chain(tmp_path, second_resid=61, gap=1.33)

    assert 3 not in read_pdb(output).ter_after_serials


def test_consecutive_numbering_does_not_hide_a_gap_in_space(tmp_path):
    """A model can number straight through a stretch it did not build."""

    output = _chain(tmp_path, second_resid=2, gap=4.0)

    assert 3 in read_pdb(output).ter_after_serials
