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
