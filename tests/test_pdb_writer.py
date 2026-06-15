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

