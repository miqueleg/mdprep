import pytest

from mdprep.ambertools.mol2 import Mol2Error, read_mol2, write_mol2


def test_parse_mol2_atom_records_and_charge_sum():
    mol2 = read_mol2("tests/data/ligands/ligand_sub.good.mol2")

    assert mol2.molecule_name == "SUB"
    assert [atom.name for atom in mol2.atoms] == ["C1", "O1"]
    assert mol2.total_charge == pytest.approx(0.0)


def test_write_mol2_and_reparse(tmp_path):
    mol2 = read_mol2("tests/data/ligands/ligand_sub.good.mol2")
    output = tmp_path / "roundtrip.mol2"

    write_mol2(mol2, output)
    reparsed = read_mol2(output)

    assert [atom.name for atom in reparsed.atoms] == ["C1", "O1"]
    assert reparsed.total_charge == pytest.approx(0.0)


def test_bad_mol2_fails_clearly(tmp_path):
    bad = tmp_path / "bad.mol2"
    bad.write_text("@<TRIPOS>MOLECULE\nBAD\n", encoding="utf-8")

    with pytest.raises(Mol2Error) as excinfo:
        read_mol2(bad)

    assert "missing @<TRIPOS>ATOM" in str(excinfo.value)
