import pytest

from mdprep.ambertools.mol2 import Mol2Error, _element_from_atom_type, read_mol2, write_mol2


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


@pytest.mark.parametrize(
    ("atom_type", "element"),
    [
        ("ca", "C"),
        ("cu", "C"),
        ("na", "N"),
        ("n2", "N"),
        ("oh", "O"),
        ("ss", "S"),
        ("p5", "P"),
        ("cl", "CL"),
        ("br", "BR"),
        ("i", "I"),
        ("f", "F"),
        ("Fe", "FE"),
        ("FE", "FE"),
        ("fe", "FE"),
        ("Au", "AU"),
        ("AU", "AU"),
        ("au", "AU"),
        ("Zn", "ZN"),
        ("SI", "SI"),
        ("si", "SI"),
    ],
)
def test_mol2_atom_type_to_element_is_forcefield_and_periodic_aware(atom_type, element):
    assert _element_from_atom_type(atom_type) == element
