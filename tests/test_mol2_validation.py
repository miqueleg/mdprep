import csv

import pytest

from mdprep.ambertools.mol2 import Mol2Error, read_mol2, validate_and_write_final_mol2
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_config(**updates):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    entry.update(updates)
    data["ligands"] = [entry]
    return make_manifest(data).ligands[0]


def extracted_atoms():
    structure = read_pdb("tests/data/ligands/ligand_sub.pdb")
    return structure.residues[0].atoms


def validate(tmp_path, mol2_path, **updates):
    return validate_and_write_final_mol2(
        mol2_path=mol2_path,
        extracted_atoms=extracted_atoms(),
        ligand=ligand_config(**updates),
        final_mol2_path=tmp_path / "final.mol2",
        charges_csv_path=tmp_path / "charges.csv",
        validation_json_path=tmp_path / "validation.json",
    )


def test_good_mol2_validates_against_extracted_ligand(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.good.mol2")

    assert result.charge_sum_final == pytest.approx(0.0)
    assert result.coordinate_max_deviation == pytest.approx(0.0)


def test_small_charge_residual_is_corrected_and_recorded(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.small_residual.mol2")
    final = read_mol2(result.final_mol2_path)

    assert result.charge_correction_applied == pytest.approx(-0.005)
    assert final.total_charge == pytest.approx(0.0)
    with result.charges_csv_path.open("r", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle))


def test_large_charge_mismatch_fails(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.bad_charge.mol2")

    assert "charge sum" in str(excinfo.value)


def test_atom_count_mismatch_fails(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.atom_count_mismatch.mol2")

    assert "atom-count mismatch" in str(excinfo.value)


def test_coordinate_deviation_fails_when_not_allowed(tmp_path):
    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, "tests/data/ligands/ligand_sub.shifted_coords.mol2")

    assert "coordinates deviate" in str(excinfo.value)


def test_renamed_atoms_are_restored_when_order_is_compatible(tmp_path):
    result = validate(tmp_path, "tests/data/ligands/ligand_sub.renamed.mol2")
    final = read_mol2(result.final_mol2_path)

    assert [atom.name for atom in final.atoms] == ["C1", "O1"]
    assert result.atom_names_preserved


def test_gaff_aromatic_carbon_type_ca_is_carbon_not_calcium(tmp_path):
    mol2 = tmp_path / "gaff_ca.mol2"
    mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "SUB",
                " 2 1 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                "      1 C1           5.0000     5.0000     5.0000 ca          1 SUB        0.100000",
                "      2 O1           6.0000     5.0000     5.0000 o           1 SUB       -0.100000",
                "@<TRIPOS>BOND",
                "     1    1    2 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = validate(tmp_path, mol2)

    assert result.charge_sum_final == pytest.approx(0.0)


def test_gaff_aromatic_nitrogen_type_na_is_nitrogen_not_sodium(tmp_path):
    mol2 = tmp_path / "gaff_na.mol2"
    mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "SUB",
                " 2 1 0 0 0",
                "SMALL",
                "USER_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                "      1 C1           5.0000     5.0000     5.0000 c3          1 SUB        0.100000",
                "      2 O1           6.0000     5.0000     5.0000 na          1 SUB       -0.100000",
                "@<TRIPOS>BOND",
                "     1    1    2 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(Mol2Error) as excinfo:
        validate(tmp_path, mol2)

    assert "['C', 'O'] != ['C', 'N']" in str(excinfo.value)


def test_an_untyped_atom_is_reported_as_antechamber_failing_to_type_it():
    """antechamber writes DU for an atom it could not assign a type to.

    Read as an element that is deuterium, so a run whose four hydrogens
    antechamber could not perceive reported "H,H,H,H against D,D,D,D" and
    sent the reader looking for isotopes. The message now says what
    happened, and which atoms.
    """

    from mdprep.ambertools.mol2 import Mol2Atom, _validate_element_order
    from mdprep.structure.models import AtomRecord

    def pdb_atom(name, element):
        return AtomRecord(
            serial=1, name=name, altloc="", resname="LIG", chain_id="A", resid=1,
            icode="", x=0.0, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0,
            element=element, record_name="HETATM", original_line="",
        )

    def mol2_atom(name, atom_type):
        return Mol2Atom(
            atom_id=1, name=name, x=0.0, y=0.0, z=0.0,
            atom_type=atom_type, subst_id=1, subst_name="LIG", charge=0.0,
        )

    with pytest.raises(Mol2Error) as caught:
        _validate_element_order(
            [mol2_atom("C1", "c3"), mol2_atom("H1", "DU")],
            [pdb_atom("C1", "C"), pdb_atom("H1", "H")],
            "a_lig_1",
        )

    message = str(caught.value)
    assert "could not assign an atom type" in message
    assert "H1" in message
    # And it no longer claims the elements simply disagree.
    assert "element-order mismatch" not in message
