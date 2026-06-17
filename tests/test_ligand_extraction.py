import pytest

from mdprep.ligands.extract import LigandExtractionError, extract_configured_ligands
from mdprep.structure.models import AtomRecord
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import format_atom_record
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_manifest(input_structure: str):
    data = manifest_data(input_structure)
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
            "charge_method": "user_mol2",
        }
    ]
    return make_manifest(data)


def test_extract_configured_ligand_preserves_identity(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    manifest = ligand_manifest("tests/data/protein_two_ligands.pdb")

    extracted = extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert len(extracted) == 1
    ligand = extracted[0]
    assert [atom.name for atom in ligand.atoms] == ["C1", "O1"]
    assert [(atom.x, atom.y, atom.z) for atom in ligand.atoms] == [(5.0, 5.0, 5.0), (6.0, 5.0, 5.0)]
    assert ligand.residue.id.resname == "SUB"
    assert ligand.residue.id.chain_id == "B"
    assert ligand.pdb_path.exists()
    assert ligand.identity_path.exists()


def test_missing_ligand_selector_fails_clearly(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["ligands"] = [ligand_entry("missing", "B", "SUB", 999)]
    manifest = make_manifest(data)

    with pytest.raises(LigandExtractionError) as excinfo:
        extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert "selector did not resolve exactly one residue" in str(excinfo.value)


def test_multiple_ligands_are_extracted_independently(tmp_path):
    structure = read_pdb("tests/data/protein_two_ligands.pdb")
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [
        ligand_entry("sub_501", "B", "SUB", 501),
        ligand_entry("cof_601", "C", "COF", 601),
    ]
    manifest = make_manifest(data)

    extracted = extract_configured_ligands(structure, manifest, output_dir=tmp_path)

    assert [ligand.config.id for ligand in extracted] == ["sub_501", "cof_601"]
    assert [atom.name for atom in extracted[0].atoms] == ["C1", "O1"]
    assert [atom.name for atom in extracted[1].atoms] == ["N1", "C1"]


def test_extract_ligand_preserves_ligand_conect_records(tmp_path):
    pdb_path = tmp_path / "with_conect.pdb"
    atoms = [
        AtomRecord(
            serial=1,
            name="CA",
            altloc=None,
            resname="ALA",
            chain_id="A",
            resid=1,
            icode=None,
            x=0.0,
            y=0.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="C",
            record_name="ATOM",
            original_line="",
        ),
        AtomRecord(
            serial=4,
            name="C1",
            altloc=None,
            resname="SUB",
            chain_id="B",
            resid=501,
            icode=None,
            x=5.0,
            y=5.0,
            z=5.0,
            occupancy=1.0,
            bfactor=0.0,
            element="C",
            record_name="HETATM",
            original_line="",
        ),
        AtomRecord(
            serial=5,
            name="O1",
            altloc=None,
            resname="SUB",
            chain_id="B",
            resid=501,
            icode=None,
            x=6.0,
            y=5.0,
            z=5.0,
            occupancy=1.0,
            bfactor=0.0,
            element="O",
            record_name="HETATM",
            original_line="",
        ),
        AtomRecord(
            serial=6,
            name="H1",
            altloc=None,
            resname="SUB",
            chain_id="B",
            resid=501,
            icode=None,
            x=5.0,
            y=6.0,
            z=5.0,
            occupancy=1.0,
            bfactor=0.0,
            element="H",
            record_name="HETATM",
            original_line="",
        ),
    ]
    pdb_path.write_text(
        "".join(format_atom_record(atom) for atom in atoms)
        + "CONECT    4    5    1    6\n"
        + "CONECT    5    4\n"
        + "END\n",
        encoding="utf-8",
    )
    data = manifest_data(str(pdb_path))
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [ligand_entry("sub_501", "B", "SUB", 501)]
    manifest = make_manifest(data)

    extracted = extract_configured_ligands(read_pdb(pdb_path), manifest, output_dir=tmp_path / "out")

    output = extracted[0].pdb_path.read_text(encoding="utf-8")
    assert "CONECT    4    5    6" in output
    assert "CONECT    4    5    1    6" not in output
    assert any("Preserved" in warning and "CONECT" in warning for warning in extracted[0].warnings)


def test_extract_ligand_renames_duplicate_atom_names_deterministically(tmp_path):
    pdb_path = tmp_path / "duplicate_names.pdb"
    atoms = [
        AtomRecord(
            serial=1,
            name="C",
            altloc=None,
            resname="SAL",
            chain_id="B",
            resid=777,
            icode=None,
            x=0.0,
            y=0.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="C",
            record_name="HETATM",
            original_line="",
        ),
        AtomRecord(
            serial=2,
            name="C",
            altloc=None,
            resname="SAL",
            chain_id="B",
            resid=777,
            icode=None,
            x=1.0,
            y=0.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="C",
            record_name="HETATM",
            original_line="",
        ),
        AtomRecord(
            serial=3,
            name="O",
            altloc=None,
            resname="SAL",
            chain_id="B",
            resid=777,
            icode=None,
            x=0.0,
            y=1.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="O",
            record_name="HETATM",
            original_line="",
        ),
        AtomRecord(
            serial=4,
            name="O",
            altloc=None,
            resname="SAL",
            chain_id="B",
            resid=777,
            icode=None,
            x=0.0,
            y=0.0,
            z=1.0,
            occupancy=1.0,
            bfactor=0.0,
            element="O",
            record_name="HETATM",
            original_line="",
        ),
    ]
    pdb_path.write_text("".join(format_atom_record(atom) for atom in atoms) + "END\n", encoding="utf-8")
    data = manifest_data(str(pdb_path))
    data["structure"]["remove_unknown_heterogens"] = False
    data["ligands"] = [ligand_entry("substrate_sal", "B", "SAL", 777)]
    manifest = make_manifest(data)

    extracted = extract_configured_ligands(read_pdb(pdb_path), manifest, output_dir=tmp_path / "out")

    assert [atom.name for atom in extracted[0].atoms] == ["C1", "C2", "O1", "O2"]
    assert len(set(atom.name for atom in extracted[0].atoms)) == 4
    assert any("not unique" in warning for warning in extracted[0].warnings)
