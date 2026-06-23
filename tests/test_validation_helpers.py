import pytest

from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.writer import write_pdb
from mdprep.validation.topology import FinalValidationError, validate_final_outputs
from tests.test_structure_normalize import make_manifest, manifest_data


def test_missing_final_files_fail(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    manifest = make_manifest(data)

    with pytest.raises(FinalValidationError) as excinfo:
        validate_final_outputs(
            manifest=manifest,
            prmtop=tmp_path / "missing.prmtop",
            inpcrd=tmp_path / "missing.inpcrd",
            pdb=tmp_path / "missing.pdb",
        )

    assert "Missing final prmtop" in str(excinfo.value)


def test_zero_size_final_file_fails(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    manifest = make_manifest(data)
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    pdb = tmp_path / "system.pdb"
    prmtop.write_text("", encoding="utf-8")
    inpcrd.write_text("x", encoding="utf-8")
    pdb.write_text("END\n", encoding="utf-8")

    with pytest.raises(FinalValidationError) as excinfo:
        validate_final_outputs(manifest=manifest, prmtop=prmtop, inpcrd=inpcrd, pdb=pdb)

    assert "empty" in str(excinfo.value)


def test_configured_solvation_ions_are_allowed_in_final_pdb(tmp_path, monkeypatch):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["validation"]["run_openmm_energy_check"] = False
    data["ligands"] = [
        {
            "id": "SUB_501",
            "selector": {"chain": "B", "resname": "SUB", "resid": 501, "icode": None},
            "net_charge": 0,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "am1bcc",
            "user_mol2": None,
            "qmmesp": None,
        }
    ]
    manifest = make_manifest(data)
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    pdb = tmp_path / "system.pdb"
    prmtop.write_text("placeholder\n", encoding="utf-8")
    inpcrd.write_text("placeholder\n", encoding="utf-8")
    _write_final_pdb(
        pdb,
        [
            _atom(1, "N", "ALA", "A", 1, "N", record_name="ATOM"),
            _atom(2, "C1", "SUB", "B", 501, "C"),
            _atom(3, "O", "WAT", "", 1, "O"),
            _atom(4, "Na+", "Na+", "", 305, "Na"),
            _atom(5, "Cl-", "Cl-", "", 306, "Cl"),
        ],
    )
    monkeypatch.setattr(
        "mdprep.validation.topology.run_parmed_check",
        lambda _prmtop, _inpcrd: {"available": False, "status": "skipped", "warning": "test"},
    )

    report = validate_final_outputs(manifest=manifest, prmtop=prmtop, inpcrd=inpcrd, pdb=pdb)

    assert report["errors"] == []


def test_unconfigured_final_heterogen_still_fails(tmp_path, monkeypatch):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["validation"]["run_openmm_energy_check"] = False
    manifest = make_manifest(data)
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    pdb = tmp_path / "system.pdb"
    prmtop.write_text("placeholder\n", encoding="utf-8")
    inpcrd.write_text("placeholder\n", encoding="utf-8")
    _write_final_pdb(
        pdb,
        [
            _atom(1, "N", "ALA", "A", 1, "N", record_name="ATOM"),
            _atom(2, "C1", "UNK", "B", 501, "C"),
            _atom(3, "O", "WAT", "", 1, "O"),
        ],
    )
    monkeypatch.setattr(
        "mdprep.validation.topology.run_parmed_check",
        lambda _prmtop, _inpcrd: {"available": False, "status": "skipped", "warning": "test"},
    )

    with pytest.raises(FinalValidationError) as excinfo:
        validate_final_outputs(manifest=manifest, prmtop=prmtop, inpcrd=inpcrd, pdb=pdb)

    assert "Unexpected heterogen residues" in str(excinfo.value)


def _atom(
    serial: int,
    name: str,
    resname: str,
    chain_id: str,
    resid: int,
    element: str,
    *,
    record_name: str = "HETATM",
) -> AtomRecord:
    return AtomRecord(
        serial=serial,
        name=name,
        altloc=None,
        resname=resname,
        chain_id=chain_id,
        resid=resid,
        icode=None,
        x=float(serial),
        y=0.0,
        z=0.0,
        occupancy=1.0,
        bfactor=20.0,
        element=element,
        record_name=record_name,  # type: ignore[arg-type]
        original_line="",
    )


def _write_final_pdb(path, atoms: list[AtomRecord]) -> None:
    residues: list[ResidueRecord] = []
    grouped: dict[tuple[str, str, int, str | None], list[AtomRecord]] = {}
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    for index, ((chain_id, resname, resid, icode), residue_atoms) in enumerate(grouped.items()):
        residues.append(
            ResidueRecord(
                id=ResidueId(chain_id=chain_id, resname=resname, resid=resid, icode=icode),
                atoms=residue_atoms,
                record_names={atom.record_name for atom in residue_atoms},
                original_index=index,
            )
        )
    write_pdb(PdbStructure(path=path, atoms=atoms, residues=residues, model_count=1), path)
