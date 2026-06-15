from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mdprep.ambertools.mol2 import read_mol2, write_mol2_with_charges
from mdprep.external.runner import CommandResult
from mdprep.leap.log_parser import parse_tleap_log_text
from mdprep.leap.runner import TLeapRun
from mdprep.ligands.pyscf_charges import QMMESP_CONFIRMATION, LigandPySCFChargeResult
from mdprep.ligands.workflow import run_ligand_stage
from mdprep.protonation.apply import apply_protonation_stage
from mdprep.qm.point_charges import PointCharge, PointChargeSelection, extract_point_charges_from_prmtop
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_ligand_workflow_mocked import command_run, qmmesp_block
from tests.test_point_charges import FakeAtom, FakeResidue, install_fake_parmed
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def qmmesp_manifest(tmp_path, *, include_other_ligands=True, include_protein=True, include_waters=True):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["structure"]["remove_unknown_heterogens"] = True
    data["protonation"]["method"] = "manual_only"
    block = qmmesp_block()
    block["environment"] = {
        "include_protein": include_protein,
        "include_waters": include_waters,
        "include_other_ligands": include_other_ligands,
        "exclude_self_ligand": True,
    }
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": block,
        },
        {
            **ligand_entry("cof_601", "C", "COF", 601),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": block,
        },
    ]
    return make_manifest(data)


def nearby_fake_parmed_structure():
    protein = FakeResidue("ALA", 1)
    water = FakeResidue("WAT", 2)
    target = FakeResidue("SUB", 501)
    other = FakeResidue("COF", 601)
    atoms = []
    coords = []
    for residue, atom_name, charge, coord in [
        (protein, "CA", 0.2, [5.2, 5.0, 5.0]),
        (water, "O", -0.8, [5.0, 5.3, 5.0]),
        (target, "C1", 0.123, [5.0, 5.0, 5.0]),
        (target, "O1", -0.123, [6.0, 5.0, 5.0]),
        (other, "N1", -0.2, [5.5, 5.5, 5.0]),
    ]:
        atom = FakeAtom(len(atoms), atom_name, charge, residue)
        residue.atoms.append(atom)
        atoms.append(atom)
        coords.append(coord)
    return SimpleNamespace(
        residues=[protein, water, target, other],
        atoms=atoms,
        coordinates=np.asarray(coords, dtype=float),
    )


def single_ligand_config(**environment):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    block = qmmesp_block()
    block["embedding_cutoff_angstrom"] = 3.0
    block["environment"].update(environment)
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": block,
        }
    ]
    manifest = make_manifest(data)
    return manifest, manifest.ligands[0]


def test_target_ligand_is_excluded_from_embedding(monkeypatch):
    manifest, ligand = single_ligand_config()
    install_fake_parmed(monkeypatch, nearby_fake_parmed_structure())

    selection = extract_point_charges_from_prmtop(
        prmtop="fake.prmtop",
        inpcrd="fake.inpcrd",
        ligand=ligand,
        manifest=manifest,
        target_coordinates=np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float),
    )

    assert selection.target_atom_indices == [2, 3]
    assert selection.to_dict()["target_ligand_excluded_from_embedding"] is True
    assert {(charge.residue_name, charge.atom_name) for charge in selection.point_charges} == {
        ("ALA", "CA"),
        ("WAT", "O"),
        ("COF", "N1"),
    }
    assert selection.categories == {"protein": 1, "water": 1, "ligand": 1}


@pytest.mark.parametrize(
    ("environment", "expected_categories"),
    [
        ({"include_other_ligands": False}, {"protein": 1, "water": 1}),
        ({"include_waters": False}, {"protein": 1, "ligand": 1}),
        ({"include_protein": False}, {"water": 1, "ligand": 1}),
    ],
)
def test_environment_category_switches_do_not_reintroduce_target(monkeypatch, environment, expected_categories):
    manifest, ligand = single_ligand_config(**environment)
    install_fake_parmed(monkeypatch, nearby_fake_parmed_structure())

    selection = extract_point_charges_from_prmtop(
        prmtop="fake.prmtop",
        inpcrd="fake.inpcrd",
        ligand=ligand,
        manifest=manifest,
        target_coordinates=np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float),
    )

    assert selection.categories == expected_categories
    assert all(charge.residue_name != "SUB" for charge in selection.point_charges)


def fake_qmmesp_antechamber(*, ligand, input_pdb, output_mol2, residue_name, work_dir):
    source = Path("tests/data/ligands/ligand_sub.good.mol2")
    charges = [0.123, -0.123]
    if residue_name == "COF":
        source = Path("tests/data/ligands/ligand_cof.good.mol2")
        charges = [0.222, -0.222]
    output = Path(output_mol2)
    write_mol2_with_charges(source, charges, output)
    return command_run(output, "antechamber")


def fake_qmmesp_parmchk2(*, ligand, input_mol2, output_frcmod, work_dir):
    source = Path("tests/data/ligands/ligand_sub.frcmod")
    if ligand.selector.resname == "COF":
        source = Path("tests/data/ligands/ligand_cof.frcmod")
    output = Path(output_frcmod)
    output.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return command_run(output, "parmchk2")


def fake_qmmesp_tleap(input_path, *, work_dir, executable="tleap"):
    work = Path(work_dir)
    for suffix in ["prmtop", "inpcrd", "pdb"]:
        (work / f"provisional.{suffix}").write_text(f"{suffix}\n", encoding="utf-8")
    log_text = "Checking 'system'....\nUnit is OK.\nTotal unperturbed charge:   0.000000\n"
    (work / "tleap.log").write_text(log_text, encoding="utf-8")
    return TLeapRun(
        command_result=CommandResult(("tleap", "-f", Path(input_path).name), str(work), 0, "", "", 0.01),
        input_path=Path(input_path),
        log_path=work / "tleap.log",
        summary=parse_tleap_log_text(log_text, returncode=0),
    )


def test_multiple_qmmesp_ligands_are_fitted_one_target_at_a_time(monkeypatch, tmp_path):
    manifest = qmmesp_manifest(tmp_path)
    normalized = normalize_structure_stage(manifest)
    protonation = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "00.pdb",
        output_protonation_pdb_path=tmp_path / "01.pdb",
    )
    extraction_calls = []
    derivation_calls = []

    def fake_extract_point_charges_from_prmtop(**kwargs):
        ligand = kwargs["ligand"]
        extraction_calls.append(ligand.id)
        other_resname = "COF" if ligand.selector.resname == "SUB" else "SUB"
        return PointChargeSelection(
            target_atom_indices=[10, 11] if ligand.id == "sub_501" else [12, 13],
            point_charges=[
                PointCharge(
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    charge=0.05,
                    residue_name=other_resname,
                    residue_number=999,
                    atom_name="X1",
                    category="ligand",
                )
            ],
            total_before_cutoff=1,
            total_after_cutoff=1,
            net_embedding_charge=0.05,
            min_distance=2.0,
            max_distance=2.0,
            categories={"ligand": 1},
        )

    def fake_derive_pyscf_charges(*, extracted, provisional_mol2_path, output_mol2_path, output_dir, method_name, point_charges):
        derivation_calls.append((extracted.config.id, tuple(point_charges.target_atom_indices)))
        charges = [0.25, -0.25] if extracted.config.id == "sub_501" else [0.35, -0.35]
        output = Path(output_mol2_path)
        write_mol2_with_charges(provisional_mol2_path, charges, output)
        qm_dir = Path(output_dir) / "ligands" / extracted.config.id / "qm" / method_name
        qm_dir.mkdir(parents=True, exist_ok=True)
        charges_csv = qm_dir / "fitted_charges.csv"
        charges_csv.write_text("atom_index,atom_name,charge\n1,A1,0.25\n2,A2,-0.25\n", encoding="utf-8")
        fit_report = qm_dir / "fit_report.json"
        fit_report.write_text('{"charge_sum_final": 0.0}\n', encoding="utf-8")
        return LigandPySCFChargeResult(
            method=method_name,
            qm_dir=qm_dir,
            charged_mol2_path=output,
            fitted_charges_csv_path=charges_csv,
            fit_report_path=fit_report,
            pyscf_result={"converged": True, "energy_hartree": -1.0},
            fit_result={"charge_sum_final": 0.0, "confirmation": QMMESP_CONFIRMATION},
            grid_point_count=12,
            embedding_summary=point_charges.to_dict(),
            warnings=[],
        )

    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_qmmesp_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_qmmesp_parmchk2)
    monkeypatch.setattr("mdprep.ligands.workflow.run_tleap", fake_qmmesp_tleap)
    monkeypatch.setattr("mdprep.ligands.workflow.extract_point_charges_from_prmtop", fake_extract_point_charges_from_prmtop)
    monkeypatch.setattr("mdprep.ligands.workflow.derive_pyscf_charges", fake_derive_pyscf_charges)

    result = run_ligand_stage(
        protonation.structure,
        manifest,
        output_dir=manifest.project.output_dir,
        protonation_result=protonation,
    )

    assert extraction_calls == ["sub_501", "cof_601"]
    assert derivation_calls == [("sub_501", (10, 11)), ("cof_601", (12, 13))]
    assert [item.ligand_id for item in result.ligands] == ["sub_501", "cof_601"]
    assert result.ligands[0].qm.embedding_summary["categories"] == {"ligand": 1}
    assert result.ligands[1].qm.embedding_summary["categories"] == {"ligand": 1}

    sub_final = read_mol2(result.ligands[0].final_mol2_path)
    cof_final = read_mol2(result.ligands[1].final_mol2_path)
    assert [atom.charge for atom in sub_final.atoms] == pytest.approx([0.25, -0.25])
    assert [atom.charge for atom in cof_final.atoms] == pytest.approx([0.35, -0.35])
