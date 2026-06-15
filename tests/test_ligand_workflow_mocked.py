from pathlib import Path

import pytest

from mdprep.ambertools.commands import AmberToolRun, AmberToolsError
from mdprep.ambertools.mol2 import read_mol2, write_mol2_with_charges
from mdprep.external.runner import CommandResult
from mdprep.leap.log_parser import parse_tleap_log_text
from mdprep.leap.runner import TLeapRun
from mdprep.ligands.pyscf_charges import LigandPySCFChargeResult
from mdprep.ligands.workflow import LigandWorkflowError, run_ligand_stage
from mdprep.protonation.apply import apply_protonation_stage
from mdprep.qm.point_charges import PointCharge, PointChargeSelection
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def command_run(output_path: Path, name: str) -> AmberToolRun:
    return AmberToolRun(
        command_result=CommandResult(
            command=(name, "fake"),
            cwd=str(output_path.parent),
            returncode=0,
            stdout="",
            stderr="",
            runtime_seconds=0.01,
        ),
        stdout_path=output_path.parent / f"{name}_stdout.txt",
        stderr_path=output_path.parent / f"{name}_stderr.txt",
        output_path=output_path,
    )


def manifest_with_ligand(entry: dict):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["structure"]["remove_unknown_heterogens"] = True
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [entry]
    return make_manifest(data)


def normalized_structure(manifest):
    return normalize_structure_stage(manifest).normalized_structure


def protonation_result(manifest, tmp_path):
    normalized = normalize_structure_stage(manifest)
    return apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "00.pdb",
        output_protonation_pdb_path=tmp_path / "01.pdb",
    )


def qmmesp_block():
    return {
        "qm_engine": "pyscf",
        "method": "HF",
        "basis": "STO-3G",
        "embedding_cutoff_angstrom": 12.0,
        "grid": {
            "type": "connolly",
            "vdw_scale_factors": [1.4],
            "points_per_atom_per_shell": 8,
            "exclude_inside_vdw_scale": 1.2,
            "max_points": 100,
        },
        "resp_fitting": {"backend": "native", "restraint": "none", "stage_2": False},
        "environment": {
            "include_protein": True,
            "include_waters": True,
            "include_other_ligands": True,
            "exclude_self_ligand": True,
        },
    }


def test_am1bcc_workflow_writes_parameter_files_with_fake_tools(monkeypatch, tmp_path):
    entry = ligand_entry("sub_501", "B", "SUB", 501)
    manifest = manifest_with_ligand(entry)

    def fake_antechamber(*, ligand, input_pdb, output_mol2, residue_name, work_dir):
        output = Path(output_mol2)
        output.write_text(Path("tests/data/ligands/ligand_sub.good.mol2").read_text(encoding="utf-8"), encoding="utf-8")
        return command_run(output, "antechamber")

    def fake_parmchk2(*, ligand, input_mol2, output_frcmod, work_dir):
        output = Path(output_frcmod)
        output.write_text(Path("tests/data/ligands/ligand_sub.frcmod").read_text(encoding="utf-8"), encoding="utf-8")
        return command_run(output, "parmchk2")

    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    item = result.ligands[0]
    assert item.final_mol2_path and item.final_mol2_path.exists()
    assert item.final_frcmod_path and item.final_frcmod_path.exists()
    assert item.validation and item.validation.validation_json_path.exists()
    assert item.validation.charges_csv_path.exists()


def test_user_mol2_workflow_copies_and_validates_mol2(tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    manifest = manifest_with_ligand(entry)

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    item = result.ligands[0]
    assert item.final_mol2_path and item.final_mol2_path.exists()
    assert item.final_frcmod_path and item.final_frcmod_path.exists()
    assert item.parmchk2 is None


def test_user_frcmod_is_copied_and_parmchk2_is_skipped(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
        "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr(
        "mdprep.ligands.workflow.run_parmchk2",
        lambda **kwargs: (_ for _ in ()).throw(AmberToolsError("should not run")),
    )

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert result.ligands[0].final_frcmod_path is not None


def test_user_mol2_without_user_frcmod_fails_if_parmchk2_unavailable(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "user_mol2",
        "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr(
        "mdprep.ligands.workflow.run_parmchk2",
        lambda **kwargs: (_ for _ in ()).throw(AmberToolsError("AmberTools executable not found: parmchk2")),
    )

    with pytest.raises(LigandWorkflowError) as excinfo:
        run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert "parmchk2" in str(excinfo.value)


def fake_antechamber(*, ligand, input_pdb, output_mol2, residue_name, work_dir):
    output = Path(output_mol2)
    output.write_text(Path("tests/data/ligands/ligand_sub.good.mol2").read_text(encoding="utf-8"), encoding="utf-8")
    return command_run(output, "antechamber")


def fake_parmchk2(*, ligand, input_mol2, output_frcmod, work_dir):
    output = Path(output_frcmod)
    output.write_text(Path("tests/data/ligands/ligand_sub.frcmod").read_text(encoding="utf-8"), encoding="utf-8")
    return command_run(output, "parmchk2")


def fake_pyscf_derivation(*, extracted, provisional_mol2_path, output_mol2_path, output_dir, method_name, point_charges=None):
    output = Path(output_mol2_path)
    write_mol2_with_charges(provisional_mol2_path, [0.25, -0.25], output)
    qm_dir = Path(output_dir) / "ligands" / extracted.config.id / "qm" / method_name
    qm_dir.mkdir(parents=True, exist_ok=True)
    charges_csv = qm_dir / "fitted_charges.csv"
    charges_csv.write_text("atom_index,atom_name,charge\n1,C1,0.25\n2,O1,-0.25\n", encoding="utf-8")
    fit_report = qm_dir / "fit_report.json"
    fit_report.write_text('{"charge_sum_final": 0.0, "rms_error": 0.0}\n', encoding="utf-8")
    embedding_summary = None if point_charges is None else point_charges.to_dict()
    return LigandPySCFChargeResult(
        method=method_name,
        qm_dir=qm_dir,
        charged_mol2_path=output,
        fitted_charges_csv_path=charges_csv,
        fit_report_path=fit_report,
        pyscf_result={"method": "HF", "basis": "STO-3G", "converged": True, "energy_hartree": -1.0},
        fit_result={"charge_sum_final": 0.0, "rms_error": 0.0},
        grid_point_count=12,
        embedding_summary=embedding_summary,
        warnings=[],
    )


def test_gas_resp_pyscf_replaces_provisional_charges(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "gas_resp_pyscf",
        "qmmesp": qmmesp_block(),
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)
    monkeypatch.setattr("mdprep.ligands.workflow.derive_pyscf_charges", fake_pyscf_derivation)

    result = run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    item = result.ligands[0]
    assert item.qm is not None
    final = read_mol2(item.final_mol2_path)
    assert [atom.charge for atom in final.atoms] == pytest.approx([0.25, -0.25])
    assert "replaced by PySCF-fitted gas-phase charges" in " ".join(item.warnings)


def fake_tleap(input_path, *, work_dir, executable="tleap"):
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


def fake_point_charges(**kwargs):
    return PointChargeSelection(
        target_atom_indices=[3, 4],
        point_charges=[
            PointCharge(
                x=0.0,
                y=0.0,
                z=0.0,
                charge=0.1,
                residue_name="ALA",
                residue_number=1,
                atom_name="CA",
                category="protein",
            )
        ],
        total_before_cutoff=1,
        total_after_cutoff=1,
        net_embedding_charge=0.1,
        min_distance=5.0,
        max_distance=5.0,
        categories={"protein": 1},
    )


def test_qmmesp_pyscf_uses_embedding_and_replaces_provisional_charges(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "qmmesp_pyscf",
        "qmmesp": qmmesp_block(),
    }
    manifest = manifest_with_ligand(entry)
    protonation = protonation_result(manifest, tmp_path)
    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)
    monkeypatch.setattr("mdprep.ligands.workflow.run_tleap", fake_tleap)
    monkeypatch.setattr("mdprep.ligands.workflow.extract_point_charges_from_prmtop", fake_point_charges)
    monkeypatch.setattr("mdprep.ligands.workflow.derive_pyscf_charges", fake_pyscf_derivation)

    result = run_ligand_stage(
        protonation.structure,
        manifest,
        output_dir=tmp_path,
        protonation_result=protonation,
    )

    item = result.ligands[0]
    assert item.qm is not None
    assert item.qm.embedding_summary["point_charge_count_after_cutoff"] == 1
    final = read_mol2(item.final_mol2_path)
    assert [atom.charge for atom in final.atoms] == pytest.approx([0.25, -0.25])


def test_qmmesp_requires_protonation_result(monkeypatch, tmp_path):
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "qmmesp_pyscf",
        "qmmesp": qmmesp_block(),
    }
    manifest = manifest_with_ligand(entry)
    monkeypatch.setattr("mdprep.ligands.workflow.run_antechamber", fake_antechamber)
    monkeypatch.setattr("mdprep.ligands.workflow.run_parmchk2", fake_parmchk2)

    with pytest.raises(LigandWorkflowError) as excinfo:
        run_ligand_stage(normalized_structure(manifest), manifest, output_dir=tmp_path)

    assert "requires a protonation result" in str(excinfo.value)
