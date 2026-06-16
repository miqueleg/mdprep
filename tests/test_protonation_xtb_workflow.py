from pathlib import Path

import pytest

from mdprep.external.runner import CommandResult
from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.protonation.xtb_runner import XtbExecutionError, XtbRunResult
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_protonation_propka_workflow import fake_propka_result
from tests.test_structure_normalize import make_manifest, manifest_data


def run_with_fakes(monkeypatch, tmp_path, data: dict, *, hid_energy: float, hie_energy: float):
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )

    def fake_run_xtb(*, config, xyz_path, work_dir, cluster_charge, stdout_path, stderr_path, input_path=None):
        energy = hid_energy if Path(xyz_path).name == "HID.xyz" else hie_energy
        Path(stdout_path).write_text(f":: total energy      {energy:.12f} Eh\n", encoding="utf-8")
        Path(stderr_path).write_text("", encoding="utf-8")
        return XtbRunResult(
            command_result=CommandResult(
                command=("xtb", Path(xyz_path).name),
                cwd=str(work_dir),
                returncode=0,
                stdout="",
                stderr="",
                runtime_seconds=0.01,
            ),
            stdout_path=Path(stdout_path),
            stderr_path=Path(stderr_path),
        )

    monkeypatch.setattr("mdprep.protonation.histidine_xtb.run_xtb", fake_run_xtb)
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    return apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
    )


def test_propka_xtb_his_runs_hid_hie_comparison_and_selects_lower_hid(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring_hydrogenated.pdb")
    data["protonation"]["method"] = "propka_xtb_his"

    result = run_with_fakes(monkeypatch, tmp_path, data, hid_energy=-40.01, hie_energy=-40.00)

    assert "HID" in [residue.id.resname for residue in result.structure.residues]
    assert result.xtb_selections[0].selected_state == "HID"
    assert (tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "HID.xyz").exists()
    assert (tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "HID_xtb.inp").exists()
    assert (tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "cluster_model.json").exists()


def test_propka_xtb_his_selects_lower_hie(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring_hydrogenated.pdb")
    data["protonation"]["method"] = "propka_xtb_his"

    result = run_with_fakes(monkeypatch, tmp_path, data, hid_energy=-40.00, hie_energy=-40.01)

    assert "HIE" in [residue.id.resname for residue in result.structure.residues]
    assert result.xtb_selections[0].selected_state == "HIE"


def test_xtb_close_call_is_reported(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring_hydrogenated.pdb")
    data["protonation"]["method"] = "propka_xtb_his"

    result = run_with_fakes(monkeypatch, tmp_path, data, hid_energy=-40.0001, hie_energy=-40.0)

    assert result.xtb_selections[0].close_call
    assert any("close-call" in warning for warning in result.warnings)


def test_missing_histidine_ring_atom_fails_clearly(monkeypatch, tmp_path):
    pdb = tmp_path / "missing_ring_atom.pdb"
    source = Path("tests/data/protein_histidine_ring_hydrogenated.pdb").read_text(encoding="utf-8")
    pdb.write_text(
        "\n".join(line for line in source.splitlines() if " ND1 " not in line) + "\n",
        encoding="utf-8",
    )
    data = manifest_data(str(pdb))
    data["protonation"]["method"] = "propka_xtb_his"
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )

    with pytest.raises(ProtonationApplicationError) as excinfo:
        manifest = make_manifest(data)
        normalized = normalize_structure_stage(manifest)
        apply_protonation_stage(
            normalized.normalized_structure,
            manifest,
            input_normalized_pdb_path=tmp_path / "normalized.pdb",
            output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
        )

    assert "missing required atoms" in str(excinfo.value)


def test_dehydrogenated_histidine_cluster_fails_clearly(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka_xtb_his"
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )

    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    with pytest.raises(ProtonationApplicationError) as excinfo:
        apply_protonation_stage(
            normalized.normalized_structure,
            manifest,
            input_normalized_pdb_path=tmp_path / "normalized.pdb",
            output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
        )

    assert "requires a hydrogenated protein model" in str(excinfo.value)


def test_xtb_unavailable_fails_only_when_neutral_his_needs_it(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring_hydrogenated.pdb")
    data["protonation"]["method"] = "propka_xtb_his"
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 8.0, "HIS 2 A 8.0")],
        ),
    )
    monkeypatch.setattr(
        "mdprep.protonation.histidine_xtb.run_xtb",
        lambda **kwargs: (_ for _ in ()).throw(XtbExecutionError("should not run")),
    )
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
    )
    assert "HIP" in [residue.id.resname for residue in result.structure.residues]

    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )
    with pytest.raises(ProtonationApplicationError) as excinfo:
        apply_protonation_stage(
            normalized.normalized_structure,
            manifest,
            input_normalized_pdb_path=tmp_path / "normalized.pdb",
            output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
        )

    assert "should not run" in str(excinfo.value)
