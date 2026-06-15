from pathlib import Path

import pytest

from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.protonation.propka import PropkaWorkflowResult
from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_manual_protonation import override
from tests.test_structure_normalize import make_manifest, manifest_data


def fake_propka_result(tmp_path: Path, records: list[PropkaRecord]) -> PropkaWorkflowResult:
    return PropkaWorkflowResult(
        executable="propka3",
        command=("propka3", "propka_input.pdb"),
        cwd=str(tmp_path),
        returncode=0,
        runtime_seconds=0.01,
        stdout_path=tmp_path / "propka_stdout.txt",
        stderr_path=tmp_path / "propka_stderr.txt",
        output_pka_path=tmp_path / "propka_output.pka",
        parsed_pkas_path=tmp_path / "parsed_pkas.csv",
        records=records,
    )


def run_with_fake_propka(monkeypatch, tmp_path, data: dict, records: list[PropkaRecord]):
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(tmp_path, records),
    )
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    return apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
    )


def test_method_propka_assigns_supported_residues_from_fixture_pkas(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka"

    result = run_with_fake_propka(
        monkeypatch,
        tmp_path,
        data,
        [
            PropkaRecord("ASP", 3, "A", 8.0, "ASP 3 A 8.0"),
            PropkaRecord("HIS", 2, "A", 8.0, "HIS 2 A 8.0"),
        ],
    )

    resnames = [residue.id.resname for residue in result.structure.residues]
    assert "ASH" in resnames
    assert "HIP" in resnames
    assert [record.pka for record in result.propka_assignments_applied] == [8.0, 8.0]


def test_method_propka_fails_for_neutral_unresolved_his(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka"

    with pytest.raises(ProtonationApplicationError) as excinfo:
        run_with_fake_propka(
            monkeypatch,
            tmp_path,
            data,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        )

    assert "requires HID/HIE assignment" in str(excinfo.value)


def test_method_propka_respects_manual_his_override(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka"
    data["protonation"]["overrides"] = [override("A", "HIS", 2, "HIE")]

    result = run_with_fake_propka(
        monkeypatch,
        tmp_path,
        data,
        [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
    )

    assert "HIE" in [residue.id.resname for residue in result.structure.residues]
    assert result.manual_overrides_applied[0].source == "manual_override"


def test_method_propka_preserves_input_hid_state(monkeypatch, tmp_path):
    source = Path("tests/data/protein_histidine_ring.pdb").read_text(encoding="utf-8")
    pdb = tmp_path / "input_hid.pdb"
    pdb.write_text(source.replace("HIS A   2", "HID A   2"), encoding="utf-8")
    data = manifest_data(str(pdb))
    data["protonation"]["method"] = "propka"

    result = run_with_fake_propka(
        monkeypatch,
        tmp_path,
        data,
        [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
    )

    assert "HID" in [residue.id.resname for residue in result.structure.residues]
    assert result.input_state_assignments_applied[0].source == "input_state"


def test_reports_contain_pka_values(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka"
    result = run_with_fake_propka(
        monkeypatch,
        tmp_path,
        data,
        [PropkaRecord("HIS", 2, "A", 8.0, "HIS 2 A 8.0")],
    )

    report = result.to_report_dict()

    assert report["parsed_pkas"][0]["pka"] == 8.0
    assert report["propka_assignments_applied"][0]["pka"] == 8.0
