import json
from math import acos, degrees, dist
from pathlib import Path

import pytest

from mdprep.external.runner import CommandResult
from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.protonation.histidine_geometry import (
    HistidineGeometryError,
    build_tautomer_cluster_model,
    build_water_hydrogen_coordinates,
    count_water_hydrogens,
    has_water_oxygen,
)
from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.protonation.report import write_protonation_reports
from mdprep.protonation.xtb_runner import XtbRunResult
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb
from tests.test_protonation_propka_workflow import fake_propka_result
from tests.test_structure_normalize import make_manifest, manifest_data


WATER_PDB = Path("tests/data/protein_histidine_water_oxygen.pdb")


def _histidine_and_water(path: Path | str = WATER_PDB):
    structure = read_pdb(path)
    histidine = next(residue for residue in structure.residues if residue.id.resname == "HIS")
    water = next(residue for residue in structure.residues if residue.id.resname == "HOH")
    return structure, histidine, water


def _angle_degrees(a, center, b) -> float:
    vector_a = (a.x - center.x, a.y - center.y, a.z - center.z)
    vector_b = (b.x - center.x, b.y - center.y, b.z - center.z)
    dot = sum(left * right for left, right in zip(vector_a, vector_b))
    norm_a = sum(value * value for value in vector_a) ** 0.5
    norm_b = sum(value * value for value in vector_b) ** 0.5
    return degrees(acos(dot / (norm_a * norm_b)))


def _with_water_atoms(tmp_path: Path, atom_lines: list[str]) -> Path:
    source = Path("tests/data/protein_histidine_ring_hydrogenated.pdb").read_text(encoding="utf-8")
    lines = [line for line in source.splitlines() if line != "END"]
    path = tmp_path / "histidine_water.pdb"
    path.write_text("\n".join([*lines, *atom_lines, "END", ""]) , encoding="utf-8")
    return path


def _run_with_fakes(monkeypatch, tmp_path, data: dict):
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )

    def fake_run_xtb(*, config, xyz_path, work_dir, cluster_charge, stdout_path, stderr_path, input_path=None):
        energy = -40.01 if Path(xyz_path).name == "HID.xyz" else -40.00
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


def test_water_with_only_oxygen_gets_two_temporary_hydrogens():
    _, histidine, water = _histidine_and_water()
    model = build_tautomer_cluster_model([histidine, water], histidine, tautomer="HID")
    oxygen = next(atom for atom in water.atoms if atom.name == "O")
    temporary = [atom for atom in model.atoms if atom.source == "temporary_water_hydrogen"]

    assert has_water_oxygen(water)
    assert count_water_hydrogens(water) == 0
    assert len(build_water_hydrogen_coordinates(water, [histidine, water])) == 2
    assert len(temporary) == 2
    assert model.to_dict()["temporary_water_hydrogens_added"] == 2
    assert dist((oxygen.x, oxygen.y, oxygen.z), (temporary[0].x, temporary[0].y, temporary[0].z)) == pytest.approx(0.9572)
    assert dist((oxygen.x, oxygen.y, oxygen.z), (temporary[1].x, temporary[1].y, temporary[1].z)) == pytest.approx(0.9572)
    assert _angle_degrees(temporary[0], oxygen, temporary[1]) == pytest.approx(104.52)


def test_water_with_one_hydrogen_gets_one_temporary_hydrogen(tmp_path):
    path = _with_water_atoms(
        tmp_path,
        [
            "HETATM   17  O   HOH A 640       2.300   5.900   3.000  1.00 20.00           O",
            "HETATM   18  H1  HOH A 640       3.257   5.900   3.000  1.00 20.00           H",
        ],
    )
    _, histidine, water = _histidine_and_water(path)
    model = build_tautomer_cluster_model([histidine, water], histidine, tautomer="HID")
    oxygen = next(atom for atom in water.atoms if atom.name == "O")
    existing = next(atom for atom in model.atoms if atom.name == "H1" and atom.source == "input")
    temporary = [atom for atom in model.atoms if atom.source == "temporary_water_hydrogen"]

    assert len(temporary) == 1
    assert (existing.x, existing.y, existing.z) == pytest.approx((3.257, 5.900, 3.000))
    assert dist((oxygen.x, oxygen.y, oxygen.z), (temporary[0].x, temporary[0].y, temporary[0].z)) == pytest.approx(0.9572)
    assert _angle_degrees(existing, oxygen, temporary[0]) == pytest.approx(104.52, abs=0.1)


def test_water_with_two_hydrogens_is_unchanged(tmp_path):
    path = _with_water_atoms(
        tmp_path,
        [
            "HETATM   17  O   HOH A 640       2.300   5.900   3.000  1.00 20.00           O",
            "HETATM   18  H1  HOH A 640       3.257   5.900   3.000  1.00 20.00           H",
            "HETATM   19  H2  HOH A 640       2.060   6.827   3.000  1.00 20.00           H",
        ],
    )
    _, histidine, water = _histidine_and_water(path)
    model = build_tautomer_cluster_model([histidine, water], histidine, tautomer="HID")

    assert [atom for atom in model.atoms if atom.source == "temporary_water_hydrogen"] == []
    assert model.temporary_water_hydrogens == []


def test_water_with_no_oxygen_fails_clearly(tmp_path):
    path = _with_water_atoms(
        tmp_path,
        ["HETATM   17  H1  HOH A 640       3.257   5.900   3.000  1.00 20.00           H"],
    )
    _, histidine, water = _histidine_and_water(path)

    with pytest.raises(HistidineGeometryError, match="has no oxygen atom"):
        build_tautomer_cluster_model([histidine, water], histidine, tautomer="HID")


def test_cluster_with_crystal_water_oxygen_no_longer_fails_by_default(monkeypatch, tmp_path):
    data = manifest_data(str(WATER_PDB))
    data["protonation"]["method"] = "propka_xtb_his"
    result = _run_with_fakes(monkeypatch, tmp_path, data)

    assert result.xtb_selections[0].temporary_water_hydrogens_added == 2
    hid_xyz = tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "HID.xyz"
    assert hid_xyz.exists()
    xyz_text = hid_xyz.read_text(encoding="utf-8")
    assert "H " in xyz_text
    cluster_model = json.loads((hid_xyz.parent / "cluster_model.json").read_text(encoding="utf-8"))
    assert cluster_model["temporary_water_hydrogens_for_xtb_only"]["hydrogens_added"] == 2


def test_strict_water_hydrogen_mode_preserves_old_failure(monkeypatch, tmp_path):
    data = manifest_data(str(WATER_PDB))
    data["protonation"]["method"] = "propka_xtb_his"
    data["protonation"]["histidine"]["xtb"]["add_missing_water_hydrogens"] = False

    with pytest.raises(ProtonationApplicationError, match="lacks hydrogens"):
        _run_with_fakes(monkeypatch, tmp_path, data)


def test_temporary_water_hydrogens_are_not_written_to_protonation_pdb(monkeypatch, tmp_path):
    data = manifest_data(str(WATER_PDB))
    data["protonation"]["method"] = "propka_xtb_his"
    result = _run_with_fakes(monkeypatch, tmp_path, data)
    output_pdb = tmp_path / "01_protonation_assigned.pdb"
    write_pdb(result.structure, output_pdb)
    text = output_pdb.read_text(encoding="utf-8")

    assert "HOH A 640" in text
    assert " H1  HOH A 640" not in text
    assert " H2  HOH A 640" not in text


def test_protonation_report_records_temporary_water_hydrogens(monkeypatch, tmp_path):
    data = manifest_data(str(WATER_PDB))
    data["protonation"]["method"] = "propka_xtb_his"
    result = _run_with_fakes(monkeypatch, tmp_path, data)
    report = write_protonation_reports(
        result,
        json_path=tmp_path / "protonation_report.json",
        csv_path=tmp_path / "protonation_report.csv",
        markdown_path=tmp_path / "protonation_report.md",
    )

    clusters = report["temporary_water_hydrogens_for_xtb_clusters"]
    assert clusters[0]["histidine"] == "A:HIS2"
    assert clusters[0]["temporary_water_hydrogens_added"] == 2
    assert clusters[0]["waters_modified_for_xtb_only"][0]["resid"] == 640
    assert clusters[0]["final_pdb_modified"] is False
    markdown = (tmp_path / "protonation_report.md").read_text(encoding="utf-8")
    assert "Temporary Water Hydrogens For xTB Clusters" in markdown
    assert "A:HOH640" in markdown
