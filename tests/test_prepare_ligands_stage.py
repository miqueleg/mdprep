import json

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.structure.pdb import read_pdb
from tests.test_structure_normalize import ligand_entry, manifest_data


def write_ligand_manifest(tmp_path, *, output_name="prepared"):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / output_name)
    data["structure"]["remove_unknown_heterogens"] = False
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "user_mol2",
            "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
        },
        {
            **ligand_entry("cof_601", "C", "COF", 601),
            "charge_method": "user_mol2",
            "user_mol2": "tests/data/ligands/ligand_cof.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_cof.frcmod",
        },
    ]
    manifest = tmp_path / "system.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return manifest, tmp_path / output_name


def test_prepare_ligands_stage_writes_expected_files(tmp_path):
    manifest, output_dir = write_ligand_manifest(tmp_path)
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "ligands"])

    assert result.exit_code == 0, result.output
    expected = [
        output_dir / "intermediate" / "00_input_normalized.pdb",
        output_dir / "intermediate" / "01_protonation_assigned.pdb",
        output_dir / "reports" / "ligand_report.json",
        output_dir / "reports" / "ligand_report.csv",
        output_dir / "reports" / "ligand_report.md",
        output_dir / "ligands" / "sub_501" / "input" / "sub_501.pdb",
        output_dir / "ligands" / "sub_501" / "parameters" / "sub_501.final.mol2",
        output_dir / "ligands" / "sub_501" / "parameters" / "sub_501.frcmod",
    ]
    for path in expected:
        assert path.exists(), path
    assigned = read_pdb(output_dir / "intermediate" / "01_protonation_assigned.pdb")
    assert assigned.residues


def test_ligand_report_contains_multiple_ligands(tmp_path):
    manifest, output_dir = write_ligand_manifest(tmp_path)
    result = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "ligands"])

    assert result.exit_code == 0
    report = json.loads((output_dir / "reports" / "ligand_report.json").read_text(encoding="utf-8"))
    assert [item["ligand_id"] for item in report["ligands"]] == ["sub_501", "cof_601"]


def test_full_prepare_without_stop_after_mentions_ligands_stage(tmp_path):
    manifest, _ = write_ligand_manifest(tmp_path)
    result = CliRunner().invoke(app, ["prepare", str(manifest)])

    assert result.exit_code != 0
    assert "--stop-after ligands" in result.output


def test_structure_and_protonation_stages_still_work(tmp_path):
    manifest, output_dir = write_ligand_manifest(tmp_path)

    structure = CliRunner().invoke(app, ["prepare", str(manifest), "--stop-after", "structure"])
    assert structure.exit_code == 0
    protonation = CliRunner().invoke(
        app,
        ["prepare", str(manifest), "--stop-after", "protonation", "--overwrite"],
    )
    assert protonation.exit_code == 0
    assert (output_dir / "intermediate" / "01_protonation_assigned.pdb").exists()
