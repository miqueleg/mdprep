import json
from types import SimpleNamespace

import yaml
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.protonation.apply import apply_protonation_stage as real_apply_protonation_stage
from mdprep.refinement.workflow import _remove_stale_refinement_failure
from mdprep.structure.writer import write_pdb
from mdprep.workflows.prepare import _external_executables_from_metal_report
from tests.test_structure_normalize import ligand_entry, manifest_data


def _write_manifest(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    output_dir = tmp_path / "prepared"
    data["project"]["output_dir"] = str(output_dir)
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [ligand_entry("substrate", "B", "SUB", 501)]
    data["refinement"] = {
        "enabled": True,
        "qm_components": {"ligands": ["substrate"]},
        "ash": {"python_executable": "/fake/ash/python"},
    }
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path, output_dir


def test_prepare_refinement_runs_protonation_before_and_after_geometry_change(
    tmp_path, monkeypatch
):
    manifest_path, output_dir = _write_manifest(tmp_path)
    protonation_calls = []

    def tracked_protonation(*args, **kwargs):
        protonation_calls.append(kwargs["output_protonation_pdb_path"])
        return real_apply_protonation_stage(*args, **kwargs)

    def fake_refinement_stage(
        *, protonated_structure, reference_structure, manifest, output_dir, protonation_result
    ):
        refined_path = output_dir / "intermediate" / "02_qmmm_refined_hydrogenated.pdb"
        refined = type(reference_structure)(
            path=refined_path,
            atoms=reference_structure.atoms,
            residues=reference_structure.residues,
            model_count=reference_structure.model_count,
            used_model=reference_structure.used_model,
            warnings=reference_structure.warnings,
            conect_bonds=reference_structure.conect_bonds,
            ter_after_serials=reference_structure.ter_after_serials,
        )
        write_pdb(refined, refined_path)
        return SimpleNamespace(structure=refined, refined_hydrogenated_pdb_path=refined_path)

    def fake_reports(result, *, json_path, markdown_path):
        report = {
            "status": "complete",
            "ash": {
                "python_executable": "/fake/ash/python",
                "xtb_executable": "/fake/bin/xtb",
            },
            "provisional_ligands": [],
            "provisional_system": None,
        }
        json_path.write_text(json.dumps(report), encoding="utf-8")
        markdown_path.write_text("# refinement\n", encoding="utf-8")
        return report

    monkeypatch.setattr("mdprep.workflows.prepare.apply_protonation_stage", tracked_protonation)
    monkeypatch.setattr("mdprep.workflows.prepare.run_refinement_stage", fake_refinement_stage)
    monkeypatch.setattr("mdprep.workflows.prepare.write_refinement_reports", fake_reports)
    monkeypatch.setattr("mdprep.workflows.prepare._write_versions", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        app,
        ["prepare", str(manifest_path), "--stop-after", "refinement"],
    )

    assert result.exit_code == 0, result.output
    assert [path.name for path in protonation_calls] == [
        "01_protonation_assigned.pdb",
        "03_final_protonation_assigned.pdb",
    ]
    assert (output_dir / "reports" / "protonation_pre_refinement_report.json").exists()
    assert (output_dir / "reports" / "refinement_report.json").exists()
    assert (output_dir / "reports" / "protonation_report.json").exists()
    lock = yaml.safe_load((output_dir / "manifest.lock.yaml").read_text(encoding="utf-8"))
    assert lock["resolved"]["refinement"]["status"] == "complete"


def test_stop_after_refinement_requires_enabled_manifest(tmp_path):
    data = manifest_data("tests/data/protein_with_waters.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    path = tmp_path / "system.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["prepare", str(path), "--stop-after", "refinement"],
    )

    assert result.exit_code != 0
    assert "requires refinement.enabled: true" in result.output


def test_refinement_run_removes_stale_failure_report(tmp_path):
    failure_path = tmp_path / "reports" / "refinement_failure.json"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text('{"status": "failed"}\n', encoding="utf-8")

    _remove_stale_refinement_failure(tmp_path)

    assert not failure_path.exists()


def test_metal_versions_include_xtb_hessian_executable():
    report = {
        "bonded_mcpb_site": {
            "runs": [],
            "pyscf": {
                "small_model_hessian_backend": "gfn2_xtb",
                "small_model": {
                    "external_commands": [
                        {
                            "command": ["/opt/xtb", "--version"],
                            "cwd": "/tmp/work",
                            "returncode": 0,
                            "stdout": "",
                            "stderr": "",
                            "runtime_seconds": 0.1,
                        }
                    ]
                },
            },
        }
    }

    assert _external_executables_from_metal_report(report) == {
        "gfn2_xtb": "/opt/xtb"
    }
