from pathlib import Path

import pytest

from mdprep.external.runner import CommandResult
from mdprep.leap.builder import run_tleap_stage
from mdprep.leap.log_parser import parse_tleap_log_text
from mdprep.leap.runner import TLeapRun
from mdprep.ligands.workflow import run_ligand_stage
from mdprep.protonation.apply import apply_protonation_stage
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.writer import write_pdb
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def manifest_for_tleap(tmp_path, *, solvation_enabled=False):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["structure"]["remove_unknown_heterogens"] = True
    data["protonation"]["method"] = "manual_only"
    data["solvation"]["enabled"] = solvation_enabled
    data["solvation"]["salt_concentration_molar"] = 0.0
    data["validation"]["run_openmm_energy_check"] = False
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "user_mol2",
            "user_mol2": "tests/data/ligands/ligand_sub.good.mol2",
            "user_frcmod": "tests/data/ligands/ligand_sub.frcmod",
        }
    ]
    return make_manifest(data)


def prepared_inputs(manifest, tmp_path):
    normalized = normalize_structure_stage(manifest)
    protonation = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "00.pdb",
        output_protonation_pdb_path=tmp_path / "01.pdb",
    )
    ligand_result = run_ligand_stage(protonation.structure, manifest, output_dir=manifest.project.output_dir)
    return protonation, ligand_result


def fake_tleap(input_path, *, work_dir, executable="tleap"):
    work = Path(work_dir)
    script = Path(input_path).read_text(encoding="utf-8")
    loadpdb = _loadpdb_path(Path(input_path), script)
    for name in ["system.dry", "system.solvated", "system.presalt"]:
        if f"{name}.prmtop" in script:
            (work / f"{name}.prmtop").write_text("prmtop\n", encoding="utf-8")
            (work / f"{name}.inpcrd").write_text("inpcrd\n", encoding="utf-8")
            (work / f"{name}.pdb").write_text(loadpdb.read_text(encoding="utf-8"), encoding="utf-8")
    log_text = "Checking 'system'....\nUnit is OK.\nTotal unperturbed charge:   0.000000\n"
    (work / "tleap.log").write_text(log_text, encoding="utf-8")
    return TLeapRun(
        command_result=CommandResult(
            command=("tleap", "-f", Path(input_path).name),
            cwd=str(work),
            returncode=0,
            stdout="",
            stderr="",
            runtime_seconds=0.01,
        ),
        input_path=Path(input_path),
        log_path=work / "tleap.log",
        summary=parse_tleap_log_text(log_text, returncode=0),
    )


def _loadpdb_path(input_path: Path, script: str) -> Path:
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("system = loadpdb "):
            raw = stripped.split("system = loadpdb ", 1)[1]
            return (input_path.parent / raw).resolve()
    raise AssertionError("mock tleap script did not contain loadpdb")


def test_tleap_stage_writes_dry_and_final_outputs(monkeypatch, tmp_path):
    manifest = manifest_for_tleap(tmp_path, solvation_enabled=False)
    protonation, ligand_result = prepared_inputs(manifest, tmp_path)
    monkeypatch.setattr("mdprep.leap.builder.run_tleap", fake_tleap)

    result = run_tleap_stage(
        structure=protonation.structure,
        manifest=manifest,
        output_dir=manifest.project.output_dir,
        protonation_result=protonation,
        ligand_result=ligand_result,
    )

    assert result.dry_outputs.prmtop.exists()
    assert result.final_outputs.prmtop.exists()
    assert result.final_run is None


def test_tleap_log_missing_parameters_fails(monkeypatch, tmp_path):
    manifest = manifest_for_tleap(tmp_path, solvation_enabled=False)
    protonation, ligand_result = prepared_inputs(manifest, tmp_path)

    def failing_tleap(input_path, *, work_dir, executable="tleap"):
        work = Path(work_dir)
        log_text = "Could not find bond parameter for c1-o\n"
        (work / "tleap.log").write_text(log_text, encoding="utf-8")
        return TLeapRun(
            command_result=CommandResult(("tleap",), str(work), 0, "", "", 0.01),
            input_path=Path(input_path),
            log_path=work / "tleap.log",
            summary=parse_tleap_log_text(log_text, returncode=0),
        )

    monkeypatch.setattr("mdprep.leap.builder.run_tleap", failing_tleap)
    with pytest.raises(ValueError):
        run_tleap_stage(
            structure=protonation.structure,
            manifest=manifest,
            output_dir=manifest.project.output_dir,
            protonation_result=protonation,
            ligand_result=ligand_result,
        )
