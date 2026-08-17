from pathlib import Path
import json
from dataclasses import replace

import pytest

from mdprep.external.runner import CommandResult
from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.protonation.temporary_hydrogenation import TemporaryHydrogenationResult
from mdprep.protonation.xtb_runner import XtbExecutionError, XtbRunResult
from mdprep.structure.models import PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb
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
    cluster_model_path = tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2" / "cluster_model.json"
    assert cluster_model_path.exists()
    cluster_model = json.loads(cluster_model_path.read_text(encoding="utf-8"))
    assert cluster_model["cluster_charge"] == 0
    assert cluster_model["charge_breakdown"]
    assert "C" in cluster_model["HID_element_counts"]
    assert "Ca" not in cluster_model["HID_element_counts"]
    assert Path(cluster_model["tautomer_work_directories"]["HID"]).name == "HID"
    assert Path(cluster_model["tautomer_work_directories"]["HIE"]).name == "HIE"


def test_xtb_tautomers_use_clean_isolated_work_directories(monkeypatch, tmp_path):
    data = manifest_data("tests/data/protein_histidine_ring_hydrogenated.pdb")
    data["protonation"]["method"] = "propka_xtb_his"
    histidine_dir = tmp_path / "prepared" / "protonation" / "histidine_xtb" / "A_HIS2"
    for tautomer in ("HID", "HIE"):
        stale_dir = histidine_dir / tautomer
        stale_dir.mkdir(parents=True)
        (stale_dir / "xtbrestart").write_text("stale restart\n", encoding="utf-8")

    observed_work_dirs: list[Path] = []
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0")],
        ),
    )

    def fake_run_xtb(*, config, xyz_path, work_dir, cluster_charge, stdout_path, stderr_path, input_path=None):
        run_dir = Path(work_dir)
        observed_work_dirs.append(run_dir)
        assert run_dir.name == Path(xyz_path).stem
        assert Path(xyz_path).parent == run_dir
        assert input_path is None or Path(input_path).parent == run_dir
        assert not (run_dir / "xtbrestart").exists()
        energy = -40.01 if run_dir.name == "HID" else -40.00
        Path(stdout_path).write_text(f":: total energy      {energy:.12f} Eh\n", encoding="utf-8")
        Path(stderr_path).write_text("", encoding="utf-8")
        return XtbRunResult(
            command_result=CommandResult(
                command=("xtb", Path(xyz_path).name),
                cwd=str(run_dir),
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
    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
    )

    assert result.xtb_selections[0].selected_state == "HID"
    assert [path.name for path in observed_work_dirs] == ["HID", "HIE"]


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


def test_multiple_neutral_histidines_with_ambiguous_input_hydrogens_do_not_fail(monkeypatch, tmp_path):
    pdb = _two_histidine_pdb_with_ambiguous_neighbor(tmp_path)
    data = manifest_data(str(pdb))
    data["protonation"]["method"] = "propka_xtb_his"
    monkeypatch.setattr(
        "mdprep.protonation.apply.run_propka_workflow",
        lambda structure, manifest, work_dir: fake_propka_result(
            tmp_path,
            [
                PropkaRecord("HIS", 2, "A", 6.0, "HIS 2 A 6.0"),
                PropkaRecord("HIS", 3, "A", 6.0, "HIS 3 A 6.0"),
            ],
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
    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "prepared" / "intermediate" / "01_protonation_assigned.pdb",
    )

    assert [selection.selected_state for selection in result.xtb_selections] == ["HID", "HID"]
    assert any("temporary environment state" in warning for warning in result.warnings)


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
    data["protonation"]["histidine"]["xtb"][
        "add_missing_protein_hydrogens"
    ] = False
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


def test_dehydrogenated_histidine_cluster_uses_temporary_pdbfixer_environment(
    monkeypatch,
    tmp_path,
):
    data = manifest_data("tests/data/protein_histidine_ring.pdb")
    data["protonation"]["method"] = "propka_xtb_his"
    dehydrogenated = read_pdb("tests/data/protein_histidine_ring.pdb")
    temporary_path = tmp_path / "temporary_hydrogenated.pdb"
    temporary_atoms = []
    serial = 0
    for residue in dehydrogenated.residues:
        for atom in residue.atoms:
            serial += 1
            temporary_atoms.append(replace(atom, serial=serial))
        anchor = next(
            (atom for atom in residue.atoms if atom.name == "CB"),
            next(atom for atom in residue.atoms if atom.name == "CA"),
        )
        serial += 1
        temporary_atoms.append(
            replace(
                anchor,
                serial=serial,
                name="HX",
                z=anchor.z + 1.3,
                element="H",
                original_line="",
            )
        )
    write_pdb(
        PdbStructure(
            path=temporary_path,
            atoms=temporary_atoms,
            residues=dehydrogenated.residues,
            model_count=1,
        ),
        temporary_path,
    )
    hydrogenated = read_pdb(temporary_path)

    def fake_hydrogenation(structure, *, ph, random_seed, work_dir):
        assert random_seed == 20260722
        return TemporaryHydrogenationResult(
            backend="pdbfixer",
            backend_version="test",
                platform="Reference",
                ph=ph,
                random_seed=random_seed,
                input_path=tmp_path / "temporary_input.pdb",
            raw_output_path=tmp_path / "temporary_raw.pdb",
            restored_output_path=temporary_path,
            structure=hydrogenated,
            added_hydrogen_count=len(hydrogenated.atoms) - len(structure.atoms),
            maximum_original_atom_displacement_angstrom=0.0,
        )

    monkeypatch.setattr(
        "mdprep.protonation.apply.add_temporary_protein_hydrogens",
        fake_hydrogenation,
    )
    result = run_with_fakes(
        monkeypatch,
        tmp_path,
        data,
        hid_energy=-40.01,
        hie_energy=-40.00,
    )

    assert result.xtb_selections[0].selected_state == "HID"
    assert result.xtb_temporary_hydrogenation is not None
    assert result.xtb_temporary_hydrogenation.final_prepared_pdb_modified is False
    assert not any(
        atom.element == "H"
        for residue in result.structure.residues
        if residue.id.resname == "HID"
        for atom in residue.atoms
    )


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


def _two_histidine_pdb_with_ambiguous_neighbor(tmp_path: Path) -> Path:
    structure = read_pdb("tests/data/protein_histidine_ring_hydrogenated.pdb")
    residue = next(residue for residue in structure.residues if residue.id.resname == "HIS")
    first_atoms = list(residue.atoms)
    second_atoms = []
    serial = 100
    for atom in residue.atoms:
        second_atoms.append(
            atom.__class__(
                **{
                    **atom.__dict__,
                    "serial": serial,
                    "resid": 3,
                    "x": atom.x + 2.0,
                    "original_line": "",
                }
            )
        )
        serial += 1
    nd1 = next(atom for atom in second_atoms if atom.name == "ND1")
    ne2 = next(atom for atom in second_atoms if atom.name == "NE2")
    second_atoms.extend(
        [
            nd1.__class__(
                **{
                    **nd1.__dict__,
                    "serial": serial,
                    "name": "HD1",
                    "x": nd1.x + 1.0,
                    "element": "H",
                    "original_line": "",
                }
            ),
            ne2.__class__(
                **{
                    **ne2.__dict__,
                    "serial": serial + 1,
                    "name": "HE2",
                    "x": ne2.x + 1.0,
                    "element": "H",
                    "original_line": "",
                }
            ),
        ]
    )
    atoms = first_atoms + second_atoms
    residues = [
        residue,
        ResidueRecord(
            id=ResidueId(chain_id="A", resname="HIS", resid=3, icode=None),
            atoms=second_atoms,
            record_names={"ATOM"},
            original_index=1,
        ),
    ]
    path = tmp_path / "two_histidines.pdb"
    write_pdb(PdbStructure(path=path, atoms=atoms, residues=residues, model_count=1), path)
    return path
