import json
import hashlib

import pytest

from mdprep.config.models import AshRefinementConfig, MacePolar1RefinementConfig
from mdprep.external.runner import CommandResult
from mdprep.refinement.ash import (
    AshRefinementError,
    capture_ash_refinement_cache,
    run_ash_refinement,
)
from mdprep.refinement.selection import RefinementSelection


def test_ash_runner_preserves_driver_io_and_external_record(tmp_path, monkeypatch):
    python_executable = tmp_path / "python"
    xtb_executable = tmp_path / "xtb"
    python_executable.write_text("fake", encoding="utf-8")
    xtb_executable.write_text("fake", encoding="utf-8")
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    prmtop.write_text("topology", encoding="utf-8")
    inpcrd.write_text("coordinates", encoding="utf-8")
    selection = RefinementSelection(
        qm_atom_indices=(0,),
        active_atom_indices=(0, 1),
        qm_residue_indices=(0,),
        active_residue_indices=(0,),
        qm_residues=(),
        active_residues=(),
        charge_components=(),
        total_qm_charge=0,
        total_qm_multiplicity=1,
        topology_atom_count=2,
        excluded_virtual_site_indices=(),
    )

    def fake_run_command(command, *, cwd):
        work = type(tmp_path)(cwd)
        (work / "ash_refinement_output.json").write_text(
            json.dumps(
                {
                    "coordinates_angstrom": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    "final_energy_hartree": -10.5,
                }
            ),
            encoding="utf-8",
        )
        return CommandResult(
            command=tuple(str(item) for item in command),
            cwd=str(work),
            returncode=0,
            stdout="ASH stdout",
            stderr="",
            runtime_seconds=1.25,
        )

    monkeypatch.setattr("mdprep.refinement.ash.run_command", fake_run_command)
    from mdprep.config.models import AshRefinementConfig

    result = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=selection,
        config=AshRefinementConfig(
            python_executable=str(python_executable),
            xtb_executable=str(xtb_executable),
            allow_unusual_link_boundaries=True,
        ),
        work_dir=tmp_path / "ash",
    )

    assert result.final_energy_hartree == -10.5
    assert result.optimized_coordinates_angstrom[1] == (4.0, 5.0, 6.0)
    assert result.driver_path.exists()
    assert result.input_path.exists()
    assert result.stdout_path.read_text(encoding="utf-8") == "ASH stdout"
    command_record = json.loads(result.command_record_path.read_text(encoding="utf-8"))
    assert command_record["returncode"] == 0
    assert command_record["stdout"] == "ASH stdout"
    payload = json.loads(result.input_path.read_text(encoding="utf-8"))
    assert payload["qm_atom_indices"] == [0]
    assert payload["active_atom_indices"] == [0, 1]
    assert payload["qm_boundary_excluded_atom_indices"] == []
    assert payload["qm_charge"] == 0
    assert payload["allow_unusual_link_boundaries"] is True
    driver = result.driver_path.read_text(encoding="utf-8")
    assert 'unusualboundary=data["allow_unusual_link_boundaries"]' in driver
    assert 'excludeboundaryatomlist=data["qm_boundary_excluded_atom_indices"]' in driver
    assert "MM_PDB_traj_write=False" in driver
    assert "MM_PDB_traj_write=True" not in driver


def test_ash_runner_resumes_only_with_identical_hashed_inputs(tmp_path, monkeypatch):
    python_executable = tmp_path / "python"
    xtb_executable = tmp_path / "xtb"
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    for path in (python_executable, xtb_executable, inpcrd):
        path.write_text("fixture", encoding="utf-8")
    prmtop.write_text(
        "%VERSION DATE = original timestamp\nfixture\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("mdprep.refinement.ash.run_command", _fake_success)
    config = AshRefinementConfig(
        python_executable=str(python_executable),
        xtb_executable=str(xtb_executable),
    )
    work = tmp_path / "ash_resume"
    first = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=_minimal_selection(),
        config=config,
        work_dir=work,
    )
    cache = capture_ash_refinement_cache(work)

    def unexpected_external_run(*args, **kwargs):
        raise AssertionError("cached ASH result should have been reused")

    monkeypatch.setattr("mdprep.refinement.ash.run_command", unexpected_external_run)
    resumed = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=_minimal_selection(),
        config=config,
        work_dir=work,
        cached_run=cache,
    )

    assert resumed.reused_cached_result is True
    assert resumed.final_energy_hartree == -9.5

    prmtop.write_text(
        "%VERSION DATE = changed timestamp\nfixture\n",
        encoding="utf-8",
    )
    original_prmtop = cache.input_payload["prmtop_path"]
    assert original_prmtop == str(prmtop.resolve())
    # Re-capturing reads immutable metadata rather than rebasing the cache from
    # the mutable provisional topology.
    timestamp_cache = capture_ash_refinement_cache(work)
    timestamp_resumed = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=_minimal_selection(),
        config=config,
        work_dir=work,
        cached_run=timestamp_cache,
    )
    assert timestamp_resumed.reused_cached_result is True

    prmtop.write_text("changed topology", encoding="utf-8")
    with pytest.raises(AshRefinementError, match="provisional prmtop"):
        run_ash_refinement(
            prmtop_path=prmtop,
            inpcrd_path=inpcrd,
            selection=_minimal_selection(),
            config=config,
            work_dir=work,
            cached_run=timestamp_cache,
        )


def _minimal_selection() -> RefinementSelection:
    return RefinementSelection(
        qm_atom_indices=(0,),
        active_atom_indices=(0, 1),
        qm_residue_indices=(0,),
        active_residue_indices=(0,),
        qm_residues=(),
        active_residues=(),
        charge_components=(),
        total_qm_charge=0,
        total_qm_multiplicity=1,
        topology_atom_count=2,
        excluded_virtual_site_indices=(),
    )


def _fake_success(command, *, cwd):
    work = type(cwd)(cwd)
    payload = json.loads(
        (work / "ash_refinement_input.json").read_text(encoding="utf-8")
    )
    (work / "ash_refinement_output.json").write_text(
        json.dumps(
            {
                "coordinates_angstrom": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "final_energy_hartree": -9.5,
                "status": "converged",
                "qm_method": payload["qm_method"],
                "embedding": payload["embedding"],
            }
        ),
        encoding="utf-8",
    )
    return CommandResult(
        command=tuple(str(item) for item in command),
        cwd=str(work),
        returncode=0,
        stdout="ASH stdout",
        stderr="",
        runtime_seconds=1.0,
    )


def test_gxtb_refinement_driver_uses_mechanical_embedding(tmp_path, monkeypatch):
    python_executable = tmp_path / "python"
    xtb_executable = tmp_path / "xtb"
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    for path in (python_executable, xtb_executable, prmtop, inpcrd):
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr("mdprep.refinement.ash.run_command", _fake_success)

    result = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=_minimal_selection(),
        config=AshRefinementConfig(
            python_executable=str(python_executable),
            xtb_executable=str(xtb_executable),
        ),
        work_dir=tmp_path / "gxtb",
        qm_method="gxtb",
        embedding="mechanical",
    )

    payload = json.loads(result.input_path.read_text(encoding="utf-8"))
    driver = result.driver_path.read_text(encoding="utf-8")
    assert payload["qm_method"] == "gxtb"
    assert payload["embedding"] == "mechanical"
    assert 'xtbmethod="GFN2" if data["qm_method"] == "gfn2_xtb" else "G-XTB"' in driver
    assert 'else "Mechanical"' in driver


def test_ash_runner_allows_fixed_qm_heavy_atoms(tmp_path, monkeypatch):
    python_executable = tmp_path / "python"
    xtb_executable = tmp_path / "xtb"
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    for path in (python_executable, xtb_executable, prmtop, inpcrd):
        path.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr("mdprep.refinement.ash.run_command", _fake_success)
    selection = RefinementSelection(
        qm_atom_indices=(0,),
        active_atom_indices=(1,),
        qm_residue_indices=(0,),
        active_residue_indices=(0,),
        qm_residues=(),
        active_residues=(),
        charge_components=(),
        total_qm_charge=0,
        total_qm_multiplicity=1,
        topology_atom_count=2,
        excluded_virtual_site_indices=(),
        movable_atom_policy="active_region_hydrogens",
    )

    result = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=selection,
        config=AshRefinementConfig(
            python_executable=str(python_executable),
            xtb_executable=str(xtb_executable),
        ),
        work_dir=tmp_path / "fixed_qm_heavy",
    )

    payload = json.loads(result.input_path.read_text(encoding="utf-8"))
    assert payload["qm_atom_indices"] == [0]
    assert payload["active_atom_indices"] == [1]


def test_mace_refinement_pins_model_and_writes_driver_settings(tmp_path, monkeypatch):
    python_executable = tmp_path / "python"
    prmtop = tmp_path / "system.prmtop"
    inpcrd = tmp_path / "system.inpcrd"
    model = tmp_path / "MACEPOLAR1Mmodel"
    search_path = tmp_path / "site-packages"
    search_path.mkdir()
    for path in (python_executable, prmtop, inpcrd):
        path.write_text("fixture", encoding="utf-8")
    model.write_bytes(b"pinned MACE model")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    config = MacePolar1RefinementConfig(
        model="polar-1-m",
        model_path=str(model),
        expected_model_sha256=digest,
        python_search_paths=[str(search_path)],
        accept_model_license=True,
    )
    monkeypatch.setattr("mdprep.refinement.ash.run_command", _fake_success)

    result = run_ash_refinement(
        prmtop_path=prmtop,
        inpcrd_path=inpcrd,
        selection=_minimal_selection(),
        config=AshRefinementConfig(python_executable=str(python_executable)),
        work_dir=tmp_path / "mace",
        qm_method="mace_polar1",
        embedding="mechanical",
        mace_polar1=config,
    )

    payload = json.loads(result.input_path.read_text(encoding="utf-8"))
    assert payload["xtb_directory"] is None
    assert payload["mace"]["model_sha256"] == digest
    assert result.mace_model_sha256 == digest
    driver = result.driver_path.read_text(encoding="utf-8")
    assert "MACETheory(" in driver
    assert 'default_dtype="float64"' in driver

    bad = config.model_copy(update={"expected_model_sha256": "0" * 64})
    with pytest.raises(AshRefinementError, match="checksum mismatch"):
        run_ash_refinement(
            prmtop_path=prmtop,
            inpcrd_path=inpcrd,
            selection=_minimal_selection(),
            config=AshRefinementConfig(python_executable=str(python_executable)),
            work_dir=tmp_path / "mace_bad",
            qm_method="mace_polar1",
            embedding="mechanical",
            mace_polar1=bad,
        )
