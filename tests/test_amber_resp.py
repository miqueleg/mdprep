from pathlib import Path

import numpy as np
import pytest

from mdprep.ambertools.resp import (
    AmberRespError,
    parse_resp_qout,
    run_amber_resp_fit,
    write_amber_esp,
)
from mdprep.external.runner import CommandResult


def test_amber_esp_writer_uses_exact_fixed_width_header(tmp_path):
    atoms = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    points = np.column_stack(
        [np.linspace(2.0, 4.0, 430), np.ones(430), np.full(430, -1.0)]
    )
    values = np.linspace(-0.1, 0.1, 430)
    path = write_amber_esp(
        atom_coordinates_bohr=atoms,
        grid_coordinates_bohr=points,
        esp_values_au=values,
        path=tmp_path / "ligand.esp",
    )

    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0] == "    2  430"
    assert int(lines[0][0:5]) == 2
    assert int(lines[0][5:10]) == 430
    assert len(lines) == 1 + len(atoms) + len(points)
    assert len(lines[1]) == 65
    assert len(lines[1 + len(atoms)]) == 65


def test_parse_resp_qout_requires_exact_finite_atom_coverage(tmp_path):
    path = tmp_path / "qout"
    path.write_text(" -0.250000  0.250000\n", encoding="utf-8")
    assert parse_resp_qout(path, atom_count=2) == pytest.approx([-0.25, 0.25])

    with pytest.raises(AmberRespError, match="expected exactly 3"):
        parse_resp_qout(path, atom_count=3)

    path.write_text("nan 0.0\n", encoding="utf-8")
    with pytest.raises(AmberRespError, match="non-finite"):
        parse_resp_qout(path, atom_count=2)


def test_two_stage_resp_orchestration_preserves_commands_and_constraints(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "mdprep.ambertools.resp.which_executable",
        lambda name: f"/fake/bin/{name}",
    )

    def fake_command(command, *, cwd):
        calls.append(list(command))
        command_name = Path(command[0]).name

        def output_for(flag):
            path = Path(command[command.index(flag) + 1])
            return path if path.is_absolute() else Path(cwd) / path

        if command_name == "antechamber":
            output_for("-o").write_text(
                "CHARGE      0.00 ( 0 )\n"
                "ATOM      1  C1  SUB     1       5.000   5.000   5.000  0.100000        c1\n"
                "ATOM      2  O1  SUB     1       6.000   5.000   5.000 -0.100000         o\n"
                "BOND    1    1    2    9     C1   O1\n",
                encoding="utf-8",
            )
        elif command_name == "respgen":
            stage = command[command.index("-f") + 1]
            stage_2 = stage == "resp2"
            output_for("-o").write_text(
                "Resp charges for organic molecule\n\n"
                " &cntrl\n"
                f" iqopt = {2 if stage_2 else 1},\n"
                f" qwt = {0.001 if stage_2 else 0.0005:.5f},\n"
                " &end\n"
                "    1.0\n"
                "Resp charges for organic molecule\n"
                "    0    2\n"
                f"    6  {-99 if stage_2 else 0}\n"
                f"    8  {-99 if stage_2 else 0}\n\n",
                encoding="utf-8",
            )
        elif command_name == "resp":
            stage_2 = "-q" in command
            for flag, content in [
                ("-o", "Convergence in    2 iterations\n"),
                ("-p", "punch\n"),
                ("-s", "esout\n"),
                ("-t", " -0.300000  0.300000\n" if stage_2 else " -0.280000  0.280000\n"),
            ]:
                output_for(flag).write_text(content, encoding="utf-8")
        return CommandResult(tuple(command), str(cwd), 0, "", "", 0.01)

    monkeypatch.setattr("mdprep.ambertools.resp.run_command", fake_command)
    atoms = np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float)
    points = np.asarray([[8.0, 5.0, 5.0], [4.0, 7.0, 5.0], [5.5, 5.0, 8.0]])
    charges = np.asarray([-0.3, 0.3])
    values = (1.0 / np.linalg.norm(points[:, None, :] - atoms[None, :, :], axis=2)) @ charges

    result = run_amber_resp_fit(
        provisional_mol2_path="tests/data/ligands/ligand_sub.good.mol2",
        atom_coordinates_bohr=atoms,
        grid_coordinates_bohr=points,
        esp_values_au=values,
        total_charge=0,
        multiplicity=1,
        atom_types="gaff2",
        work_dir=tmp_path / "resp",
    )

    assert result.charges == pytest.approx([-0.3, 0.3])
    assert result.fitting_mode == "ambertools_two_stage_resp"
    assert result.stage_1_ivary == (0, 0)
    assert result.stage_2_ivary == (-99, -99)
    assert [run.name for run in result.command_runs] == [
        "antechamber_mol2_to_ac",
        "respgen_resp1",
        "respgen_resp2",
        "resp_stage1",
        "resp_stage2",
    ]
    assert all(run.record_path.exists() for run in result.command_runs)
    assert all("Multiwfn" not in " ".join(command) for command in calls)
    assert (tmp_path / "resp" / "resp_fit.json").exists()
