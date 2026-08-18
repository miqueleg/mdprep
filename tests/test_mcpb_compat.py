from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mdprep.external.runner import CommandResult
from mdprep.metals.mcpb import _run_mcpb
from mdprep.metals.mcpb_compat import (
    McpbSeminarioCompatibilityError,
    coerce_negligible_imaginary,
)


def test_compatibility_adapter_coerces_only_negligible_imaginary_values():
    value = coerce_negligible_imaginary(
        np.complex128(12.5 + 1.0e-12j),
        label="bond force constant",
    )

    assert value == pytest.approx(12.5)
    assert isinstance(value, float)


def test_compatibility_adapter_rejects_material_complex_values():
    with pytest.raises(McpbSeminarioCompatibilityError, match="materially complex"):
        coerce_negligible_imaginary(
            np.complex128(12.5 + 1.0e-4j),
            label="bond force constant",
        )


def test_mcpb_seminario_step_uses_recorded_compatibility_command(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[tuple[str, ...]] = []

    def fake_run_command(command, *, cwd):
        calls.append(tuple(command))
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=0,
            stdout="ok\n",
            stderr="",
            runtime_seconds=0.1,
        )

    monkeypatch.setattr("mdprep.metals.mcpb.run_command", fake_run_command)
    input_path = tmp_path / "mcpb.in"
    input_path.write_text("group_name test\n", encoding="utf-8")

    run = _run_mcpb("/amber/bin/MCPB.py", input_path, "2s", tmp_path)

    assert Path(calls[0][1]).name == "mcpb_compat.py"
    assert calls[0][2] == "--mcpb-script"
    assert "/amber/bin/MCPB.py" in calls[0]
    assert calls[0][-4:] == ("-i", "mcpb.in", "-s", "2s")
    assert run.result.command == calls[0]


def test_non_seminario_mcpb_step_remains_direct(monkeypatch, tmp_path: Path):
    calls: list[tuple[str, ...]] = []

    def fake_run_command(command, *, cwd):
        calls.append(tuple(command))
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=0,
            stdout="ok\n",
            stderr="",
            runtime_seconds=0.1,
        )

    monkeypatch.setattr("mdprep.metals.mcpb.run_command", fake_run_command)
    input_path = tmp_path / "mcpb.in"
    input_path.write_text("group_name test\n", encoding="utf-8")

    _run_mcpb("/amber/bin/MCPB.py", input_path, "3b", tmp_path)

    assert calls == [("/amber/bin/MCPB.py", "-i", "mcpb.in", "-s", "3b")]


def test_mcpb_step_exports_resolved_amberhome(monkeypatch, tmp_path: Path):
    captured_environment = None

    def fake_run_command(command, *, cwd, env):
        nonlocal captured_environment
        captured_environment = env
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=0,
            stdout="ok\n",
            stderr="",
            runtime_seconds=0.1,
        )

    monkeypatch.setattr("mdprep.metals.mcpb.run_command", fake_run_command)
    input_path = tmp_path / "mcpb.in"
    input_path.write_text("group_name test\n", encoding="utf-8")

    _run_mcpb(
        "/amber/bin/MCPB.py",
        input_path,
        "1",
        tmp_path,
        environment={"AMBERHOME": "/amber"},
    )

    assert captured_environment == {"AMBERHOME": "/amber"}
