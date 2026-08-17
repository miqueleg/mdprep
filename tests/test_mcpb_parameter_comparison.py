from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.config.models import McpbParameterComparisonConfig
from mdprep.metals.parameter_comparison import (
    McpbParameterComparisonError,
    compare_mcpb_parameter_files,
    parse_mcpb_seminario_terms,
)


def _write_frcmod(
    path: Path,
    *,
    bond_k: float = 100.0,
    bond_r: float = 2.0,
    angle_k: float = 50.0,
    angle_value: float = 110.0,
    reverse: bool = False,
) -> Path:
    bond = "M1-Y1" if not reverse else "Y1-M1"
    angle = "Y1-M1-Y2" if not reverse else "Y2-M1-Y1"
    path.write_text(
        "MCPB parameters\n"
        "BOND\n"
        f"{bond:<5s} {bond_k:8.3f} {bond_r:8.4f}  "
        "Created by Seminario method using MCPB.py\n"
        "C -N   300.0 1.40 inherited parameter\n\n"
        "ANGL\n"
        f"{angle:<8s} {angle_k:8.3f} {angle_value:8.3f}  "
        "Created by Seminario method using MCPB.py\n"
        "C -N -H    40.0 120.0 inherited parameter\n\n"
        "NONB\nM1 1.0 0.1\n",
        encoding="utf-8",
    )
    return path


def test_parser_selects_only_mcpb_seminario_terms_and_canonicalizes(tmp_path):
    terms = parse_mcpb_seminario_terms(
        _write_frcmod(tmp_path / "candidate.frcmod", reverse=True)
    )

    assert sorted(terms) == ["ANGL:Y1-M1-Y2", "BOND:M1-Y1"]
    assert terms["BOND:M1-Y1"].force_constant == pytest.approx(100.0)


def test_parser_rejects_nonpositive_force_constant_by_default(tmp_path):
    frcmod = _write_frcmod(tmp_path / "zero.frcmod", bond_k=0.0)

    with pytest.raises(McpbParameterComparisonError, match="Non-positive"):
        parse_mcpb_seminario_terms(frcmod)


def test_comparison_reports_term_metrics_and_reversed_keys_match(tmp_path):
    candidate = _write_frcmod(
        tmp_path / "candidate.frcmod",
        bond_k=110.0,
        bond_r=2.02,
        angle_k=45.0,
        angle_value=112.0,
    )
    reference = _write_frcmod(
        tmp_path / "reference.frcmod",
        reverse=True,
    )
    result = compare_mcpb_parameter_files(
        candidate,
        config=McpbParameterComparisonConfig(
            reference_frcmod=str(reference),
            max_bond_force_constant_relative_rmse=0.11,
            max_angle_force_constant_relative_rmse=0.11,
            max_bond_equilibrium_distance_rmse_angstrom=0.03,
            max_angle_equilibrium_value_rmse_degrees=2.1,
        ),
        output_dir=tmp_path / "reports",
    )

    assert result.passed
    assert result.metrics["bonds"]["force_constant_relative_rmse"] == pytest.approx(
        0.1
    )
    assert result.metrics["angles"]["force_constant_relative_rmse"] == pytest.approx(
        0.1
    )
    assert result.json_path.is_file()
    assert result.csv_path.is_file()
    assert result.markdown_path.is_file()


def test_threshold_failure_preserves_machine_readable_report(tmp_path):
    candidate = _write_frcmod(tmp_path / "candidate.frcmod", bond_k=150.0)
    reference = _write_frcmod(tmp_path / "reference.frcmod")
    output = tmp_path / "reports"

    with pytest.raises(McpbParameterComparisonError, match="acceptance criteria"):
        compare_mcpb_parameter_files(
            candidate,
            config=McpbParameterComparisonConfig(
                reference_frcmod=str(reference),
                max_bond_force_constant_relative_rmse=0.2,
            ),
            output_dir=output,
        )

    report = json.loads(
        (output / "mcpb_parameter_comparison.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "fail"
    assert report["violations"]


def test_comparison_reports_nonpositive_reference_without_hiding_terms(tmp_path):
    candidate = _write_frcmod(tmp_path / "candidate.frcmod")
    reference = _write_frcmod(
        tmp_path / "reference.frcmod",
        bond_k=0.0,
        angle_k=0.0,
    )
    output = tmp_path / "reports"

    result = compare_mcpb_parameter_files(
        candidate,
        config=McpbParameterComparisonConfig(
            reference_frcmod=str(reference),
            fail_on_thresholds=False,
        ),
        output_dir=output,
    )

    assert result.status == "fail"
    assert result.nonpositive_candidate_force_constants == ()
    assert result.nonpositive_reference_force_constants == (
        "ANGL:Y1-M1-Y2",
        "BOND:M1-Y1",
    )
    assert result.metrics["bonds"]["force_constant_relative_count"] == 0
    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["matched_term_count"] == 2
    assert report["nonpositive_reference_force_constants"]


def test_compare_mcpb_cli_writes_reports(tmp_path):
    candidate = _write_frcmod(tmp_path / "candidate.frcmod")
    reference = _write_frcmod(tmp_path / "reference.frcmod", reverse=True)
    output = tmp_path / "cli_reports"

    result = CliRunner().invoke(
        app,
        [
            "compare-mcpb",
            str(candidate),
            str(reference),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "REPORT ONLY" in result.output
    assert (output / "mcpb_parameter_comparison.json").is_file()


def test_compare_mcpb_cli_report_only_preserves_invalid_zero_terms(tmp_path):
    candidate = _write_frcmod(tmp_path / "candidate.frcmod")
    reference = _write_frcmod(tmp_path / "reference.frcmod", bond_k=0.0)
    output = tmp_path / "cli_reports"

    result = CliRunner().invoke(
        app,
        [
            "compare-mcpb",
            str(candidate),
            str(reference),
            "--output-dir",
            str(output),
            "--report-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "FAIL (report only)" in result.output
    report = json.loads(
        (output / "mcpb_parameter_comparison.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "fail"
    assert report["nonpositive_reference_force_constants"] == ["BOND:M1-Y1"]
