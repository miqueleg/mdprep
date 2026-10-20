"""The equilibration and production logs, as tables to plot.

``equilibration.csv`` puts every Roe-Brooks stage on one time axis. Runs
from before it existed only have ``density_step10.csv`` and the end-of-stage
records in ``md_report.json``; those are read instead, so an older run still
plots.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mdprep.md.roe_brooks import STAGES


def _number(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def read_table(path: str | Path) -> list[dict[str, float | str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [{key: _number(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def equilibration_table(md_dir: str | Path) -> list[dict[str, float | str]]:
    """One row per sample of the whole equilibration, oldest first."""

    directory = Path(md_dir)
    log = directory / "equilibration.csv"
    if log.is_file():
        return read_table(log)
    # An older run: step 10's density trace is all there is over time.
    density = directory / "density_step10.csv"
    if not density.is_file():
        raise FileNotFoundError(f"No equilibration log in {directory}")
    return [
        {
            "stage": "step10",
            "time_ps": row["time_ps"],
            "stage_time_ps": row["time_ps"],
            "density_g_ml": row["density_g_ml"],
            "box_volume_nm3": row["box_volume_nm3"],
        }
        for row in read_table(density)
    ]


def stage_spans(rows: list[dict[str, float | str]]) -> list[dict[str, object]]:
    """Each stage's start and end on the shared time axis, with what it does."""

    spans: list[dict[str, object]] = []
    for row in rows:
        stage = str(row["stage"])
        time = float(row["time_ps"])
        if spans and spans[-1]["stage"] == stage:
            spans[-1]["end_ps"] = time
            continue
        start = float(spans[-1]["end_ps"]) if spans else 0.0
        spans.append(
            {"stage": stage, "start_ps": start, "end_ps": time, "what": STAGES.get(stage, "")}
        )
    return spans


def production_table(md_dir: str | Path) -> list[dict[str, float | str]]:
    """OpenMM's production state log, whatever separator it was written with."""

    path = Path(md_dir) / "production.log"
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].lstrip("#")
    separator = "\t" if "\t" in header else ","
    columns = [column.strip().strip('"') for column in header.split(separator)]
    rows = []
    for line in lines[1:]:
        values = [value.strip() for value in line.split(separator)]
        if len(values) != len(columns):
            continue
        rows.append({column: _number(value) for column, value in zip(columns, values)})
    return rows


def md_report(md_dir: str | Path) -> dict[str, object]:
    return json.loads((Path(md_dir) / "md_report.json").read_text(encoding="utf-8"))
