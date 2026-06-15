"""Quick package-level self-test."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from rich.console import Console
from rich.table import Table

import mdprep
from mdprep.config.loader import load_manifest
from mdprep.external.discovery import optional_executable_report


OPTIONAL_EXECUTABLES = ["tleap", "antechamber", "parmchk2", "propka3", "propka", "xtb"]


@dataclass(frozen=True)
class SelftestSummary:
    passed: bool
    checked_examples: int


def _project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[3],
    ]
    for candidate in candidates:
        if (candidate / "examples").is_dir():
            return candidate
    return Path.cwd()


def _blocked_tokens() -> tuple[str, ...]:
    return (
        "".join(("open", "babel")),
        "".join(("Open", "Babel")),
        "".join(("py", "bel")),
        "".join(("o", "babel")),
    )


def _source_has_blocked_tokens() -> list[Path]:
    source_root = Path(__file__).resolve().parents[1]
    blocked = _blocked_tokens()
    matches: list[Path] = []
    for path in source_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in blocked):
            matches.append(path)
    return matches


def run_selftest(*, quick: bool = False, console: Console | None = None) -> SelftestSummary:
    out = console or Console()
    root = _project_root()
    examples = sorted((root / "examples").glob("*.yaml"))

    out.print(f"mdprep {mdprep.__version__}")
    out.print("Mode: quick" if quick else "Mode: standard")

    checks: list[tuple[str, bool, str]] = []

    try:
        import_module("mdprep")
        import_module("mdprep.config.loader")
        import_module("mdprep.external.discovery")
        checks.append(("Python imports", True, "ok"))
    except Exception as exc:
        checks.append(("Python imports", False, str(exc)))

    example_errors = []
    for example in examples:
        try:
            load_manifest(example)
        except Exception as exc:
            example_errors.append(f"{example.name}: {exc}")
    checks.append(
        (
            "Example manifests",
            not example_errors and bool(examples),
            f"{len(examples)} validated" if not example_errors else "; ".join(example_errors),
        )
    )

    blocked_matches = _source_has_blocked_tokens()
    checks.append(
        (
            "Prohibited toolkit source scan",
            not blocked_matches,
            "ok" if not blocked_matches else ", ".join(str(path) for path in blocked_matches),
        )
    )

    table = Table(title="mdprep self-test")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, ok, detail in checks:
        table.add_row(name, "PASS" if ok else "FAIL", detail)
    out.print(table)

    exec_table = Table(title="Optional executables")
    exec_table.add_column("Executable")
    exec_table.add_column("Path")
    for name, path in optional_executable_report(OPTIONAL_EXECUTABLES).items():
        exec_table.add_row(name, path or "not found")
    out.print(exec_table)

    return SelftestSummary(
        passed=all(ok for _, ok, _ in checks),
        checked_examples=len(examples),
    )

