"""Quick package-level self-test."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path
from tempfile import TemporaryDirectory

from rich.console import Console
from rich.table import Table

import mdprep
from mdprep.ambertools.mol2 import Mol2File, read_mol2
from mdprep.ambertools.resp import write_amber_esp
from mdprep.charges.esp_grid import generate_merz_kollman_grid
from mdprep.config.loader import load_manifest
from mdprep.external.discovery import optional_executable_report
from mdprep.leap.forcefields import protein_leaprc, water_box, water_leaprc
from mdprep.qm.pyscf_runner import pyscf_available
from mdprep.validation.openmm_check import openmm_available
from mdprep.validation.parmed_check import parmed_available
import numpy as np


OPTIONAL_EXECUTABLES = [
    "tleap",
    "antechamber",
    "parmchk2",
    "respgen",
    "resp",
    "propka3",
    "propka",
    "xtb",
]


@dataclass(frozen=True)
class SelftestSummary:
    passed: bool
    checked_examples: int


def _project_root() -> Path | None:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[3],
    ]
    for candidate in candidates:
        if (candidate / "examples").is_dir():
            return candidate
    return None


def _validate_example_manifests(root: Path | None) -> tuple[int, list[str]]:
    if root is not None:
        examples = sorted((root / "examples").glob("*.yaml"))
        errors: list[str] = []
        for example in examples:
            try:
                load_manifest(example)
            except Exception as exc:
                errors.append(f"{example.name}: {exc}")
        return len(examples), errors

    resource = files("mdprep").joinpath("_selftest_data/basic_protein.yaml")
    with as_file(resource) as example:
        try:
            load_manifest(example)
        except Exception as exc:
            return 1, [f"{example.name}: {exc}"]
    return 1, []


def _read_mol2_fixture(root: Path | None) -> Mol2File:
    if root is not None:
        return read_mol2(root / "tests" / "data" / "ligands" / "ligand_sub.good.mol2")

    resource = files("mdprep").joinpath("_selftest_data/ligand_sub.mol2")
    with as_file(resource) as fixture:
        return read_mol2(fixture)


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

    example_count, example_errors = _validate_example_manifests(root)
    checks.append(
        (
            "Example manifests",
            not example_errors and example_count > 0,
            f"{example_count} validated" if not example_errors else "; ".join(example_errors),
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

    try:
        mol2 = _read_mol2_fixture(root)
        checks.append(("mol2 parser fixture", len(mol2.atoms) == 2, f"{len(mol2.atoms)} atoms"))
    except Exception as exc:
        checks.append(("mol2 parser fixture", False, str(exc)))

    try:
        ok = (
            protein_leaprc("ff19SB") == "leaprc.protein.ff19SB"
            and water_leaprc("TIP3P") == "leaprc.water.tip3p"
            and water_box("OPC") == "OPCBOX"
        )
        checks.append(("force-field mappings", ok, "ok" if ok else "unexpected mapping"))
    except Exception as exc:
        checks.append(("force-field mappings", False, str(exc)))

    checks.append(("ParmEd import", True, "available" if parmed_available() else "not available"))
    checks.append(("OpenMM import", True, "available" if openmm_available() else "not available"))
    checks.append(("PySCF import", True, "available" if pyscf_available() else "not available"))

    try:
        coords = np.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]], dtype=float)
        grid_a = generate_merz_kollman_grid(
            elements=["O", "H"],
            coordinates=coords,
            vdw_scale_factors=[1.4],
            point_density_per_square_angstrom=0.25,
            exclude_inside_vdw_scale=1.4,
            max_points=100,
        )
        grid_b = generate_merz_kollman_grid(
            elements=["O", "H"],
            coordinates=coords,
            vdw_scale_factors=[1.4],
            point_density_per_square_angstrom=0.25,
            exclude_inside_vdw_scale=1.4,
            max_points=100,
        )
        checks.append(("ESP grid determinism", bool(np.allclose(grid_a.points, grid_b.points)), f"{len(grid_a.points)} points"))
    except Exception as exc:
        checks.append(("ESP grid determinism", False, str(exc)))

    try:
        atom_coords = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
        grid_coords = np.asarray(
            [[0.0, 3.0, 0.0], [2.0, 3.0, 0.0], [1.0, 3.5, 0.0], [1.0, -3.0, 0.0]],
            dtype=float,
        )
        esp = np.asarray([-0.1, 0.1, 0.0, 0.0], dtype=float)
        with TemporaryDirectory(prefix="mdprep-selftest-") as temporary_dir:
            path = write_amber_esp(
                atom_coordinates_bohr=atom_coords,
                grid_coordinates_bohr=grid_coords,
                esp_values_au=esp,
                path=Path(temporary_dir) / "test.esp",
            )
            lines = path.read_text(encoding="ascii").splitlines()
        fixed_width_ok = lines[0] == "    2    4" and len(lines[1]) == 65 and len(lines[3]) == 65
        checks.append(("Amber RESP ESP format", fixed_width_ok, "2 atoms, 4 grid points"))
    except Exception as exc:
        checks.append(("Amber RESP ESP format", False, str(exc)))

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
        checked_examples=example_count,
    )
