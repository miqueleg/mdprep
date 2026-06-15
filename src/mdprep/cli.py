"""Typer command-line interface for mdprep."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mdprep import __version__
from mdprep.config.loader import load_manifest
from mdprep.structure.inspect import InspectionSummary
from mdprep.structure.pdb import PdbParseError, VALID_ALTLOC_POLICIES
from mdprep.workflows.inspect import inspect_structure
from mdprep.workflows.prepare import prepare_system
from mdprep.workflows.selftest import run_selftest
from mdprep.workflows.validate import validate_system


app = typer.Typer(
    name="mdprep",
    help="Reproducible Amber MD preparation workflow manager.",
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mdprep {__version__}")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print the mdprep version and exit.",
    ),
) -> None:
    return None


@app.command("config-check")
def config_check(
    paths: list[Path] = typer.Argument(..., help="One or more YAML manifest paths."),
) -> None:
    """Validate one or more manifest files."""

    table = Table(title="Manifest validation")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Message")

    failed = False
    for path in paths:
        try:
            load_manifest(path)
        except Exception as exc:
            failed = True
            table.add_row(str(path), "FAIL", str(exc))
        else:
            table.add_row(str(path), "PASS", "valid")

    console.print(table)
    if failed:
        raise typer.Exit(1)


@app.command("inspect")
def inspect_command(
    input_structure: Path = typer.Argument(..., help="Input PDB file."),
    altloc_policy: str | None = typer.Option(
        None,
        "--altloc-policy",
        help="Alternate-location policy: highest_occupancy, first, or fail.",
    ),
    disulfide_cutoff: float | None = typer.Option(
        None,
        "--disulfide-cutoff",
        help="SG-SG distance cutoff for possible disulfides.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Manifest path to supply structure and disulfide inspection defaults.",
    ),
) -> None:
    """Inspect an input structure."""

    try:
        config_altloc_policy = None
        config_disulfide_cutoff = None
        if config is not None:
            manifest = load_manifest(config)
            config_altloc_policy = manifest.structure.altloc_policy
            config_disulfide_cutoff = manifest.disulfides.detection_cutoff_angstrom

        selected_altloc_policy = altloc_policy or config_altloc_policy or "highest_occupancy"
        selected_disulfide_cutoff = disulfide_cutoff or config_disulfide_cutoff or 2.2
        if selected_altloc_policy not in VALID_ALTLOC_POLICIES:
            raise ValueError(
                f"Invalid altloc policy {selected_altloc_policy!r}; expected one of {sorted(VALID_ALTLOC_POLICIES)}"
            )

        summary = inspect_structure(
            input_structure,
            altloc_policy=selected_altloc_policy,  # type: ignore[arg-type]
            disulfide_cutoff_angstrom=selected_disulfide_cutoff,
        )
        if json_output:
            console.print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        else:
            _render_inspection(summary)
    except (FileNotFoundError, PdbParseError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("init")
def init_command(
    input_structure: Path = typer.Argument(..., help="Input PDB/mmCIF file."),
    output: Path = typer.Option(Path("system.yaml"), "-o", "--output", help="Output YAML path."),
) -> None:
    """Create an initial manifest from an input structure."""

    message = (
        f"mdprep init is not implemented yet for {input_structure} -> {output}; "
        "Task 1 only bootstraps the CLI and config validation."
    )
    console.print(f"[yellow]{message}[/yellow]")
    raise typer.Exit(1)


@app.command("prepare")
def prepare_command(manifest: Path = typer.Argument(..., help="YAML manifest path.")) -> None:
    """Prepare an Amber system from a manifest."""

    try:
        prepare_system(manifest)
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1) from exc


@app.command("validate")
def validate_command(
    prmtop: Path = typer.Argument(..., help="Amber topology file."),
    inpcrd: Path = typer.Argument(..., help="Amber coordinate file."),
) -> None:
    """Validate a prepared Amber system."""

    try:
        validate_system(prmtop, inpcrd)
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1) from exc


@app.command("selftest")
def selftest_command(
    quick: bool = typer.Option(False, "--quick", help="Run package-level checks only."),
) -> None:
    """Run package self-tests that do not require external chemistry tools."""

    summary = run_selftest(quick=quick, console=console)
    if not summary.passed:
        raise typer.Exit(1)


def main() -> None:
    app()


def _render_inspection(summary: InspectionSummary) -> None:
    data = summary.to_dict()
    counts = data["counts"]
    assert isinstance(counts, dict)

    overview = Table(title="Structure summary")
    overview.add_column("Field")
    overview.add_column("Value")
    overview.add_row("Path", str(data["path"]))
    overview.add_row("Atoms", str(data["total_atoms"]))
    overview.add_row("Residues", str(data["total_residues"]))
    overview.add_row("MODEL records", str(data["model_count"]))
    overview.add_row("Used MODEL", str(data["used_model"]))
    overview.add_row("Protein residues", str(counts["protein_residues"]))
    overview.add_row("Water residues", str(counts["water_residues"]))
    overview.add_row("Heterogen residues", str(counts["heterogen_residues"]))
    overview.add_row("Likely ligands/cofactors", str(counts["likely_ligands"]))
    console.print(overview)

    chains = Table(title="Chains")
    chains.add_column("Chain ID")
    chains.add_column("Display")
    for chain in data["chains"]:
        assert isinstance(chain, dict)
        chains.add_row(str(chain["chain_id"]), str(chain["display"]))
    console.print(chains)

    _render_residue_table("Likely ligands/cofactors", summary.likely_ligands)
    _render_residue_table("Histidines", summary.histidines)
    _render_residue_table("Titratable residues", summary.titratable_residues)
    _render_disulfide_table(summary)

    if summary.structure.warnings:
        warnings = Table(title="Warnings")
        warnings.add_column("Message")
        for warning in summary.structure.warnings:
            warnings.add_row(warning)
        console.print(warnings)


def _render_residue_table(title: str, residues: list[object]) -> None:
    table = Table(title=title)
    table.add_column("Chain")
    table.add_column("Resname")
    table.add_column("Resid")
    table.add_column("Icode")
    table.add_column("Atoms")
    for residue in residues:
        residue_id = residue.id  # type: ignore[attr-defined]
        table.add_row(
            residue_id.chain_id or "<blank>",
            residue_id.resname,
            str(residue_id.resid),
            residue_id.icode or "",
            str(len(residue.atoms)),  # type: ignore[attr-defined]
        )
    console.print(table)


def _render_disulfide_table(summary: InspectionSummary) -> None:
    table = Table(title="Possible disulfides")
    table.add_column("Residue A")
    table.add_column("Residue B")
    table.add_column("Distance (angstrom)")
    for candidate in summary.possible_disulfides:
        table.add_row(
            candidate.a.display(),
            candidate.b.display(),
            f"{candidate.distance_angstrom:.3f}",
        )
    console.print(table)


if __name__ == "__main__":
    main()
