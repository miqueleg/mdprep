"""Typer command-line interface for mdprep."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mdprep import __version__
from mdprep.config.decisions import DecisionError, plan_decisions
from mdprep.config.loader import load_manifest
from mdprep.config.models import McpbParameterComparisonConfig
from mdprep.metals.parameter_comparison import (
    McpbParameterComparisonError,
    compare_mcpb_parameter_files,
)
from mdprep.md.roe_brooks import RoeBrooksError, run_roe_brooks
from mdprep.structure.inspect import InspectionSummary
from mdprep.structure.normalize import StructureNormalizationError
from mdprep.structure.pdb import PdbParseError, VALID_ALTLOC_POLICIES
from mdprep.workflows.init import generate_starter_manifest
from mdprep.workflows.wizard import format_question, run_wizard
from mdprep.workflows.inspect import inspect_structure
from mdprep.workflows.prepare import PrepareWorkflowError, prepare_system
from mdprep.workflows.selftest import run_selftest
from mdprep.workflows.validate import validate_system


app = typer.Typer(
    name="mdprep",
    help="Reproducible Amber MD preparation workflow manager.",
    no_args_is_help=True,
)
console = Console()


def _print_json(payload: object) -> None:
    """Emit JSON on stdout unwrapped and unstyled.

    rich reflows to the console width and interprets square brackets as markup,
    either of which would corrupt output another program has to parse.
    """

    print(json.dumps(payload, indent=2, sort_keys=True))


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
            _print_json(summary.to_dict())
        else:
            _render_inspection(summary)
    except (FileNotFoundError, PdbParseError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("plan")
def plan_command(
    input_structure: Path = typer.Argument(..., help="Input PDB file."),
    json_output: bool = typer.Option(
        False, "--json", help="Print the decision plan as machine-readable JSON."
    ),
    altloc_policy: str | None = typer.Option(
        None,
        "--altloc-policy",
        help="Alternate-location policy: highest_occupancy, first, or fail.",
    ),
    disulfide_cutoff: float | None = typer.Option(
        None, "--disulfide-cutoff", help="SG-SG distance cutoff for possible disulfides."
    ),
) -> None:
    """List the manifest decisions this structure actually requires.

    The JSON form is the contract any other front end builds on: an interactive
    wizard, a generated web form, or a notebook all consume this same plan.
    """

    try:
        selected_policy = altloc_policy or "highest_occupancy"
        if selected_policy not in VALID_ALTLOC_POLICIES:
            raise ValueError(
                f"Invalid altloc policy {selected_policy!r}; expected one of "
                f"{sorted(VALID_ALTLOC_POLICIES)}"
            )
        plan = plan_decisions(
            input_structure,
            altloc_policy=selected_policy,  # type: ignore[arg-type]
            disulfide_cutoff_angstrom=disulfide_cutoff or 2.2,
        )
    except (FileNotFoundError, PdbParseError, DecisionError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if json_output:
        _print_json(plan.to_dict())
        return
    _render_plan(plan)


def _render_plan(plan) -> None:
    findings = Table(title="What mdprep found")
    findings.add_column("Item")
    findings.add_column("Value")
    for key in (
        "total_atoms",
        "total_residues",
        "protein_residues",
        "water_residues",
    ):
        findings.add_row(key.replace("_", " "), str(plan.findings[key]))
    for key in ("chains", "ligands", "metal_ions", "histidines", "possible_disulfides"):
        values = plan.findings[key]
        findings.add_row(
            key.replace("_", " "), ", ".join(str(value) for value in values) or "none"
        )
    console.print(findings)

    table = Table(title="Decisions required by this structure")
    table.add_column("Decision", overflow="fold")
    table.add_column("Default", overflow="fold")
    table.add_column("Conditional on", overflow="fold")
    for decision in plan.decisions:
        default = (
            "[red]you must choose[/red]"
            if decision.requires_user_input or decision.default is None
            else decision.default
        )
        condition = (
            ", ".join(
                f"{item.decision_id}={'|'.join(item.values)}" for item in decision.when
            )
            or "-"
        )
        table.add_row(decision.id, default, condition)
    console.print(table)
    blocking = plan.blocking()
    console.print(
        f"{len(plan.decisions)} decisions, of which [red]{len(blocking)}[/red] cannot be "
        "defaulted and must be answered by a person."
    )
    console.print("Run [bold]mdprep init --interactive[/bold] to answer them and write a manifest.")


@app.command("init")
def init_command(
    input_structure: Path = typer.Argument(..., help="Input PDB/mmCIF file."),
    output: Path = typer.Option(Path("system.yaml"), "-o", "--output", help="Output YAML path."),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        "--force",
        help="Overwrite an existing output manifest.",
    ),
    forcefield: str = typer.Option("ff14SB", "--forcefield", help="Protein force field."),
    water_model: str = typer.Option("TIP3P", "--water-model", help="Water model."),
    ph: float = typer.Option(7.0, "--ph", help="Target pH for starter manifest defaults."),
    protonation_method: str = typer.Option(
        "manual_only",
        "--protonation-method",
        help="Initial protonation method: manual_only, propka, or propka_xtb_his.",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Preparation output directory. Defaults to prepared/<input stem>.",
    ),
    include_ligand_placeholders: bool = typer.Option(
        False,
        "--include-ligand-placeholders",
        help="Add active placeholder ligand blocks for detected ligands. Review net charges before use.",
    ),
    interactive: bool = typer.Option(
        False,
        "-i",
        "--interactive",
        help="Ask only the questions this structure needs, then write a validated manifest.",
    ),
) -> None:
    """Create an initial manifest from an input structure."""

    if interactive:
        _run_interactive_init(
            input_structure,
            output=output,
            overwrite=overwrite,
            output_dir=output_dir,
        )
        return

    if forcefield not in {"ff14SB", "ff19SB"}:
        console.print("[red]Error:[/red] --forcefield must be ff14SB or ff19SB")
        raise typer.Exit(1)
    if water_model not in {"TIP3P", "OPC"}:
        console.print("[red]Error:[/red] --water-model must be TIP3P or OPC")
        raise typer.Exit(1)
    if protonation_method not in {"manual_only", "propka", "propka_xtb_his"}:
        console.print("[red]Error:[/red] --protonation-method must be manual_only, propka, or propka_xtb_his")
        raise typer.Exit(1)
    try:
        manifest_path = generate_starter_manifest(
            input_structure,
            output_path=output,
            overwrite=overwrite,
            forcefield=forcefield,
            water_model=water_model,
            ph=ph,
            output_dir=output_dir,
            protonation_method=protonation_method,
            include_ligand_placeholders=include_ligand_placeholders,
        )
    except (FileExistsError, FileNotFoundError, PdbParseError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(f"Wrote starter manifest: {manifest_path}")


def _run_interactive_init(
    input_structure: Path,
    *,
    output: Path,
    overwrite: bool,
    output_dir: str | None,
) -> None:
    preset: dict[str, object] = {}
    if output_dir is not None:
        preset["project.output_dir"] = output_dir

    def prompt(decision) -> str:
        console.print(format_question(decision))
        return typer.prompt("  >", default="", show_default=False)

    try:
        result = run_wizard(
            input_structure,
            output_path=output,
            prompt=prompt,
            overwrite=overwrite,
            preset=preset,
        )
    except (FileExistsError, FileNotFoundError, PdbParseError, DecisionError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        f"\nWrote validated manifest: {result.manifest_path} "
        f"({result.questions_asked} questions answered)"
    )
    for site in result.mcpb_sites_needing_manual_work:
        console.print(
            f"[yellow]Metal site {site!r} needs a bonded MCPB block that this wizard "
            "cannot generate.[/yellow] Add it by hand following docs/metals.md."
        )
    console.print("Check it with: [bold]mdprep config-check "
                  f"{result.manifest_path}[/bold]")


@app.command("prepare")
def prepare_command(
    manifest: Path = typer.Argument(..., help="YAML manifest path."),
    stop_after: str | None = typer.Option(
        None,
        "--stop-after",
        help=(
            "Stop after a supported workflow stage: structure, protonation, refinement, "
            "ligands, metals, tleap, or md."
        ),
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite mdprep-generated outputs."),
    resume: bool = typer.Option(
        False,
        "--resume",
        help=(
            "Resume from a matching converged ASH refinement after an interrupted "
            "downstream stage."
        ),
    ),
    quiet: bool = typer.Option(False, "--quiet", help="Reduce command output."),
) -> None:
    """Prepare an Amber system from a manifest."""

    try:
        result = prepare_system(
            manifest,
            stop_after=stop_after,
            overwrite=overwrite,
            resume=resume,
        )
    except (
        PrepareWorkflowError,
        StructureNormalizationError,
        FileExistsError,
        FileNotFoundError,
        PdbParseError,
        ValueError,
    ) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    if not quiet:
        console.print(f"Wrote output PDB: {result.output_path}")


@app.command("validate")
def validate_command(
    prmtop: Path = typer.Argument(..., help="Amber topology file."),
    inpcrd: Path = typer.Argument(..., help="Amber coordinate file."),
) -> None:
    """Validate a prepared Amber system."""

    try:
        validate_system(prmtop, inpcrd)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print("Validation passed")


@app.command("run-md")
def run_md_command(
    manifest_path: Path = typer.Argument(..., help="Manifest containing molecular_dynamics settings."),
    prmtop: Path | None = typer.Option(
        None,
        "--prmtop",
        help="Amber topology; defaults to <project.output_dir>/final/system.prmtop.",
    ),
    inpcrd: Path | None = typer.Option(
        None,
        "--inpcrd",
        help="Amber coordinates; defaults to <project.output_dir>/final/system.inpcrd.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="MD output directory; defaults to <project.output_dir>/md/roe_brooks_2020.",
    ),
) -> None:
    """Run manifest-configured Roe--Brooks MD from an existing Amber system."""

    try:
        manifest = load_manifest(manifest_path, resolve_paths=True)
        project_output = Path(manifest.project.output_dir)
        result = run_roe_brooks(
            prmtop_path=prmtop or project_output / "final" / "system.prmtop",
            inpcrd_path=inpcrd or project_output / "final" / "system.inpcrd",
            output_dir=(
                output_dir
                or project_output / "md" / "roe_brooks_2020"
            ),
            config=manifest.molecular_dynamics,
        )
    except (FileNotFoundError, RoeBrooksError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    console.print(
        f"MD completed on {result.platform}; final coordinates: {result.final_pdb}"
    )


@app.command("compare-mcpb")
def compare_mcpb_command(
    candidate_frcmod: Path = typer.Argument(
        ..., help="Candidate MCPB.py-generated frcmod file."
    ),
    reference_frcmod: Path = typer.Argument(
        ..., help="Selected reference MCPB.py-generated frcmod file."
    ),
    output_dir: Path = typer.Option(
        Path("mcpb_parameter_comparison"),
        "--output-dir",
        help="Directory for JSON, CSV, and Markdown comparison reports.",
    ),
    candidate_label: str = typer.Option("xTB MCPB", "--candidate-label"),
    reference_label: str = typer.Option("MCPB reference", "--reference-label"),
    require_exact_term_set: bool = typer.Option(
        True,
        "--require-exact-term-set/--allow-term-set-difference",
        help="Require candidate and reference to contain identical MCPB Seminario terms.",
    ),
    max_bond_relative_rmse: float | None = typer.Option(
        None, "--max-bond-relative-rmse"
    ),
    max_angle_relative_rmse: float | None = typer.Option(
        None, "--max-angle-relative-rmse"
    ),
    max_bond_distance_rmse: float | None = typer.Option(
        None, "--max-bond-distance-rmse"
    ),
    max_angle_value_rmse: float | None = typer.Option(
        None, "--max-angle-value-rmse"
    ),
    report_only: bool = typer.Option(
        False,
        "--report-only",
        help="Write a failing report without returning a nonzero exit status.",
    ),
) -> None:
    """Compare MCPB Seminario bond/angle terms against a selected reference."""

    try:
        config = McpbParameterComparisonConfig(
            reference_frcmod=str(reference_frcmod),
            candidate_label=candidate_label,
            reference_label=reference_label,
            require_exact_term_set=require_exact_term_set,
            max_bond_force_constant_relative_rmse=max_bond_relative_rmse,
            max_angle_force_constant_relative_rmse=max_angle_relative_rmse,
            max_bond_equilibrium_distance_rmse_angstrom=max_bond_distance_rmse,
            max_angle_equilibrium_value_rmse_degrees=max_angle_value_rmse,
            fail_on_thresholds=not report_only,
        )
        result = compare_mcpb_parameter_files(
            candidate_frcmod,
            config=config,
            output_dir=output_dir,
        )
    except (McpbParameterComparisonError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    status = {
        "pass": "PASS",
        "fail": "FAIL (report only)",
        "report_only": "REPORT ONLY (no numeric acceptance thresholds)",
    }[result.status]
    console.print(f"MCPB parameter comparison: {status}")
    console.print(f"Wrote report: {result.json_path}")


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
