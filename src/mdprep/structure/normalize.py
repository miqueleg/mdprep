"""Safe structure-only normalization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from pathlib import Path

from mdprep.config.models import LigandConfig, ManifestConfig
from mdprep.residues.structure import (
    KeptResidueError,
    describe_unconfigured,
    prepare_noncanonical_residues,
    resolve_noncanonical_residues,
    unconfigured_modified_residues,
)
from mdprep.structure.classify import (
    is_bulk_ion_residue,
    is_likely_ligand_or_cofactor,
    is_nonstandard_nonwater_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.coordination import metal_bound_waters
from mdprep.structure.hydrogens import (
    InputHydrogenRemoval,
    is_hydrogen,
    remove_input_hydrogens,
)
from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.pdb import ATOMIC_NUMBER, read_pdb
from mdprep.structure.reader import read_structure
from mdprep.structure.repair import (
    StructureRepairError,
    StructureRepairResult,
    repair_structure,
)
from mdprep.structure.selectors import SelectorError, resolve_atom_selector, resolve_residue_selector


class StructureNormalizationError(ValueError):
    """Raised when structure normalization would require an unsafe decision."""


@dataclass(frozen=True)
class NormalizedLigand:
    ligand_id: str
    residue: dict[str, object]


@dataclass(frozen=True)
class NormalizedMetalIon:
    site_id: str
    element: str
    charge: int
    atom: dict[str, object]


@dataclass
class StructureNormalizationResult:
    input_path: Path
    output_path: Path | None
    input_summary: InspectionSummary
    normalized_structure: PdbStructure
    waters_kept: list[dict[str, object]] = field(default_factory=list)
    waters_removed: list[dict[str, object]] = field(default_factory=list)
    crystal_ions_kept: list[dict[str, object]] = field(default_factory=list)
    crystal_ions_removed: list[dict[str, object]] = field(default_factory=list)
    configured_ligands_kept: list[NormalizedLigand] = field(default_factory=list)
    configured_metal_ions_kept: list[NormalizedMetalIon] = field(default_factory=list)
    unknown_heterogens_removed: list[dict[str, object]] = field(default_factory=list)
    residues_removed: list[dict[str, object]] = field(default_factory=list)
    chains_removed: list[dict[str, object]] = field(default_factory=list)
    unknown_heterogens_causing_failure: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repair_result: StructureRepairResult | None = None
    input_hydrogens: InputHydrogenRemoval = field(default_factory=InputHydrogenRemoval)
    # Each ligand with a SMILES: the hydrogens placed from it, or the input
    # hydrogens it kept and how they compared with it.
    ligand_hydrogens: list[dict[str, object]] = field(default_factory=list)
    # Each residue noncanonical_residues keeps: its name in the build, and
    # the hydrogens it was given or left to tleap.
    noncanonical_residues_kept: list[dict[str, object]] = field(default_factory=list)

    def to_report_dict(self) -> dict[str, object]:
        normalized_summary = inspect_pdb_structure_from_structure(self.normalized_structure)
        return {
            "input_path": str(self.input_path),
            "output_normalized_pdb_path": str(self.output_path) if self.output_path else None,
            "atom_count_before": len(self.input_summary.structure.atoms),
            "atom_count_after": len(self.normalized_structure.atoms),
            "residue_count_before": len(self.input_summary.structure.residues),
            "residue_count_after": len(self.normalized_structure.residues),
            "chains": [{"chain_id": chain, "display": chain or "<blank>"} for chain in normalized_summary.chain_ids],
            "waters_kept": self.waters_kept,
            "waters_removed": self.waters_removed,
            "crystal_ions_kept": self.crystal_ions_kept,
            "crystal_ions_removed": self.crystal_ions_removed,
            "configured_ligands_kept": [
                {"id": ligand.ligand_id, **ligand.residue} for ligand in self.configured_ligands_kept
            ],
            "configured_metal_ions_kept": [
                {
                    "site_id": metal.site_id,
                    "element": metal.element,
                    "charge": metal.charge,
                    **metal.atom,
                }
                for metal in self.configured_metal_ions_kept
            ],
            "unknown_heterogens_removed": self.unknown_heterogens_removed,
            "residues_removed": self.residues_removed,
            "chains_removed": self.chains_removed,
            "unknown_heterogens_causing_failure": self.unknown_heterogens_causing_failure,
            "input_hydrogens": self.input_hydrogens.to_dict(),
            "ligand_hydrogens": self.ligand_hydrogens,
            "noncanonical_residues_kept": self.noncanonical_residues_kept,
            "histidines": _residue_dicts(self.input_summary.histidines),
            "titratable_residues": _residue_dicts(self.input_summary.titratable_residues),
            "possible_disulfides": [
                candidate.to_dict() for candidate in self.input_summary.possible_disulfides
            ],
            "structure_repair": (
                self.repair_result.to_dict()
                if self.repair_result is not None
                else None
            ),
            "warnings": self.warnings,
        }


def normalize_structure_stage(
    manifest: ManifestConfig,
    *,
    output_path: str | Path | None = None,
) -> StructureNormalizationResult:
    input_path = Path(manifest.project.input_structure)
    input_summary = inspect_pdb_structure(
        input_path,
        altloc_policy=manifest.structure.altloc_policy,
        disulfide_cutoff_angstrom=manifest.disulfides.detection_cutoff_angstrom,
    )
    # Hydrogens the input arrived with go before anything looks at the
    # structure: every stage after this one places its own, and the ones a
    # file brings were placed by another program under its own names and
    # its own idea of the protonation. Counted on the whole input, so the
    # record does not depend on whether PDBFixer read it first.
    _, input_hydrogens = remove_input_hydrogens(input_summary.structure, manifest)
    repair_result: StructureRepairResult | None = None
    if manifest.structure.repair.backend != "none":
        if output_path is None:
            raise StructureNormalizationError(
                "Structure repair requires an explicit normalization output path."
            )
        normalized_target = Path(output_path)
        repair_path = normalized_target.with_name("00_structure_repaired.pdb")
        try:
            repair_result = repair_structure(manifest, output_path=repair_path)
        except StructureRepairError as exc:
            raise StructureNormalizationError(str(exc)) from exc
        assert repair_result is not None
        structure = repair_result.structure
    else:
        structure, _ = remove_input_hydrogens(
            read_structure(input_path, altloc_policy=manifest.structure.altloc_policy),
            manifest,
        )

    # Residues the manifest leaves out on purpose go first, so none of them
    # is mistaken below for an unknown heterogen, a ligand or a donor.
    removed = _resolve_removed_residues(structure, manifest)
    configured_ligands = _resolve_configured_ligands(structure, manifest)
    try:
        configured_residues = resolve_noncanonical_residues(structure, manifest)
    except KeptResidueError as exc:
        raise StructureNormalizationError(str(exc)) from exc
    configured_metals = _resolve_configured_metals(structure, manifest)
    _validate_disulfide_selectors(structure, manifest)
    configured_ligand_keys = {id(residue) for _, residue in configured_ligands}
    configured_metal_ion_keys = {id(residue) for _, _, residue in configured_metals}
    # Being both a ligand and a metal ion is a contradiction. Being both a
    # ligand and a coordinator is not: alpha-ketoglutarate needs GAFF
    # parameters and also bonds the iron, so the two sets are kept apart.
    overlap = configured_ligand_keys & configured_metal_ion_keys
    if overlap:
        residues = [residue for residue in structure.residues if id(residue) in overlap]
        formatted = ", ".join(_format_residue_for_error(residue) for residue in residues)
        raise StructureNormalizationError(
            "A residue cannot be configured as both a ligand and a metal ion: " + formatted
        )
    # A residue named as an MCPB coordinator or additional residue is part of
    # the metal site and is parameterized with it, so it is not an unknown
    # heterogen. A coordinating halide is the case that matters: it cannot be a
    # metal ion, because an ion's charge is positive by definition, and it
    # cannot be a GAFF ligand either, because antechamber's semi-empirical step
    # refuses a single atom.
    configured_metal_residue_keys = configured_metal_ion_keys | _mcpb_model_residue_keys(
        structure, manifest
    )
    # A water in the first shell of a metal the build keeps -- bonded or not
    # -- is one of its ligands. The crystal-water question promises it stays
    # with its metal whatever the answer, and this is the geometry that
    # promise was made from.
    metal_bound_water_keys = {
        id(residue)
        for residue in metal_bound_waters(
            structure.residues, [residue for _, _, residue in configured_metals]
        )
    }
    # Only the chains the build keeps. A deposition of four copies has four
    # sets of cofactors, and demanding a declaration for the three about to be
    # discarded would reject a manifest that is already complete. The repair
    # stage filters chains before PDBFixer sees them, so this check used to
    # look correct whenever repair was enabled, and wrong when it was not.
    kept_chains = {chain.strip() for chain in manifest.structure.keep_chains}
    likely_heterogens = [
        residue
        for residue in structure.residues
        if is_likely_ligand_or_cofactor(residue)
        and id(residue) not in removed
        and (not kept_chains or residue.id.chain_id in kept_chains)
    ]
    # A modified residue of a chain is not a heterogen to remove: it has a
    # peptide bond on each side. One the manifest does not keep, replace or
    # remove stops the build, whatever remove_unknown_heterogens says.
    unsettled = unconfigured_modified_residues(
        structure,
        settled=set(configured_residues) | set(removed),
        kept_chains=kept_chains,
    )
    if unsettled:
        raise StructureNormalizationError(describe_unconfigured(unsettled))
    unknown_heterogens = [
        residue
        for residue in likely_heterogens
        if id(residue) not in configured_ligand_keys
        and id(residue) not in configured_metal_residue_keys
        and id(residue) not in configured_residues
    ]

    unknown_heterogen_dicts = _residue_dicts(unknown_heterogens)
    if unknown_heterogens and not manifest.structure.remove_unknown_heterogens:
        formatted = ", ".join(_format_residue_for_error(residue) for residue in unknown_heterogens)
        raise StructureNormalizationError(
            "Unknown heterogens are present but structure.remove_unknown_heterogens is false: "
            f"{formatted}. Add them to ligands: for molecular heterogens or metals: for metal ions, "
            "or set structure.remove_unknown_heterogens: true."
        )

    keep_residue_ids: set[int] = set()
    waters_kept: list[ResidueRecord] = []
    waters_removed: list[ResidueRecord] = []
    ions_kept: list[ResidueRecord] = []
    ions_removed: list[ResidueRecord] = []
    unknown_removed: list[ResidueRecord] = []
    residues_removed: list[dict[str, object]] = []
    chains_removed: list[ResidueRecord] = []

    selected_chains = {chain.strip() for chain in manifest.structure.keep_chains}
    if selected_chains:
        present = {residue.id.chain_id for residue in structure.residues}
        missing = sorted(selected_chains - present)
        if missing:
            raise StructureNormalizationError(
                f"structure.keep_chains names chains not present in the input: "
                f"{missing}; the input has {sorted(present)}."
            )

    for residue in structure.residues:
        if selected_chains and residue.id.chain_id not in selected_chains:
            chains_removed.append(residue)
            continue
        if id(residue) in removed:
            residues_removed.append({**_residue_dict(residue), "reason": removed[id(residue)]})
            continue
        if is_standard_protein_residue(residue) or id(residue) in configured_residues:
            keep_residue_ids.add(id(residue))
        elif id(residue) in configured_metal_residue_keys or id(residue) in metal_bound_water_keys:
            # Before the water and ion branches, on purpose. A water bound to
            # a bonded iron is one of its MCPB coordinators: dropping it with
            # the other crystal waters would leave a bond pointing at a
            # residue that no longer exists, and that only surfaces at the
            # MCPB step, after protonation and the ligand parameters. A water
            # on a nonbonded ion is its first shell all the same. The same
            # holds for a coordinating halide among the crystal ions.
            keep_residue_ids.add(id(residue))
            if is_water_residue(residue):
                waters_kept.append(residue)
        elif is_water_residue(residue):
            if manifest.structure.keep_crystal_waters:
                keep_residue_ids.add(id(residue))
                waters_kept.append(residue)
            else:
                waters_removed.append(residue)
        elif id(residue) in configured_ligand_keys:
            keep_residue_ids.add(id(residue))
        elif is_bulk_ion_residue(residue):
            if manifest.structure.keep_crystal_ions:
                keep_residue_ids.add(id(residue))
                ions_kept.append(residue)
            else:
                ions_removed.append(residue)
        elif residue in unknown_heterogens:
            unknown_removed.append(residue)
        else:
            keep_residue_ids.add(id(residue))

    filtered_atoms: list[AtomRecord] = []
    for residue in structure.residues:
        if id(residue) in keep_residue_ids:
            filtered_atoms.extend(residue.atoms)

    # Ligand hydrogens belong to the structure, not just to the ligand's own
    # file: the QM/MM refinement, the leap build and the ligand parameters all
    # have to see the same molecule. Placing them here is what keeps them
    # consistent.
    filtered_atoms, hydrogenations, hydrogenation_warnings, rewritten = (
        _hydrogenate_configured_ligands(structure, manifest, keep_residue_ids, filtered_atoms)
    )
    # A kept noncanonical residue is built as a residue of its chain, under
    # its template's name or with its SMILES's hydrogens.
    try:
        residue_atoms, kept_noncanonical, residue_warnings = prepare_noncanonical_residues(
            [residue for residue in structure.residues if id(residue) in keep_residue_ids],
            configured_residues,
        )
    except KeptResidueError as exc:
        raise StructureNormalizationError(str(exc)) from exc
    if residue_atoms:
        filtered_atoms = _splice(structure, filtered_atoms, residue_atoms)
        rewritten = {**rewritten, **residue_atoms}
    hydrogenation_warnings = hydrogenation_warnings + residue_warnings
    # A ligand whose hydrogens were placed or replaced is a new residue as
    # well as new atoms. Keeping the old record let the two disagree: the
    # temporary hydrogenation for xTB checks every atom of every residue it is
    # handed, and found a replaced hydrogen two angstrom from where the stale
    # residue still had it.
    kept_residues = [
        ResidueRecord(
            id=replace(residue.id, resname=rewritten[id(residue)][0].resname),
            atoms=rewritten[id(residue)],
            record_names={atom.record_name for atom in rewritten[id(residue)]},
            original_index=residue.original_index,
        )
        if id(residue) in rewritten
        else residue
        for residue in _filter_residues_by_ids(structure.residues, keep_residue_ids)
    ]

    normalized_structure = PdbStructure(
        path=Path(output_path) if output_path is not None else structure.path,
        atoms=filtered_atoms,
        residues=kept_residues,
        model_count=structure.model_count,
        used_model=structure.used_model,
        warnings=list(structure.warnings),
        conect_bonds={
            bond
            for bond in structure.conect_bonds
            if bond[0] in {atom.serial for atom in filtered_atoms}
            and bond[1] in {atom.serial for atom in filtered_atoms}
        },
        ter_after_serials={
            serial
            for serial in structure.ter_after_serials
            if serial in {atom.serial for atom in filtered_atoms}
        },
    )

    warnings = list(structure.warnings)
    # What the reader said about the input itself -- alternate locations it
    # had to label, serials it renumbered -- is said once, by the read of the
    # input. PDBFixer's copy no longer has the problem to report.
    warnings.extend(
        warning for warning in input_summary.structure.warnings if warning not in warnings
    )
    if repair_result is not None:
        warnings.extend(repair_result.warnings)
    warnings.extend(hydrogenation_warnings)

    return StructureNormalizationResult(
        input_path=input_path,
        output_path=Path(output_path) if output_path is not None else None,
        input_summary=input_summary,
        normalized_structure=normalized_structure,
        waters_kept=_residue_dicts(waters_kept),
        waters_removed=_residue_dicts(waters_removed),
        crystal_ions_kept=_residue_dicts(ions_kept),
        crystal_ions_removed=_residue_dicts(ions_removed),
        configured_ligands_kept=[
            NormalizedLigand(ligand_id=ligand_id, residue=_residue_dict(residue))
            for ligand_id, residue in configured_ligands
        ],
        configured_metal_ions_kept=[
            NormalizedMetalIon(
                site_id=site_id,
                element=element,
                charge=charge,
                atom={
                    **residue.id.to_dict(),
                    "atom_name": atom.name,
                    "atom_serial": atom.serial,
                },
            )
            for site_id, (element, charge, atom), residue in configured_metals
        ],
        unknown_heterogens_removed=_residue_dicts(unknown_removed),
        residues_removed=residues_removed,
        chains_removed=_residue_dicts(chains_removed),
        unknown_heterogens_causing_failure=unknown_heterogen_dicts
        if unknown_heterogens and not manifest.structure.remove_unknown_heterogens
        else [],
        warnings=warnings,
        repair_result=repair_result,
        input_hydrogens=input_hydrogens,
        ligand_hydrogens=hydrogenations,
        noncanonical_residues_kept=[item.to_dict() for item in kept_noncanonical],
    )


def inspect_pdb_structure_from_structure(structure: PdbStructure) -> InspectionSummary:
    from mdprep.structure.classify import is_histidine, is_titratable_residue
    from mdprep.structure.disulfides import detect_possible_disulfides

    residues = structure.residues
    protein_residues = [residue for residue in residues if is_standard_protein_residue(residue)]
    water_residues = [residue for residue in residues if is_water_residue(residue)]
    heterogen_residues = [
        residue for residue in residues if is_nonstandard_nonwater_residue(residue)
    ]
    likely_ligands = [residue for residue in residues if is_likely_ligand_or_cofactor(residue)]
    histidines = [residue for residue in residues if is_histidine(residue)]
    titratable = [residue for residue in residues if is_titratable_residue(residue)]
    disulfides = detect_possible_disulfides(residues)
    return InspectionSummary(
        structure=structure,
        protein_residues=protein_residues,
        water_residues=water_residues,
        heterogen_residues=heterogen_residues,
        likely_ligands=likely_ligands,
        histidines=histidines,
        titratable_residues=titratable,
        possible_disulfides=disulfides,
    )


def _resolve_removed_residues(
    structure: PdbStructure, manifest: ManifestConfig
) -> dict[int, str]:
    """Each residue structure.remove_residues names, with its reason."""

    removed: dict[int, str] = {}
    for item in manifest.structure.remove_residues:
        try:
            residue = resolve_residue_selector(structure, item.selector.model_dump())
        except SelectorError as exc:
            raise StructureNormalizationError(
                f"structure.remove_residues names a residue the structure does not "
                f"have: {exc}"
            ) from exc
        removed[id(residue)] = item.reason
    return removed


def _resolve_configured_ligands(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[tuple[str, ResidueRecord]]:
    resolved: list[tuple[str, ResidueRecord]] = []
    for ligand in manifest.ligands:
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError as exc:
            raise StructureNormalizationError(
                f"Ligand {ligand.id!r} selector did not resolve exactly one residue: {exc}"
            ) from exc
        resolved.append((ligand.id, residue))
    return resolved


def _hydrogenate_configured_ligands(
    structure: PdbStructure,
    manifest: ManifestConfig,
    keep_residue_ids: set[int],
    filtered_atoms: list[AtomRecord],
) -> tuple[list[AtomRecord], list[dict[str, object]], list[str], dict[int, list[AtomRecord]]]:
    """Give every configured ligand the hydrogens its SMILES states.

    Hydrogens a ligand arrives with are compared with its SMILES and kept or
    replaced (see add_ligand_hydrogens), so a ligand that has them needs a
    SMILES as much as one that does not: without it they could only be
    trusted, and a docking program's polar hydrogens were trusted all the way
    to antechamber. A user mol2 is the exception, being a statement of the
    molecule of its own. A ligand whose charges mdprep computes from its
    SMILES must then have an electron count its multiplicity allows.

    Returns the atoms, the outcomes, the warnings, and the new atoms of each
    ligand residue whose hydrogens changed, keyed by the old residue's id.
    """

    from mdprep.ligands.hydrogenate import (
        LigandHydrogenationError,
        add_ligand_hydrogens,
    )

    results: list[dict[str, object]] = []
    warnings: list[str] = []
    replacements: dict[int, list[AtomRecord]] = {}
    for ligand in manifest.ligands:
        try:
            residue = resolve_residue_selector(structure, ligand.selector.model_dump())
        except SelectorError:
            # The ligand stage reports an unresolvable selector in full.
            continue
        if id(residue) not in keep_residue_ids:
            continue
        if ligand.smiles is None:
            _require_a_smiles(ligand, residue)
        else:
            try:
                outcome = add_ligand_hydrogens(
                    residue,
                    ligand.smiles,
                    ligand_id=ligand.id,
                    net_charge=ligand.net_charge,
                    # A user mol2 lists the residue's atoms, so its hydrogens
                    # can be checked but not exchanged.
                    keep_input_hydrogens=(
                        ligand.keep_input_hydrogens
                        if ligand.keep_input_hydrogens is not None
                        else (True if ligand.user_mol2 is not None else None)
                    ),
                )
            except LigandHydrogenationError as exc:
                raise StructureNormalizationError(str(exc)) from exc
            results.append({"id": ligand.id, **outcome.to_dict()})
            if outcome.replaced_hydrogens:
                warnings.extend(outcome.warnings)
            if not outcome.skipped:
                replacements[id(residue)] = list(outcome.residue.atoms)
            if ligand.user_mol2 is None:
                _check_electron_count(ligand, residue, outcome.residue.atoms)

    if not replacements:
        return filtered_atoms, results, warnings, replacements
    return _splice(structure, filtered_atoms, replacements), results, warnings, replacements


def _splice(
    structure: PdbStructure,
    filtered_atoms: list[AtomRecord],
    replacements: dict[int, list[AtomRecord]],
) -> list[AtomRecord]:
    """``filtered_atoms`` with each replaced residue's atoms swapped for its new ones.

    Each rewritten residue takes the place of the residue it replaces, so
    residues stay contiguous and in order in the output.
    """

    owner = {
        id(atom): id(residue)
        for residue in structure.residues
        if id(residue) in replacements
        for atom in residue.atoms
    }
    pending = dict(replacements)
    rebuilt: list[AtomRecord] = []
    for atom in filtered_atoms:
        residue_key = owner.get(id(atom))
        if residue_key is None:
            rebuilt.append(atom)
        elif residue_key in pending:
            rebuilt.extend(pending.pop(residue_key))
    return rebuilt


def _require_a_smiles(ligand: LigandConfig, residue: ResidueRecord) -> None:
    """Refuse a ligand whose protonation nothing would state or check."""

    if ligand.user_mol2 is not None:
        return
    hydrogens = sum(1 for atom in residue.atoms if is_hydrogen(atom))
    if hydrogens:
        raise StructureNormalizationError(
            f"Ligand {ligand.id} ({residue.id.display()}) arrives with {hydrogens} "
            "hydrogen(s) and no SMILES to check them against. Whatever wrote the "
            "file put them there -- a docking program often writes only the polar "
            "ones -- and mdprep takes a ligand's protonation from its SMILES, "
            "comparing these with it: kept if they agree, replaced if not. Add "
            "ligands[].smiles for it (the input builder asks for it)."
        )
    declared = ligand.expected_formula
    if declared is not None and "H" not in re.findall(r"[A-Z][a-z]?", declared):
        # Declared hydrogen-free, which is a statement of its own.
        return
    raise StructureNormalizationError(
        f"Ligand {ligand.id} ({residue.id.display()}) has no hydrogens in the "
        "input and no SMILES to place them from. mdprep will not guess a "
        "ligand's protonation -- a carboxylate and a carboxylic acid are the same "
        "heavy atoms -- so add ligands[].smiles for it (the input builder asks "
        "for it), or declare a hydrogen-free molecule with expected_formula."
    )


def _check_electron_count(
    ligand: LigandConfig, residue: ResidueRecord, atoms: list[AtomRecord]
) -> None:
    """The ligand's electrons must be a number its charge and spin allow.

    Electrons are counted from the atoms and the declared charge. A closed
    shell has an even number of them, a doublet an odd one, and so on; a
    count of the wrong parity is a molecule that cannot exist, which is what
    antechamber was handed when a docked substrate kept two of its hydrogens.
    It said so only as an aside, before its semi-empirical step failed.
    """

    numbers = [_atomic_number(atom) for atom in atoms]
    if any(number is None for number in numbers):
        return
    electrons = sum(numbers) - ligand.net_charge  # type: ignore[arg-type]
    unpaired = ligand.multiplicity - 1
    if electrons >= unpaired and (electrons - unpaired) % 2 == 0:
        return
    hydrogens = sum(1 for atom in atoms if is_hydrogen(atom))
    raise StructureNormalizationError(
        f"Ligand {ligand.id} ({residue.id.display()}) has {electrons} electrons "
        f"at net charge {ligand.net_charge:+d}, with {hydrogens} hydrogen(s), "
        f"which multiplicity {ligand.multiplicity} cannot have. Its hydrogens, "
        "its net charge or its multiplicity is wrong."
    )


def _atomic_number(atom: AtomRecord) -> int | None:
    if is_hydrogen(atom):
        return 1
    return ATOMIC_NUMBER.get((atom.element or "").strip().upper())


def _mcpb_model_residue_keys(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> set[int]:
    """Residues drawn into a bonded MCPB model besides the metal itself."""

    keys: set[int] = set()
    for site in manifest.metals:
        # A coordinating ion is part of the site under either metal model.
        for coordinating in site.coordinating_ions:
            selector = coordinating.selector.model_dump()
            selector.pop("atom_name", None)
            try:
                keys.add(id(resolve_residue_selector(structure, selector)))
            except SelectorError:
                continue
        if site.model != "bonded_mcpb" or site.mcpb is None:
            continue
        selectors = [bond.coordinator.model_dump() for bond in site.mcpb.bonds]
        selectors += [item.model_dump() for item in site.mcpb.additional_residues]
        for selector in selectors:
            selector.pop("atom_name", None)
            try:
                residue = resolve_residue_selector(structure, selector)
            except SelectorError:
                # An unresolvable selector is reported by the metal stage,
                # which can say far more about it than this can.
                continue
            keys.add(id(residue))
    return keys


def _resolve_configured_metals(
    structure: PdbStructure,
    manifest: ManifestConfig,
) -> list[tuple[str, tuple[str, int, AtomRecord], ResidueRecord]]:
    resolved: list[tuple[str, tuple[str, int, AtomRecord], ResidueRecord]] = []
    seen_atoms: set[int] = set()
    for site in manifest.metals:
        for ion in site.ions:
            try:
                atom = resolve_atom_selector(structure, ion.selector.model_dump())
                residue = resolve_residue_selector(structure, ion.selector.model_dump())
            except SelectorError as exc:
                raise StructureNormalizationError(
                    f"Metal site {site.id!r} ion selector did not resolve exactly one atom: {exc}"
                ) from exc
            if id(atom) in seen_atoms:
                raise StructureNormalizationError(
                    f"Metal atom {atom.atom_identity} is configured more than once."
                )
            seen_atoms.add(id(atom))
            observed_element = (atom.element or "").capitalize()
            if observed_element and observed_element != ion.element:
                raise StructureNormalizationError(
                    f"Metal site {site.id!r} declares element {ion.element} for "
                    f"{atom.atom_identity}, but the PDB element is {observed_element}."
                )
            if len(residue.atoms) != 1:
                raise StructureNormalizationError(
                    f"Metal ion residue {residue.id.display()} contains {len(residue.atoms)} atoms; "
                    "each configured metal ion must occupy a one-atom residue."
                )
            resolved.append((site.id, (ion.element, ion.charge, atom), residue))
    return resolved


def _validate_disulfide_selectors(structure: PdbStructure, manifest: ManifestConfig) -> None:
    for group_name, pairs in (
        ("force", manifest.disulfides.force),
        ("forbid", manifest.disulfides.forbid),
    ):
        for index, pair in enumerate(pairs, start=1):
            for side_name, selector in (("a", pair.a), ("b", pair.b)):
                try:
                    resolve_residue_selector(structure, selector.model_dump())
                except SelectorError as exc:
                    raise StructureNormalizationError(
                        f"Disulfide {group_name}[{index}].{side_name} selector did not resolve exactly one residue: {exc}"
                    ) from exc


def _filter_residues_by_ids(
    residues: list[ResidueRecord],
    keep_residue_ids: set[int],
) -> list[ResidueRecord]:
    return [residue for residue in residues if id(residue) in keep_residue_ids]


def _residue_dicts(residues: list[ResidueRecord]) -> list[dict[str, object]]:
    return [_residue_dict(residue) for residue in residues]


def _residue_dict(residue: ResidueRecord) -> dict[str, object]:
    return {
        **residue.id.to_dict(),
        "atom_count": len(residue.atoms),
        "record_names": sorted(residue.record_names),
        "original_index": residue.original_index,
    }


def _format_residue_for_error(residue: ResidueRecord) -> str:
    chain = residue.id.chain_id if residue.id.chain_id else "<blank>"
    icode = residue.id.icode or ""
    return f"{chain}:{residue.id.resname}{residue.id.resid}{icode}"
