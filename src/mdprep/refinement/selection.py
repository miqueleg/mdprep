"""Deterministic QM and active-region selection for QM/MM refinement."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from mdprep.config.models import ManifestConfig, ResidueSelector
from mdprep.metals.coordination import MetalCoordinationError, resolve_metal_sites
from mdprep.refinement.mapping import (
    MappedResidue,
    RefinementMappingError,
    is_virtual_site,
    map_residues_by_coordinates,
)
from mdprep.residues.workflow import kept_residue_net_charges
from mdprep.structure.classify import (
    is_bulk_ion_residue,
    is_nonstandard_nonwater_residue,
    is_standard_protein_residue,
    is_water_residue,
)
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


class RefinementSelectionError(ValueError):
    """Raised when the requested QM/MM regions cannot be mapped unambiguously."""


@dataclass(frozen=True)
class ChargeComponent:
    label: str
    kind: str
    charge: int

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "kind": self.kind, "charge": self.charge}


@dataclass(frozen=True)
class RefinementSelection:
    qm_atom_indices: tuple[int, ...]
    active_atom_indices: tuple[int, ...]
    qm_residue_indices: tuple[int, ...]
    active_residue_indices: tuple[int, ...]
    qm_residues: tuple[dict[str, object], ...]
    active_residues: tuple[dict[str, object], ...]
    charge_components: tuple[ChargeComponent, ...]
    total_qm_charge: int
    total_qm_multiplicity: int
    topology_atom_count: int
    excluded_virtual_site_indices: tuple[int, ...]
    qm_boundary_excluded_atom_indices: tuple[int, ...] = ()
    active_region_cutoff_angstrom: float = 4.0
    active_water_cutoff_angstrom: float = 8.0
    movable_atom_policy: str = "all_active_region"

    def to_dict(self) -> dict[str, object]:
        return {
            "qm_atom_indices_zero_based": list(self.qm_atom_indices),
            "active_atom_indices_zero_based": list(self.active_atom_indices),
            "qm_residue_indices_zero_based": list(self.qm_residue_indices),
            "active_residue_indices_zero_based": list(self.active_residue_indices),
            "qm_residues": list(self.qm_residues),
            "active_residues": list(self.active_residues),
            "charge_components": [item.to_dict() for item in self.charge_components],
            "total_qm_charge": self.total_qm_charge,
            "total_qm_multiplicity": self.total_qm_multiplicity,
            "topology_atom_count": self.topology_atom_count,
            "excluded_virtual_site_indices_zero_based": list(
                self.excluded_virtual_site_indices
            ),
            "qm_boundary_excluded_atom_indices_zero_based": list(
                self.qm_boundary_excluded_atom_indices
            ),
            "active_region_cutoff_angstrom": self.active_region_cutoff_angstrom,
            "active_water_cutoff_angstrom": self.active_water_cutoff_angstrom,
            "movable_atom_policy": self.movable_atom_policy,
        }


_PROTONATION_FAMILIES = (
    frozenset({"HIS", "HID", "HIE", "HIP"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"CYS", "CYM", "CYX"}),
    frozenset({"LYS", "LYN"}),
    frozenset({"HOH", "WAT", "H2O", "TIP3", "OPC"}),
)

_SIDECHAIN_FORMAL_CHARGES = {
    "ARG": 1,
    "ASP": -1,
    "ASH": 0,
    "GLU": -1,
    "GLH": 0,
    "HIS": 0,
    "HID": 0,
    "HIE": 0,
    "HIP": 1,
    "LYS": 1,
    "LYN": 0,
    "CYS": 0,
    "CYM": -1,
    "CYX": 0,
}


def select_refinement_regions(
    *,
    leap_input_structure: PdbStructure,
    topology_structure: PdbStructure,
    manifest: ManifestConfig,
) -> RefinementSelection:
    """Map configured components onto the provisional Amber atom ordering."""

    if not manifest.refinement.enabled:
        raise RefinementSelectionError("Refinement region selection requires refinement.enabled")
    try:
        pairs = map_residues_by_coordinates(
            leap_input_structure,
            topology_structure,
        )
    except RefinementMappingError as exc:
        raise RefinementSelectionError(str(exc)) from exc
    input_index_by_object = {
        id(residue): index
        for index, residue in enumerate(leap_input_structure.residues)
    }
    atom_index_by_object = {id(atom): index for index, atom in enumerate(topology_structure.atoms)}

    ligand_by_id = {ligand.id: ligand for ligand in manifest.ligands}
    selected_ligand_residues: dict[int, str] = {}
    for ligand_id in manifest.refinement.qm_components.ligands:
        ligand = ligand_by_id[ligand_id]
        try:
            residue = resolve_residue_selector(
                leap_input_structure, ligand.selector.model_dump()
            )
        except SelectorError as exc:
            raise RefinementSelectionError(
                f"Refinement ligand {ligand_id!r} selector failed: {exc}"
            ) from exc
        selected_ligand_residues[input_index_by_object[id(residue)]] = ligand_id

    try:
        resolved_metal_sites = resolve_metal_sites(
            leap_input_structure,
            manifest,
            validate_mcpb_cutoff=False,
        )
    except MetalCoordinationError as exc:
        raise RefinementSelectionError(str(exc)) from exc
    resolved_by_id = {site.config.id: site for site in resolved_metal_sites}
    selected_metal_residues: dict[int, tuple[str, int]] = {}
    selected_coordinating_ions: dict[int, tuple[str, object]] = {}
    coordinator_residue_indices: set[int] = set()
    selected_sites = manifest.refinement.qm_components.metal_sites
    configured_coordinators = (
        manifest.refinement.qm_components.metal_coordinating_residues
    )
    for site_id in selected_sites:
        site = resolved_by_id[site_id]
        for ion_index, ion in enumerate(site.ions):
            residue_index = input_index_by_object[id(ion.residue)]
            selected_metal_residues[residue_index] = (site_id, ion_index)
        for bond in site.bonds:
            coordinator_residue_indices.add(
                input_index_by_object[id(bond.coordinator_residue)]
            )
        # A coordinating ion is declared on the site rather than among its
        # ions, because an ion entry is a metal and a halide is not. It still
        # belongs to the site: it sits in the QM region and brings its own
        # declared charge, so it must not be mistaken for an undeclared
        # heterogen that wandered into the coordination sphere.
        for ion_config in site.config.coordinating_ions:
            # Resolved by residue, not atom: the selector names the atom so
            # the ion can be identified, but it is the whole one-atom residue
            # that enters the QM region. No protonation-aware fallback is
            # needed, since an ion has no alternative residue names.
            try:
                residue = resolve_residue_selector(
                    leap_input_structure,
                    ion_config.selector.model_dump(exclude={"atom_name"}),
                )
            except SelectorError as exc:
                raise RefinementSelectionError(
                    f"Coordinating ion of metal site {site_id} could not be "
                    f"found in the provisional input: {exc}"
                ) from exc
            residue_index = input_index_by_object[id(residue)]
            selected_coordinating_ions[residue_index] = (site_id, ion_config)
            coordinator_residue_indices.add(residue_index)
        for selector in configured_coordinators.get(site_id, []):
            residue = _resolve_protonation_aware(leap_input_structure, selector)
            coordinator_residue_indices.add(input_index_by_object[id(residue)])

    qm_residue_indices = set(selected_ligand_residues)
    qm_residue_indices.update(selected_metal_residues)
    qm_residue_indices.update(selected_coordinating_ions)
    qm_residue_indices.update(coordinator_residue_indices)
    if not qm_residue_indices:
        raise RefinementSelectionError("The configured QM region contains no residues")

    topology_atom_indices_by_residue: list[list[int]] = []
    virtual_sites: set[int] = set()
    for pair in pairs:
        topology_residue = pair.topology_residue
        indices = [atom_index_by_object[id(atom)] for atom in topology_residue.atoms]
        topology_atom_indices_by_residue.append(indices)
        virtual_sites.update(
            index
            for index in indices
            if is_virtual_site(topology_structure.atoms[index])
        )
    _validate_protonated_waters(pairs)

    metal_boundary_exclusions: list[int] = []
    for residue_index, (site_id, ion_index) in selected_metal_residues.items():
        ion = resolved_by_id[site_id].ions[ion_index]
        topology_residue = pairs[residue_index].topology_residue
        candidates = [
            atom
            for atom in topology_residue.atoms
            if atom.name == ion.atom.name
            and (atom.element or "").strip().upper() == ion.element.upper()
        ]
        if len(candidates) != 1:
            raise RefinementSelectionError(
                "Could not map the explicitly selected metal ion uniquely into the "
                f"provisional topology: {ion.atom.atom_identity}"
            )
        metal_boundary_exclusions.append(atom_index_by_object[id(candidates[0])])

    qm_atoms = sorted(
        index
        for residue_index in qm_residue_indices
        for index in topology_atom_indices_by_residue[residue_index]
        if index not in virtual_sites
    )
    if not qm_atoms:
        raise RefinementSelectionError("The configured QM region contains no physical atoms")

    # An ion whose parameters the build already has -- a nonbonded metal
    # site, a crystal ion from Amber's ion library -- is part of the MM
    # environment like any residue of the protein, not an undeclared
    # heterogen. 5OXU's magnesium sits on the heme's propionate.
    kept_charges = kept_residue_net_charges(manifest)
    mm_ions = {
        input_index_by_object[id(ion.residue)]
        for site in resolved_metal_sites
        if site.config.model == "nonbonded"
        for ion in site.ions
    }
    active_residue_indices = set(qm_residue_indices)
    residue_cutoff = manifest.refinement.active_region_cutoff_angstrom
    water_cutoff = manifest.refinement.active_water_cutoff_angstrom
    qm_atom_records = [topology_structure.atoms[index] for index in qm_atoms]
    for residue_index, pair in enumerate(pairs):
        input_residue = pair.input_residue
        topology_residue = pair.topology_residue
        cutoff = water_cutoff if is_water_residue(input_residue) else residue_cutoff
        contacts_qm = any(
            _distance(atom, qm_atom) <= cutoff
            for atom in topology_residue.atoms
            if not is_virtual_site(atom)
            for qm_atom in qm_atom_records
        )
        if (
            contacts_qm
            and residue_index not in qm_residue_indices
            and residue_index not in mm_ions
            and is_nonstandard_nonwater_residue(input_residue)
            and not is_bulk_ion_residue(input_residue)
            # A kept noncanonical residue is a residue of the protein.
            and input_residue.id.resname not in kept_charges
        ):
            raise RefinementSelectionError(
                "A non-protein, non-water component contacts the configured QM region but "
                "was not explicitly selected. Configure it as a ligand/cofactor with charge "
                "and multiplicity and add its id to refinement.qm_components.ligands, or "
                f"move it outside the active-site model: {input_residue.id.display()}"
            )
        if contacts_qm:
            active_residue_indices.add(residue_index)

    active_region_atoms = sorted(
        index
        for residue_index in active_residue_indices
        for index in topology_atom_indices_by_residue[residue_index]
        if index not in virtual_sites
    )
    movable_atom_policy = manifest.refinement.movable_atoms
    if movable_atom_policy == "active_region_hydrogens":
        active_atoms = [
            index
            for index in active_region_atoms
            if is_hydrogen(topology_structure.atoms[index])
        ]
        if not active_atoms:
            raise RefinementSelectionError(
                "refinement.movable_atoms selects active-region hydrogens, but the "
                "provisional topology contains no hydrogen atoms in the active region"
            )
    else:
        active_atoms = active_region_atoms
    charge_components = _charge_components(
        pairs=pairs,
        manifest=manifest,
        kept_charges=kept_charges,
        selected_ligand_residues=selected_ligand_residues,
        selected_metal_residues=selected_metal_residues,
        selected_coordinating_ions=selected_coordinating_ions,
        coordinator_residue_indices=coordinator_residue_indices,
    )
    multiplicity = _total_multiplicity(manifest)
    return RefinementSelection(
        qm_atom_indices=tuple(qm_atoms),
        active_atom_indices=tuple(active_atoms),
        qm_residue_indices=tuple(sorted(qm_residue_indices)),
        active_residue_indices=tuple(sorted(active_residue_indices)),
        qm_residues=tuple(
            _residue_description(pairs[index], index)
            for index in sorted(qm_residue_indices)
        ),
        active_residues=tuple(
            _residue_description(pairs[index], index)
            for index in sorted(active_residue_indices)
        ),
        charge_components=tuple(charge_components),
        total_qm_charge=sum(item.charge for item in charge_components),
        total_qm_multiplicity=multiplicity,
        topology_atom_count=len(topology_structure.atoms),
        excluded_virtual_site_indices=tuple(sorted(virtual_sites)),
        qm_boundary_excluded_atom_indices=tuple(sorted(metal_boundary_exclusions)),
        active_region_cutoff_angstrom=residue_cutoff,
        active_water_cutoff_angstrom=water_cutoff,
        movable_atom_policy=movable_atom_policy,
    )


def _resolve_protonation_aware(
    structure: PdbStructure,
    selector: ResidueSelector,
) -> ResidueRecord:
    data = selector.model_dump()
    try:
        return resolve_residue_selector(structure, data)
    except SelectorError as original:
        family = next(
            (item for item in _PROTONATION_FAMILIES if selector.resname in item),
            None,
        )
        if family is None:
            raise RefinementSelectionError(str(original)) from original
        relaxed = dict(data)
        relaxed.pop("resname", None)
        try:
            residue = resolve_residue_selector(structure, relaxed)
        except SelectorError:
            raise RefinementSelectionError(str(original)) from original
        if residue.id.resname not in family:
            raise RefinementSelectionError(str(original)) from original
        return residue


def _charge_components(
    *,
    pairs: list[MappedResidue],
    manifest: ManifestConfig,
    selected_ligand_residues: dict[int, str],
    selected_metal_residues: dict[int, tuple[str, int]],
    selected_coordinating_ions: dict[int, tuple[str, object]],
    coordinator_residue_indices: set[int],
    kept_charges: dict[str, int] | None = None,
) -> list[ChargeComponent]:
    ligand_by_id = {ligand.id: ligand for ligand in manifest.ligands}
    site_by_id = {site.id: site for site in manifest.metals}
    components: list[ChargeComponent] = []
    for residue_index, ligand_id in sorted(selected_ligand_residues.items()):
        residue = pairs[residue_index].input_residue
        components.append(
            ChargeComponent(
                label=f"ligand:{ligand_id}:{residue.id.display()}",
                kind="ligand",
                charge=ligand_by_id[ligand_id].net_charge,
            )
        )
    for residue_index, (site_id, ion_index) in sorted(selected_metal_residues.items()):
        residue = pairs[residue_index].input_residue
        ion = site_by_id[site_id].ions[ion_index]
        components.append(
            ChargeComponent(
                label=f"metal:{site_id}:{residue.id.display()}",
                kind="metal",
                charge=ion.charge,
            )
        )
    for residue_index, (site_id, ion_config) in sorted(
        selected_coordinating_ions.items()
    ):
        residue = pairs[residue_index].input_residue
        components.append(
            ChargeComponent(
                label=f"coordinating_ion:{site_id}:{residue.id.display()}",
                kind="metal",
                charge=ion_config.charge,
            )
        )
    for residue_index in sorted(coordinator_residue_indices):
        input_residue = pairs[residue_index].input_residue
        topology_residue = pairs[residue_index].topology_residue
        if residue_index in selected_ligand_residues:
            continue
        if residue_index in selected_metal_residues:
            continue
        if residue_index in selected_coordinating_ions:
            continue
        if is_water_residue(input_residue):
            # An aqua ligand: the open-shell iron of an alpha-ketoglutarate
            # enzyme at rest usually carries one, where O2 later binds. Built
            # by tleap as a neutral water, which is what is counted here. A
            # hydroxide would need a residue template this version does not
            # build, so there is no way to ask for one.
            components.append(
                ChargeComponent(
                    label=f"water:{input_residue.id.display()}",
                    kind="water",
                    charge=0,
                )
            )
            continue
        if input_residue.id.resname in (kept_charges or {}):
            # Its charge is the one its block states, a closed shell; it sits
            # inside the chain, so no terminus changes it.
            assert kept_charges is not None
            components.append(
                ChargeComponent(
                    label=f"protein:{input_residue.id.display()}",
                    kind="protein_residue",
                    charge=kept_charges[input_residue.id.resname],
                )
            )
            continue
        if not is_standard_protein_residue(input_residue):
            raise RefinementSelectionError(
                "A non-protein metal coordinator must be configured as a ligand/cofactor "
                f"QM component with explicit charge and multiplicity: {input_residue.id.display()}"
            )
        charge = _SIDECHAIN_FORMAL_CHARGES.get(input_residue.id.resname, 0)
        atom_names = set(topology_residue.atom_names())
        if len(atom_names.intersection({"H1", "H2", "H3"})) >= 2:
            charge += 1
        if "OXT" in atom_names:
            charge -= 1
        components.append(
            ChargeComponent(
                label=f"protein:{input_residue.id.display()}",
                kind="protein_residue",
                charge=charge,
            )
        )
    return components


def _total_multiplicity(manifest: ManifestConfig) -> int:
    explicit = manifest.refinement.total_qm_multiplicity
    if explicit is not None:
        return explicit
    ligand_by_id = {ligand.id: ligand for ligand in manifest.ligands}
    site_by_id = {site.id: site for site in manifest.metals}
    multiplicities = [
        ligand_by_id[ligand_id].multiplicity
        for ligand_id in manifest.refinement.qm_components.ligands
    ]
    multiplicities.extend(
        ion.multiplicity or 1
        for site_id in manifest.refinement.qm_components.metal_sites
        for ion in site_by_id[site_id].ions
    )
    open_shell = [value for value in multiplicities if value > 1]
    if open_shell:
        # The manifest schema refuses this case, so reaching it means a
        # selection was built from something that skipped validation. Guess
        # here and the whole refinement runs at a spin nobody chose.
        raise RefinementSelectionError(
            "refinement.total_qm_multiplicity must be set: the QM region "
            f"contains {len(open_shell)} open-shell component(s) and its "
            "total spin state is a chemical decision, not the multiplicity "
            "of any one of them"
        )
    return 1


def _validate_protonated_waters(
    pairs: list[MappedResidue],
) -> None:
    for pair in pairs:
        input_residue = pair.input_residue
        topology_residue = pair.topology_residue
        if not is_water_residue(input_residue):
            continue
        elements = [(atom.element or "").strip().upper() for atom in topology_residue.atoms]
        if elements.count("O") != 1 or elements.count("H") < 2:
            raise RefinementSelectionError(
                f"Water {input_residue.id.display()} is not protonated in the provisional "
                "Amber topology; QM/MM refinement requires one oxygen and two hydrogens"
            )


def _distance(left: AtomRecord, right: AtomRecord) -> float:
    return dist((left.x, left.y, left.z), (right.x, right.y, right.z))


def _residue_description(
    pair: MappedResidue,
    index: int,
) -> dict[str, object]:
    input_residue = pair.input_residue
    topology_residue = pair.topology_residue
    return {
        "residue_index_zero_based": index,
        "topology_residue_index_zero_based": pair.topology_index,
        "mapping_anchor_rmsd_angstrom": pair.anchor_rmsd_angstrom,
        "input_identity": input_residue.id.to_dict(),
        "topology_identity": topology_residue.id.to_dict(),
        "atom_count": len(topology_residue.atoms),
        "water": is_water_residue(input_residue),
    }
