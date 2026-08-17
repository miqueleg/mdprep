"""Resolve and validate explicitly declared MCPB.py coordination sites."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from mdprep.config.models import ManifestConfig, MetalSiteConfig
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_atom_selector, resolve_residue_selector


MCPB_AUTOMATIC_DONOR_ELEMENTS = frozenset({"N", "O", "S", "F", "Cl", "Br", "I"})


class MetalCoordinationError(ValueError):
    """Raised when a metal site is ambiguous or inconsistent."""


@dataclass(frozen=True)
class ResolvedMetalIon:
    site_id: str
    element: str
    charge: int
    atom: AtomRecord
    residue: ResidueRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "selector": _atom_dict(self.atom),
            "element": self.element,
            "charge": self.charge,
            "atom_serial": self.atom.serial,
        }


@dataclass(frozen=True)
class ResolvedMetalBond:
    ion: ResolvedMetalIon
    coordinator: AtomRecord
    coordinator_residue: ResidueRecord
    distance_angstrom: float
    discovered_by_cutoff: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "ion": _atom_dict(self.ion.atom),
            "coordinator": _atom_dict(self.coordinator),
            "distance_angstrom": self.distance_angstrom,
            "discovered_by_mcpb_cutoff": self.discovered_by_cutoff,
        }


@dataclass(frozen=True)
class ResolvedMetalSite:
    config: MetalSiteConfig
    ions: tuple[ResolvedMetalIon, ...]
    bonds: tuple[ResolvedMetalBond, ...]
    additional_residues: tuple[ResidueRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.config.id,
            "model": self.config.model,
            "ions": [ion.to_dict() for ion in self.ions],
            "bonds": [bond.to_dict() for bond in self.bonds],
            "additional_residues": [residue.id.to_dict() for residue in self.additional_residues],
        }


def resolve_metal_sites(
    structure: PdbStructure,
    manifest: ManifestConfig,
    *,
    validate_mcpb_cutoff: bool = True,
) -> list[ResolvedMetalSite]:
    """Resolve all configured ions and explicit bonds against ``structure``."""

    bonded_sites = [site for site in manifest.metals if site.model == "bonded_mcpb"]
    if len(bonded_sites) > 1:
        raise MetalCoordinationError(
            "Only one bonded_mcpb site is supported in one preparation. Put interacting ions "
            "in the same site's ions list; independent MCPB.py residue-renaming outputs cannot "
            "be combined safely."
        )

    resolved: list[ResolvedMetalSite] = []
    global_ion_keys: set[tuple[str, str, int, str | None, str]] = set()
    for site in manifest.metals:
        ions = tuple(_resolve_ion(structure, site, ion) for ion in site.ions)
        for ion in ions:
            key = ion.atom.atom_identity
            if key in global_ion_keys:
                raise MetalCoordinationError(f"Metal atom {_atom_label(ion.atom)} occurs in multiple sites.")
            global_ion_keys.add(key)

        bonds: tuple[ResolvedMetalBond, ...] = ()
        additional: tuple[ResidueRecord, ...] = ()
        if site.model == "bonded_mcpb":
            assert site.mcpb is not None
            ion_by_key = {ion.atom.atom_identity: ion for ion in ions}
            bond_items: list[ResolvedMetalBond] = []
            seen_pairs: set[tuple[tuple[str, str, int, str | None, str], tuple[str, str, int, str | None, str]]] = set()
            for bond in site.mcpb.bonds:
                try:
                    ion_atom, _ = _resolve_atom_and_residue(
                        structure, bond.ion.model_dump()
                    )
                    coordinator, coordinator_residue = _resolve_atom_and_residue(
                        structure, bond.coordinator.model_dump()
                    )
                except SelectorError as exc:
                    raise MetalCoordinationError(
                        f"Metal site {site.id!r} bond selector failed: {exc}"
                    ) from exc
                ion = ion_by_key.get(ion_atom.atom_identity)
                if ion is None:
                    raise MetalCoordinationError(
                        f"Metal site {site.id!r} bond ion {_atom_label(ion_atom)} is not listed "
                        "under that site's ions."
                    )
                if coordinator.atom_identity in global_ion_keys:
                    raise MetalCoordinationError(
                        f"Metal site {site.id!r} uses metal atom {_atom_label(coordinator)} as a "
                        "coordinator; metal--metal bonds are not inferred by this workflow."
                    )
                pair = (ion.atom.atom_identity, coordinator.atom_identity)
                if pair in seen_pairs:
                    raise MetalCoordinationError(
                        f"Duplicate metal bond {_atom_label(ion.atom)}--{_atom_label(coordinator)}."
                    )
                seen_pairs.add(pair)
                distance = _distance(ion.atom, coordinator)
                discovered = (
                    distance <= site.mcpb.cutoff_angstrom
                    and _element(coordinator) in MCPB_AUTOMATIC_DONOR_ELEMENTS
                )
                bond_items.append(
                    ResolvedMetalBond(
                        ion=ion,
                        coordinator=coordinator,
                        coordinator_residue=coordinator_residue,
                        distance_angstrom=distance,
                        discovered_by_cutoff=discovered,
                    )
                )
            bonds = tuple(bond_items)
            additional = tuple(
                _resolve_additional_residue(structure, site.id, selector.model_dump())
                for selector in site.mcpb.additional_residues
            )
            if validate_mcpb_cutoff:
                _validate_exact_mcpb_candidates(structure, site, ions, bonds)

        resolved.append(
            ResolvedMetalSite(config=site, ions=ions, bonds=bonds, additional_residues=additional)
        )
    return resolved


def _resolve_ion(structure: PdbStructure, site: MetalSiteConfig, config: object) -> ResolvedMetalIon:
    selector = config.selector.model_dump()  # type: ignore[attr-defined]
    try:
        atom, residue = _resolve_atom_and_residue(structure, selector)
    except SelectorError as exc:
        raise MetalCoordinationError(f"Metal site {site.id!r} ion selector failed: {exc}") from exc
    element = config.element  # type: ignore[attr-defined]
    observed = _element(atom)
    if observed and observed != element:
        raise MetalCoordinationError(
            f"Metal site {site.id!r} declares {element} for {_atom_label(atom)}, "
            f"but the PDB element is {observed}."
        )
    if len(residue.atoms) != 1:
        raise MetalCoordinationError(
            f"Metal ion residue {residue.id.display()} contains {len(residue.atoms)} atoms; "
            "a metal ion must be a one-atom residue."
        )
    if atom.serial is None:
        raise MetalCoordinationError(f"Metal atom {_atom_label(atom)} has no PDB serial number.")
    return ResolvedMetalIon(
        site_id=site.id,
        element=element,
        charge=config.charge,  # type: ignore[attr-defined]
        atom=atom,
        residue=residue,
    )


def _resolve_additional_residue(
    structure: PdbStructure,
    site_id: str,
    selector: dict[str, object],
) -> ResidueRecord:
    try:
        return _resolve_residue_protonation_aware(structure, selector)
    except SelectorError as exc:
        raise MetalCoordinationError(
            f"Metal site {site_id!r} additional residue selector failed: {exc}"
        ) from exc


def _validate_exact_mcpb_candidates(
    structure: PdbStructure,
    site: MetalSiteConfig,
    ions: tuple[ResolvedMetalIon, ...],
    bonds: tuple[ResolvedMetalBond, ...],
) -> None:
    assert site.mcpb is not None
    declared = {(bond.ion.atom.atom_identity, bond.coordinator.atom_identity) for bond in bonds}
    discovered: set[
        tuple[
            tuple[str, str, int, str | None, str],
            tuple[str, str, int, str | None, str],
        ]
    ] = set()
    ion_keys = {ion.atom.atom_identity for ion in ions}
    for ion in ions:
        for atom in structure.atoms:
            if atom.atom_identity in ion_keys:
                continue
            if _element(atom) not in MCPB_AUTOMATIC_DONOR_ELEMENTS:
                continue
            if _distance(ion.atom, atom) <= site.mcpb.cutoff_angstrom:
                discovered.add((ion.atom.atom_identity, atom.atom_identity))
    undeclared = discovered - declared
    if undeclared:
        formatted = ", ".join(
            f"{_identity_label(ion)}--{_identity_label(atom)}"
            for ion, atom in sorted(undeclared, key=str)
        )
        raise MetalCoordinationError(
            f"MCPB.py would create undeclared bonds within {site.mcpb.cutoff_angstrom:.3f} A "
            f"for site {site.id!r}: {formatted}. Add every intended bond under mcpb.bonds or "
            "choose a cutoff that represents the explicitly approved coordination sphere."
        )


def _distance(a: AtomRecord, b: AtomRecord) -> float:
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))


_PROTONATION_FAMILIES = (
    frozenset({"HIS", "HID", "HIE", "HIP"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"CYS", "CYM", "CYX"}),
    frozenset({"LYS", "LYN"}),
    frozenset({"HOH", "WAT", "H2O", "TIP3", "OPC"}),
)


def _resolve_atom_and_residue(
    structure: PdbStructure,
    selector: dict[str, object],
) -> tuple[AtomRecord, ResidueRecord]:
    residue = _resolve_residue_protonation_aware(structure, selector)
    atom_name = str(selector["atom_name"])
    matches = [atom for atom in residue.atoms if atom.name == atom_name]
    if len(matches) != 1:
        available = ", ".join(residue.atom_names())
        raise SelectorError(
            f"Expected one atom {atom_name!r} in {residue.id.display()}, found {len(matches)}. "
            f"Available atoms: {available}"
        )
    return matches[0], residue


def _resolve_residue_protonation_aware(
    structure: PdbStructure,
    selector: dict[str, object],
) -> ResidueRecord:
    try:
        return resolve_residue_selector(structure, selector)
    except SelectorError as original:
        requested = str(selector.get("resname", ""))
        family = next((item for item in _PROTONATION_FAMILIES if requested in item), None)
        if family is None:
            raise
        relaxed = dict(selector)
        relaxed.pop("atom_name", None)
        relaxed.pop("resname", None)
        try:
            residue = resolve_residue_selector(structure, relaxed)
        except SelectorError:
            raise original
        if residue.id.resname not in family:
            raise original
        return residue


def _element(atom: AtomRecord) -> str:
    return (atom.element or "").strip().capitalize()


def _atom_dict(atom: AtomRecord) -> dict[str, object]:
    return {
        "chain": atom.chain_id,
        "resname": atom.resname,
        "resid": atom.resid,
        "icode": atom.icode,
        "atom_name": atom.name,
    }


def _atom_label(atom: AtomRecord) -> str:
    return _identity_label(atom.atom_identity)


def _identity_label(identity: tuple[str, str, int, str | None, str]) -> str:
    chain, resname, resid, icode, atom_name = identity
    return f"{chain or '<blank>'}:{resname}{resid}{icode or ''}@{atom_name}"
