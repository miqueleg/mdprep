"""A kept residue's charges, for the MM embedding of the histidine calculations.

The xTB comparison of a histidine's tautomers puts the side chain in a field
of the surroundings' MM charges, taken from the protein force field's
templates. A kept noncanonical residue has none there -- a phosphoserine's
-2 beside the histidine being decided would simply be missing -- so its own
are supplied: the AmberTools template's, or the charges mdprep fitted for it
from its SMILES, which the residue stage has written before protonation runs.
A residue whose charges are fitted with QMMESP has only its provisional
AM1-BCC ones by then -- the fit needs the protonated system around it -- so
those are the ones used.

A template's residue has no hydrogens yet (tleap builds them), so each
hydrogen's charge is given to the atom it is bonded to: the residue keeps
its integer charge, and at the distances an embedding reaches a hydrogen
and its heavy atom are one point.
"""

from __future__ import annotations

import json

from mdprep.config.models import ManifestConfig, NoncanonicalResidueConfig
from mdprep.residues.offlib import OffLibraryError, read_off_unit
from mdprep.residues.templates import AMBER_RESIDUE_TEMPLATES
from mdprep.residues.workflow import final_resname, parameterisation_directory
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


def kept_residue_charges(
    structure: PdbStructure, manifest: ManifestConfig
) -> tuple[dict[int, list[float]], list[str]]:
    """Charges per atom of each kept residue in ``structure``, by residue id.

    Returns them with the reason any residue has none, which the embedding
    then reports as it does any residue without a template.
    """

    charges: dict[int, list[float]] = {}
    missing: list[str] = []
    for block in manifest.noncanonical_residues:
        try:
            table = _block_charges(block, manifest)
        except (OSError, OffLibraryError, ValueError, KeyError) as exc:
            missing.append(f"{block.resname}: its charges could not be read ({exc})")
            continue
        for selector in block.residues:
            try:
                residue = resolve_residue_selector(
                    structure, {**selector.model_dump(), "resname": final_resname(block)}
                )
            except SelectorError:
                continue
            aligned = _aligned(residue, *table)
            if aligned is None:
                missing.append(
                    f"{residue.id.display()}: its atoms are not those of its "
                    f"{final_resname(block)} charges"
                )
                continue
            charges[id(residue)] = aligned
    return charges, missing


def _block_charges(
    block: NoncanonicalResidueConfig, manifest: ManifestConfig
) -> tuple[dict[str, float], dict[str, set[str]]]:
    """Charge by atom name, and each atom's bonded neighbours by name."""

    if block.parameters != "amber_library":
        path = (
            parameterisation_directory(manifest.project.output_dir, block) / "parameters.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        return {name: float(value) for name, value in data["charges"]["charges"].items()}, {}
    assert block.template is not None
    template = AMBER_RESIDUE_TEMPLATES[block.template]
    from mdprep.metals.nonbonded import resolve_amberhome

    library = (
        resolve_amberhome()
        / "dat"
        / "leap"
        / "lib"
        / template.library_file(manifest.protein.forcefield)
    )
    unit = read_off_unit(library.read_text(encoding="utf-8"), template.name)
    names = [atom.name for atom in unit.atoms]
    neighbours: dict[str, set[str]] = {name: set() for name in names}
    for first, second in unit.bonds:
        neighbours[names[first]].add(names[second])
        neighbours[names[second]].add(names[first])
    return {atom.name: atom.charge for atom in unit.atoms}, neighbours


def _aligned(
    residue: ResidueRecord, charges: dict[str, float], neighbours: dict[str, set[str]]
) -> list[float] | None:
    """The charges in the residue's atom order, absent hydrogens folded in."""

    present = [atom.name.strip() for atom in residue.atoms]
    if any(name not in charges for name in present):
        return None
    folded = {name: charges[name] for name in present}
    for name, charge in charges.items():
        if name in folded:
            continue
        hosts = [other for other in neighbours.get(name, ()) if other in folded]
        if not name.startswith("H") or len(hosts) != 1:
            return None
        folded[hosts[0]] += charge
    return [folded[name] for name in present]


def describe_missing(reasons: list[str]) -> str:
    return (
        "Kept noncanonical residues without charges for the histidine embedding: "
        + "; ".join(reasons)
        + "."
    )

