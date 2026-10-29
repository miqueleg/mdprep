"""Kept noncanonical residues in the normalised structure.

Normalisation is where a kept residue becomes what the build will see:

* it is written with ATOM records, as a residue of its chain. The PDB writer
  starts a new chain at a HETATM residue that follows an ATOM one, so as a
  heterogen tleap would never have bonded it to the residue before it;
* built from a template, it takes the template's name -- every rename is
  reported -- and leaves its hydrogens to tleap, as a standard residue does;
* parameterised from its SMILES, it is given the SMILES's hydrogens, placed
  between the caps it is parameterised in, so the structure, the charges and
  the library all describe one molecule.

A modified residue of a chain that the manifest neither keeps, replaces nor
removes stops the build. Left to ``remove_unknown_heterogens`` it would be
cut out of its chain, leaving two termini where there was a peptide bond.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from mdprep.config.models import ManifestConfig, NoncanonicalResidueConfig
from mdprep.residues.templates import AMBER_RESIDUE_TEMPLATES, missing_template_atoms
from mdprep.structure.heterogens import classify_heterogens
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueRecord
from mdprep.structure.selectors import SelectorError, resolve_residue_selector


class KeptResidueError(ValueError):
    """Raised when a kept residue cannot be prepared as configured."""


@dataclass(frozen=True)
class KeptResidue:
    """What normalisation did to one kept residue."""

    chain: str
    resname: str
    resid: int
    icode: str | None
    final_resname: str
    parameters: str
    template: str | None
    hydrogens_placed: int
    input_hydrogens_removed: int
    record_names_before: tuple[str, ...]
    smiles_form: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "chain": self.chain,
            "resname": self.resname,
            "resid": self.resid,
            "icode": self.icode,
            "final_resname": self.final_resname,
            "renamed": self.final_resname != self.resname,
            "parameters": self.parameters,
            "template": self.template,
            "hydrogens_placed": self.hydrogens_placed,
            "input_hydrogens_removed": self.input_hydrogens_removed,
            "record_names_before": list(self.record_names_before),
            "smiles_form": self.smiles_form,
        }


def resolve_noncanonical_residues(
    structure: PdbStructure, manifest: ManifestConfig
) -> dict[int, NoncanonicalResidueConfig]:
    """Each residue a noncanonical_residues block names, by id, with its block."""

    kept_chains = {chain.strip() for chain in manifest.structure.keep_chains}
    resolved: dict[int, NoncanonicalResidueConfig] = {}
    for block in manifest.noncanonical_residues:
        for selector in block.residues:
            if kept_chains and selector.chain not in kept_chains:
                raise KeptResidueError(
                    f"noncanonical_residues {block.resname} names "
                    f"{selector.chain}:{selector.resname}{selector.resid}, in a chain "
                    f"structure.keep_chains leaves out ({sorted(kept_chains)})."
                )
            try:
                residue = resolve_residue_selector(structure, selector.model_dump())
            except SelectorError as exc:
                raise KeptResidueError(
                    f"noncanonical_residues {block.resname} names a residue the "
                    f"structure does not have: {exc}"
                ) from exc
            resolved[id(residue)] = block
    return resolved


def unconfigured_modified_residues(
    structure: PdbStructure,
    *,
    settled: set[int],
    kept_chains: set[str],
) -> list[ResidueRecord]:
    """Modified residues of a kept chain that nothing in the manifest settles.

    ``settled`` holds the ids of residues kept, removed or otherwise accounted
    for. A replaced one is no longer modified: PDBFixer has built its parent.
    """

    groups = classify_heterogens(
        structure.residues,
        getattr(structure, "links", ()),
        getattr(structure, "modified_residue_parents", {}),
    )
    return [
        item.residue
        for item in groups.modified_residues
        if id(item.residue) not in settled
        and (not kept_chains or item.residue.id.chain_id in kept_chains)
    ]


def describe_unconfigured(residues: Sequence[ResidueRecord]) -> str:
    listed = ", ".join(residue.id.display() for residue in residues)
    many = len(residues) > 1
    return (
        f"{listed} {'are residues' if many else 'is a residue'} of a protein chain "
        "that the manifest does not account for: not one of the twenty amino "
        "acids, and not a ligand -- removing "
        f"{'them' if many else 'it'} as an unknown heterogen would cut the chain "
        f"in two. Keep {'them' if many else 'it'} under noncanonical_residues, "
        "replace by the standard parent with structure.repair."
        "replace_modified_residues, or leave out on purpose with "
        "structure.remove_residues (the input builder asks)."
    )


def prepare_noncanonical_residues(
    residues: Sequence[ResidueRecord],
    configured: dict[int, NoncanonicalResidueConfig],
) -> tuple[dict[int, list[AtomRecord]], list[KeptResidue], list[str]]:
    """The atoms each kept residue is built from, and what was done to it.

    ``residues`` are those the structure keeps, in order: a residue's
    neighbours, which its caps are built from, must be kept too.
    Returns the new atoms keyed by the old residue's id, an outcome per
    residue, and warnings.
    """

    rewritten: dict[int, list[AtomRecord]] = {}
    outcomes: list[KeptResidue] = []
    warnings: list[str] = []
    heavy_names: dict[str, tuple[str, list[str]]] = {}
    for residue in residues:
        block = configured.get(id(residue))
        if block is None:
            continue
        heavy = [atom for atom in residue.atoms if not is_hydrogen(atom)]
        hydrogens = [atom for atom in residue.atoms if is_hydrogen(atom)]
        names = sorted(atom.name.strip() for atom in heavy)
        first = heavy_names.setdefault(block.resname, (residue.id.display(), names))
        if first[1] != names:
            raise KeptResidueError(
                f"{residue.id.display()} and {first[0]} are both kept as "
                f"{block.resname}, which tleap builds from one template, but their "
                f"heavy atoms differ: {names} against {first[1]}."
            )
        as_atom = [replace(atom, record_name="ATOM") for atom in heavy]
        if block.parameters == "amber_library":
            assert block.template is not None
            template = AMBER_RESIDUE_TEMPLATES[block.template]
            extra, absent = missing_template_atoms(template, (atom.name for atom in heavy))
            if extra or absent:
                raise KeptResidueError(
                    f"{residue.id.display()} cannot be built from the {template.name} "
                    f"template ({template.description}): "
                    + "; ".join(
                        part
                        for part in (
                            f"it has {', '.join(extra)}, which the template does not"
                            if extra
                            else "",
                            f"it lacks {', '.join(absent)}" if absent else "",
                        )
                        if part
                    )
                    + ". tleap matches atoms by name, and would drop the first and "
                    "build the second from nothing."
                )
            atoms = [replace(atom, resname=template.name) for atom in as_atom]
            placed = 0
            form = None
            if template.name != residue.id.resname:
                warnings.append(
                    f"{residue.id.display()} renamed {template.name}: AmberTools' "
                    f"{template.name} template is {template.description}."
                )
        else:
            from mdprep.residues.capping import (
                ResidueCappingError,
                capped_residue,
                chain_context,
                residue_hydrogens,
            )

            assert block.smiles is not None and block.net_charge is not None
            try:
                context = chain_context(residues, residue)
                capped = capped_residue(
                    context,
                    block.smiles,
                    net_charge=block.net_charge,
                    label=f"noncanonical residue {residue.id.display()}",
                )
            except ResidueCappingError as exc:
                raise KeptResidueError(str(exc)) from exc
            placed_atoms = residue_hydrogens(capped, residue)
            atoms = as_atom + placed_atoms
            placed = len(placed_atoms)
            form = capped.smiles_form
            warnings.append(
                f"WARNING [RESIDUE_HYDROGENS_ADDED] {residue.id.display()} kept as "
                f"{block.resname}: mdprep placed {placed} hydrogens from its SMILES "
                f"{block.smiles!r}. Their positions are geometric, and the protonation "
                "is the SMILES's; inspect the residue before trusting its charges."
            )
        if hydrogens:
            warnings.append(
                f"{residue.id.display()} arrived with {len(hydrogens)} hydrogen(s), "
                "removed: a kept residue's hydrogens are "
                + (
                    "its template's, built by tleap."
                    if block.parameters == "amber_library"
                    else "its SMILES's."
                )
            )
        rewritten[id(residue)] = atoms
        outcomes.append(
            KeptResidue(
                chain=residue.id.chain_id,
                resname=residue.id.resname,
                resid=residue.id.resid,
                icode=residue.id.icode,
                final_resname=atoms[0].resname,
                parameters=block.parameters,
                template=block.template,
                hydrogens_placed=placed,
                input_hydrogens_removed=len(hydrogens),
                record_names_before=tuple(sorted(residue.record_names)),
                smiles_form=form,
            )
        )
    return rewritten, outcomes, warnings
