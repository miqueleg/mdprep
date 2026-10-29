"""Amber force-field mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mdprep.config.models import LigandConfig

if TYPE_CHECKING:
    from mdprep.residues.workflow import ResidueLeapInputs


class ForceFieldError(ValueError):
    """Raised when a manifest requests an unsupported force-field setting."""


PROTEIN_LEAPRC = {
    "ff14SB": "leaprc.protein.ff14SB",
    "ff19SB": "leaprc.protein.ff19SB",
}

WATER_LEAPRC = {
    "TIP3P": "leaprc.water.tip3p",
    "OPC": "leaprc.water.opc",
}

WATER_BOX = {
    "TIP3P": "TIP3PBOX",
    "OPC": "OPCBOX",
}

LIGAND_LEAPRC = {
    "gaff": "leaprc.gaff",
    "gaff2": "leaprc.gaff2",
}


@dataclass(frozen=True)
class ResidueLibraryFiles:
    """A noncanonical residue's tleap library and the parameters it needs."""

    resname: str
    library_path: Path
    frcmod_path: Path


@dataclass(frozen=True)
class ForceFieldSources:
    protein: str
    water: str
    ligand: list[str]
    water_box: str
    warnings: list[str]
    # AmberTools' templates for kept noncanonical residues, sourced right
    # after the protein force field they were fitted to.
    residues: list[str] = field(default_factory=list)
    # Libraries mdprep made for kept noncanonical residues, loaded after
    # every source.
    residue_files: list[ResidueLibraryFiles] = field(default_factory=list)

    @property
    def all_sources(self) -> list[str]:
        return [self.protein, *self.residues, self.water, *self.ligand]


def protein_leaprc(name: str) -> str:
    try:
        return PROTEIN_LEAPRC[name]
    except KeyError as exc:
        raise ForceFieldError(f"Unsupported protein force field: {name}") from exc


def water_leaprc(name: str) -> str:
    try:
        return WATER_LEAPRC[name]
    except KeyError as exc:
        raise ForceFieldError(f"Unsupported water model: {name}") from exc


def water_box(name: str) -> str:
    try:
        return WATER_BOX[name]
    except KeyError as exc:
        raise ForceFieldError(f"Unsupported water model: {name}") from exc


def ligand_leaprc(name: str) -> str:
    try:
        return LIGAND_LEAPRC[name]
    except KeyError as exc:
        raise ForceFieldError(f"Unsupported ligand atom type family: {name}") from exc


def forcefield_sources(
    *,
    protein_forcefield: str,
    water_model: str,
    ligands: list[LigandConfig],
    residues: "ResidueLeapInputs | None" = None,
) -> ForceFieldSources:
    """What tleap sources for this system.

    ``residues`` are the kept noncanonical residues' inputs
    (``mdprep.residues.workflow.residue_leap_inputs``): a parameterised one's
    GAFF2 side chain needs GAFF2 sourced as a ligand's does.
    """

    residue_families: Sequence[str] = residues.atom_types if residues is not None else ()
    ligand_families = sorted({ligand.atom_types for ligand in ligands} | set(residue_families))
    ligand_sources = [ligand_leaprc(family) for family in ligand_families]
    warnings: list[str] = []
    if len(ligand_families) > 1:
        warnings.append("Both GAFF and GAFF2 ligand force fields are sourced.")
    return ForceFieldSources(
        protein=protein_leaprc(protein_forcefield),
        water=water_leaprc(water_model),
        ligand=ligand_sources,
        water_box=water_box(water_model),
        warnings=warnings,
        residues=list(residues.sources) if residues is not None else [],
        residue_files=list(residues.files) if residues is not None else [],
    )
