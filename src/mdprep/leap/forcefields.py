"""Amber force-field mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

from mdprep.config.models import LigandConfig


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
class ForceFieldSources:
    protein: str
    water: str
    ligand: list[str]
    water_box: str
    warnings: list[str]

    @property
    def all_sources(self) -> list[str]:
        return [self.protein, self.water, *self.ligand]


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
) -> ForceFieldSources:
    ligand_families = sorted({ligand.atom_types for ligand in ligands})
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
    )
