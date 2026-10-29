"""The frcmod of a residue whose side chain is GAFF2 and whose backbone is not.

parmchk2 only knows one force field at a time. Asked about the capped residue
against GAFF2 (``-s gaff2``) it finds the side chain's terms, estimates the
ones joining it to the backbone, and also "estimates" every term of the
backbone and the caps -- the protein force field's own atom types are unknown
to it in GAFF2 -- and writes them out with GAFF2 values: C-N-CX-C, the phi
torsion of every residue in the protein, among them. tleap applies an frcmod
to every atom of those types, so loading that file would rewrite the
backbone of the whole protein.

So only the terms that involve at least one GAFF2 type are kept. A term of
the protein's types alone is the protein force field's, which tleap has
already loaded, and one of GAFF2 types alone is GAFF2's own. What remains is
the junction between the two, estimated from GAFF2 analogues, and any GAFF2
term parmchk2 had to estimate.

ff19SB applies a backbone CMAP by residue name, so a new residue name gets
none unless the frcmod lists it; it is given its parent's, as AmberTools'
phosphorylated residues are given Ser's, Thr's and the general one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SECTIONS = ("MASS", "BOND", "ANGLE", "ANGL", "DIHE", "IMPROPER", "IMPR", "NONBON", "NONB")
_TYPE_FIELD_WIDTH = {"BOND": 5, "ANGLE": 8, "ANGL": 8, "DIHE": 11, "IMPROPER": 11, "IMPR": 11}


class ResidueFrcmodError(ValueError):
    """Raised when a residue's parameter file cannot be written safely."""


def is_gaff_type(atom_type: str) -> bool:
    """GAFF and GAFF2 types are lower case; the protein force fields' are not."""

    return atom_type.strip()[:1].islower()


def junction_parameters(
    frcmod_text: str, rename: dict[str, str] | None = None
) -> tuple[str, list[str]]:
    """The frcmod with only the terms that involve a GAFF2 atom type.

    Returns the text and the terms dropped, so the report can say what was
    left to the protein force field. ``rename`` replaces atom types in the
    terms kept: parmchk2 has no entry for ff19SB's CA type XC, so it is asked
    about CX, ff14SB's, and its answers are given back to XC.
    """

    rename = rename or {}
    kept: list[str] = []
    dropped: list[str] = []
    section: str | None = None
    for line in frcmod_text.splitlines():
        stripped = line.strip()
        if stripped in SECTIONS:
            section = stripped
            kept.append(line)
            continue
        if section is None or not stripped:
            kept.append(line)
            continue
        types = _types(line, section)
        if any(is_gaff_type(atom_type) for atom_type in types):
            kept.append(_renamed(line, section, rename) if rename else line)
        else:
            dropped.append(f"{section} {'-'.join(types)}")
    return "\n".join(kept).rstrip() + "\n", dropped


def attention_lines(frcmod_text: str) -> list[str]:
    """Terms parmchk2 could not estimate and marked for revision."""

    return [line.strip() for line in frcmod_text.splitlines() if "ATTN" in line.upper()]


def _renamed(line: str, section: str, rename: dict[str, str]) -> str:
    if section in {"MASS", "NONBON", "NONB"}:
        head, _, rest = line.partition(line.split()[0])
        return head + rename.get(line.split()[0], line.split()[0]) + rest
    width = _TYPE_FIELD_WIDTH[section]
    fields = line[:width].split("-")
    renamed = []
    for field in fields:
        name = field.strip()
        replacement = rename.get(name, name)
        renamed.append(replacement.ljust(len(field)) if len(replacement) <= len(field) else replacement)
    return "-".join(renamed) + line[width:]


def _types(line: str, section: str) -> list[str]:
    if section in {"MASS", "NONBON", "NONB"}:
        return [line.split()[0]]
    width = _TYPE_FIELD_WIDTH[section]
    return [field.strip() for field in line[:width].split("-")]


@dataclass(frozen=True)
class CmapBlock:
    title: str
    residues: tuple[str, ...]
    resolution: int
    parameters: tuple[str, ...]


def cmap_blocks(frcmod_text: str) -> list[CmapBlock]:
    """The CMAP blocks of an frcmod such as ff19SB's, in order."""

    lines = frcmod_text.splitlines()
    blocks: list[CmapBlock] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("%FLAG CMAP_COUNT"):
            index += 1
            continue
        title = ""
        residues: list[str] = []
        resolution = 0
        parameters: list[str] = []
        index += 1
        reading: str | None = None
        while index < len(lines) and not lines[index].startswith("%FLAG CMAP_COUNT"):
            line = lines[index]
            if line.startswith("%FLAG CMAP_TITLE"):
                reading = "title"
            elif line.startswith("%FLAG CMAP_RESLIST"):
                reading = "residues"
            elif line.startswith("%FLAG CMAP_RESOLUTION"):
                resolution = int(line.split()[-1])
                reading = None
            elif line.startswith("%FLAG CMAP_PARAMETER"):
                reading = "parameters"
            elif line.startswith("%"):
                reading = None
            elif not line.strip():
                if reading == "parameters":
                    reading = None
            elif reading == "title":
                title = line.strip()
                reading = None
            elif reading == "residues":
                residues.extend(line.split())
            elif reading == "parameters":
                parameters.append(line.rstrip())
            index += 1
        blocks.append(
            CmapBlock(
                title=title,
                residues=tuple(residues),
                resolution=resolution,
                parameters=tuple(parameters),
            )
        )
    return blocks


def parent_cmap(frcmod_text: str, parent: str) -> CmapBlock:
    """The CMAP ff19SB applies to ``parent``."""

    for block in cmap_blocks(frcmod_text):
        if parent in block.residues:
            values = sum(len(line.split()) for line in block.parameters)
            if values != block.resolution * block.resolution:
                raise ResidueFrcmodError(
                    f"The {block.title} for {parent} has {values} values, not "
                    f"{block.resolution}x{block.resolution}."
                )
            return block
    raise ResidueFrcmodError(f"The force field has no CMAP for {parent}.")


def cmap_section(resname: str, parent: str, block: CmapBlock, source: str) -> str:
    """An frcmod CMAP section applying ``block`` to residues named ``resname``."""

    return "\n".join(
        [
            "",
            "CMAP",
            "%FLAG CMAP_COUNT   1",
            "%FLAG CMAP_TITLE",
            f"{resname}: the {block.title} {source} applies to {parent}",
            "%FLAG CMAP_RESLIST  1",
            resname,
            f"%FLAG CMAP_RESOLUTION {block.resolution}",
            "%FLAG CMAP_PARAMETER",
            *block.parameters,
            "",
        ]
    )


def ff19sb_frcmod_path(amberhome: Path) -> Path:
    return amberhome / "dat" / "leap" / "parm" / "frcmod.ff19SB"
