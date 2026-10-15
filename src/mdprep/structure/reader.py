"""Read an input structure, whichever of the two formats it is in.

Prediction writes mmCIF: Boltz emits ModelCIF, and the RCSB serves mmCIF for
everything. Everything mdprep generates downstream is PDB, so this exists for
the one file the user supplies rather than as a general format layer.
"""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.models import PdbStructure
from mdprep.structure.pdb import AltlocPolicy, PdbParseError, read_pdb

MMCIF_SUFFIXES = {".cif", ".mmcif"}
PDB_SUFFIXES = {".pdb", ".ent"}


def is_mmcif_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in MMCIF_SUFFIXES


def read_structure(
    path: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
) -> PdbStructure:
    """Read a PDB or mmCIF input into one structure model.

    Chosen by extension rather than by sniffing the contents: a file named
    .pdb that is secretly mmCIF is a mistake worth reporting, not one worth
    absorbing silently.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in MMCIF_SUFFIXES:
        from mdprep.structure.mmcif import read_mmcif

        return read_mmcif(source, altloc_policy=altloc_policy)
    if suffix in PDB_SUFFIXES or suffix == "":
        return read_pdb(source, altloc_policy=altloc_policy)
    raise PdbParseError(
        f"Unrecognised structure format {suffix!r} for {source}. mdprep reads "
        f"PDB ({', '.join(sorted(PDB_SUFFIXES))}) and mmCIF "
        f"({', '.join(sorted(MMCIF_SUFFIXES))})."
    )
