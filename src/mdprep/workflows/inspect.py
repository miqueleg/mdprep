"""Structure inspection workflow."""

from __future__ import annotations

from pathlib import Path

from mdprep.structure.inspect import InspectionSummary, inspect_pdb_structure
from mdprep.structure.pdb import AltlocPolicy


def inspect_structure(
    input_structure: str | Path,
    *,
    altloc_policy: AltlocPolicy = "highest_occupancy",
    disulfide_cutoff_angstrom: float = 2.2,
) -> InspectionSummary:
    return inspect_pdb_structure(
        input_structure,
        altloc_policy=altloc_policy,
        disulfide_cutoff_angstrom=disulfide_cutoff_angstrom,
    )
