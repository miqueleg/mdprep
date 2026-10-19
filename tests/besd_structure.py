"""The BesD reference structure, in the form mdprep will accept.

6NIE is vendored verbatim -- it is the deposition, and the tests that measure
mdprep against a real file are worth nothing against an edited one. But its
substrate is a free lysine recorded as ``LYS A 301``, and mdprep refuses a
ligand whose residue name an Amber force field already defines, because tleap
resolves residues by name and would build it from the protein template.

So the renaming the user is told to do is done here, once per test session,
by the same code the GUI's rename button runs. Four copies of the complex,
four substrate residues; the chain's own twenty-eight lysines are untouched,
which is the whole reason the rename is by residue and not by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdprep.structure.pdb import read_pdb
from mdprep.structure.rename import rename_residues
from mdprep.structure.writer import write_pdb


DEPOSITION = Path("examples/tutorials/besd_6nie/6NIE.pdb")
SUBSTRATE_RESID = 301
SUBSTRATE_DEPOSITED_NAME = "LYS"
SUBSTRATE_NAME = "LIG"

needs_structure = pytest.mark.skipif(
    not DEPOSITION.exists(),
    reason="6NIE is not available locally; fetch it from RCSB to run this test",
)


def substrate_renames(structure) -> list[dict[str, object]]:
    """The rename mdprep asks for, as the GUI would send it."""

    return [
        {
            "chain": residue.id.chain_id,
            "resname": residue.id.resname,
            "resid": residue.id.resid,
            "icode": residue.id.icode,
            "new_resname": SUBSTRATE_NAME,
        }
        for residue in structure.residues
        if residue.id.resid == SUBSTRATE_RESID
        and residue.id.resname == SUBSTRATE_DEPOSITED_NAME
    ]


def write_prepared_besd(path: Path) -> Path:
    deposition = read_pdb(DEPOSITION)
    write_pdb(rename_residues(deposition, substrate_renames(deposition)), path)
    return path


@pytest.fixture(scope="session")
def besd_structure(tmp_path_factory) -> Path:
    """6NIE with its substrate renamed, written once for the whole session."""

    if not DEPOSITION.exists():
        pytest.skip("6NIE is not available locally; fetch it from RCSB")
    return write_prepared_besd(
        tmp_path_factory.mktemp("besd") / "6NIE_prepared.pdb"
    )
