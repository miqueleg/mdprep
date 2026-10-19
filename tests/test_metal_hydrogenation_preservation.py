"""The pre-MCPB coordinate check: tleap must hand back every input atom.

This file used to be about a ligand built under another name. A ligand whose
deposited name is one the force fields already define -- BesD's free lysine,
called LYS -- was renamed for the build, so tleap returned it as LY1 while the
input structure still called it LYS, and comparing the two spellings reported
every atom of that residue as deleted:

    Pre-MCPB tleap removed an input atom before MCPB.py: ('A', 'LYS', 301, None, 'N')

The fix then was to teach this check about the rename. The fix now is that
there is no rename: mdprep refuses the name and says to change it in the
input, so one residue has one name on both sides of tleap. The check keeps the
job it was always for, which is catching atoms tleap really did drop or move,
and water is the only representation rename left.
"""

from pathlib import Path

import pytest

from mdprep.metals.hydrogenate import (
    MetalHydrogenationError,
    _preservation_identity,
    _validate_preserved_input_coordinates,
)
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord


def _atom(name: str, resname: str, x: float = 0.0) -> AtomRecord:
    return AtomRecord(
        serial=1, name=name, altloc=None, resname=resname, chain_id="A", resid=301,
        icode=None, x=x, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0, element=name[0],
        record_name="HETATM", original_line="",
    )


def _structure(atoms: list[AtomRecord]) -> PdbStructure:
    residue = ResidueRecord(
        id=ResidueId(chain_id="A", resname=atoms[0].resname, resid=301, icode=None),
        atoms=atoms, record_names={"HETATM"}, original_index=0,
    )
    return PdbStructure(path=Path("x.pdb"), atoms=atoms, residues=[residue], model_count=1)


def test_an_unchanged_residue_passes():
    atoms = [_atom("N", "LIG"), _atom("CA", "LIG", 1.5)]

    assert _validate_preserved_input_coordinates(_structure(atoms), _structure(atoms)) == 0.0


def test_a_residue_coming_back_under_another_name_is_deleted_atoms():
    """Which is the right answer now. Nothing renames a ligand any more, so a
    different name on the far side of tleap is not a spelling to accept -- it
    is a residue that is not the one that went in."""

    reference = _structure([_atom("N", "LIG")])
    observed = _structure([_atom("N", "LG1")])

    with pytest.raises(MetalHydrogenationError, match="removed an input atom"):
        _validate_preserved_input_coordinates(reference, observed)


def test_a_moved_atom_is_caught():
    reference = _structure([_atom("N", "LIG")])
    observed = _structure([_atom("N", "LIG", x=0.5)])

    with pytest.raises(MetalHydrogenationError, match="moved an existing atom"):
        _validate_preserved_input_coordinates(reference, observed)


def test_a_missing_atom_is_caught():
    reference = _structure([_atom("N", "LIG"), _atom("CA", "LIG", 1.5)])
    observed = _structure([_atom("N", "LIG")])

    with pytest.raises(MetalHydrogenationError, match="removed an input atom"):
        _validate_preserved_input_coordinates(reference, observed)


def test_water_canonicalisation_is_the_one_rename_left():
    """tleap standardises retained HOH/H2O/TIP3/OPC input residues to WAT."""

    assert _preservation_identity(_atom("O", "HOH")) == _preservation_identity(
        _atom("O", "WAT")
    )
    assert _preservation_identity(_atom("O", "LIG")) != _preservation_identity(
        _atom("O", "LG1")
    )
