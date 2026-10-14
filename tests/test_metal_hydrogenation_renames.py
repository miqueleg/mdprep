"""The pre-MCPB coordinate check, when a ligand was built under another name.

A ligand whose deposited name is one the force fields already define is built
under a substitute, because loadpdb would otherwise map a free lysine called
LYS to the protein template. tleap therefore returns it as LY1 while the input
structure still calls it LYS. Comparing the two spellings reports every atom
of that residue as deleted, and the BesD example stopped with

    Pre-MCPB tleap removed an input atom before MCPB.py: ('A', 'LYS', 301, None, 'N')

before MCPB.py ever ran.
"""

from dataclasses import replace

from mdprep.leap.residues import LigandResidueRename
from mdprep.metals.hydrogenate import (
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
    from pathlib import Path

    return PdbStructure(path=Path("x.pdb"), atoms=atoms, residues=[residue], model_count=1)


RENAME = LigandResidueRename(
    ligand_id="a_lys_301", chain_id="A", resid=301, icode=None,
    original_resname="LYS", tleap_resname="LY1",
    reason="LYS is defined by the protein force fields",
)


def test_a_renamed_ligand_is_recognised_as_the_same_residue():
    reference = _structure([_atom("N", "LYS"), _atom("CA", "LYS", 1.5)])
    observed = _structure([_atom("N", "LY1"), _atom("CA", "LY1", 1.5)])

    deviation = _validate_preserved_input_coordinates(
        reference, observed, residue_renames=[RENAME]
    )

    assert deviation == 0.0


def test_without_the_rename_the_same_pair_looks_like_deleted_atoms():
    """The check is still doing its job; it just needed to be told."""

    import pytest

    from mdprep.metals.hydrogenate import MetalHydrogenationError

    reference = _structure([_atom("N", "LYS")])
    observed = _structure([_atom("N", "LY1")])

    with pytest.raises(MetalHydrogenationError, match="removed an input atom"):
        _validate_preserved_input_coordinates(reference, observed)


def test_a_renamed_ligand_that_actually_moved_is_still_caught():
    """Accepting the other spelling must not accept a moved atom with it."""

    import pytest

    from mdprep.metals.hydrogenate import MetalHydrogenationError

    reference = _structure([_atom("N", "LYS")])
    observed = _structure([_atom("N", "LY1", x=0.5)])

    with pytest.raises(MetalHydrogenationError, match="moved an existing atom"):
        _validate_preserved_input_coordinates(
            reference, observed, residue_renames=[RENAME]
        )


def test_a_renamed_ligand_missing_an_atom_is_still_caught():
    import pytest

    from mdprep.metals.hydrogenate import MetalHydrogenationError

    reference = _structure([_atom("N", "LYS"), _atom("CA", "LYS", 1.5)])
    observed = _structure([_atom("N", "LY1")])

    with pytest.raises(MetalHydrogenationError, match="removed an input atom"):
        _validate_preserved_input_coordinates(
            reference, observed, residue_renames=[RENAME]
        )


def test_water_canonicalisation_is_unaffected():
    """tleap standardises retained waters to WAT, which was already allowed."""

    assert _preservation_identity(_atom("O", "HOH")) == _preservation_identity(
        _atom("O", "WAT")
    )
