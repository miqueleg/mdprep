import pytest

from mdprep.leap.forcefields import (
    ForceFieldError,
    ligand_leaprc,
    protein_leaprc,
    water_box,
    water_leaprc,
)


def test_forcefield_mappings():
    assert protein_leaprc("ff14SB") == "leaprc.protein.ff14SB"
    assert protein_leaprc("ff19SB") == "leaprc.protein.ff19SB"
    assert water_leaprc("TIP3P") == "leaprc.water.tip3p"
    assert water_leaprc("OPC") == "leaprc.water.opc"
    assert water_box("TIP3P") == "TIP3PBOX"
    assert water_box("OPC") == "OPCBOX"
    assert ligand_leaprc("gaff") == "leaprc.gaff"
    assert ligand_leaprc("gaff2") == "leaprc.gaff2"


def test_unsupported_forcefield_fails():
    with pytest.raises(ForceFieldError):
        protein_leaprc("bad")
