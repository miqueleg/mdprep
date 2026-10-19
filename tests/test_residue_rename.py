"""Renaming a residue in the input, which is what mdprep asks the user to do.

The refusal is in test_leap_ligand_residue_names.py. This is the rename the
GUI offers alongside it, because the structure it matters most for is 6NIE --
fetched by PDB id, substrate deposited as LYS A 301 -- and telling a student
to go and edit a megabyte of PDB is where the workflow would end.

The property that makes it safe is that it renames residues and never names:
6NIE's chain A has a free lysine substrate and twenty-eight lysines of its
own, all called LYS.
"""

import pytest

from mdprep.config.decisions import ReservedResidueNameError, plan_decisions
from mdprep.structure.classify import free_residue_name
from mdprep.structure.pdb import read_pdb
from mdprep.structure.rename import ResidueRenameError, rename_residues
from mdprep.structure.writer import write_pdb
from tests.besd_structure import (
    DEPOSITION,
    SUBSTRATE_NAME,
    needs_structure,
    substrate_renames,
)


FIXTURE = "tests/data/protein_two_ligands.pdb"


def test_a_rename_touches_only_the_residue_it_names():
    structure = read_pdb(FIXTURE)
    target = next(r for r in structure.residues if r.id.resname == "SUB")

    renamed = rename_residues(
        structure,
        [
            {
                "chain": target.id.chain_id,
                "resname": "SUB",
                "resid": target.id.resid,
                "new_resname": "LIG",
            }
        ],
    )

    assert "SUB" not in {r.id.resname for r in renamed.residues}
    assert "LIG" in {r.id.resname for r in renamed.residues}
    # Everything else is the same structure, atom for atom.
    assert [a.name for a in renamed.atoms] == [a.name for a in structure.atoms]
    assert [(a.x, a.y, a.z) for a in renamed.atoms] == [
        (a.x, a.y, a.z) for a in structure.atoms
    ]


@needs_structure
def test_the_protein_keeps_its_own_lysines():
    """The reason the rename is by residue and not by name. Renaming every
    LYS in 6NIE would dissolve four chains of protein."""

    structure = read_pdb(DEPOSITION)
    before = [r.id.resname for r in structure.residues]

    renamed = rename_residues(structure, substrate_renames(structure))

    after = [r.id.resname for r in renamed.residues]
    assert before.count("LYS") == 32
    assert after.count("LYS") == 28
    assert after.count(SUBSTRATE_NAME) == 4
    # Same residues in the same order; only four names differ.
    assert sum(1 for a, b in zip(before, after) if a != b) == 4


@needs_structure
def test_the_renamed_structure_is_one_mdprep_will_plan(tmp_path):
    """Which is the whole point: the refusal goes away, and what the user
    has afterwards is a file that really is named that."""

    structure = read_pdb(DEPOSITION)
    path = tmp_path / "renamed.pdb"
    write_pdb(rename_residues(structure, substrate_renames(structure)), path)

    plan = plan_decisions(path)

    assert any("a_lig_301" in decision.id for decision in plan.decisions)


@needs_structure
def test_the_refusal_hands_back_exactly_what_the_rename_needs():
    """The page does not parse the message to work out what to do. mdprep
    knows which residues and what to call them, so it says so."""

    with pytest.raises(ReservedResidueNameError) as excinfo:
        plan_decisions(DEPOSITION)

    residues = excinfo.value.residues
    assert len(residues) == 4
    assert {item["chain"] for item in residues} == set("ABCD")
    assert {item["resid"] for item in residues} == {301}
    # One replacement for all four copies, not four unrelated ones.
    assert {item["new_resname"] for item in residues} == {"LIG"}

    structure = read_pdb(DEPOSITION)
    renamed = rename_residues(structure, residues)
    assert sum(1 for r in renamed.residues if r.id.resname == "LIG") == 4


def test_a_name_something_else_already_uses_is_refused():
    """Merging two residues under one template is the failure being fixed."""

    structure = read_pdb(FIXTURE)
    target = next(r for r in structure.residues if r.id.resname == "SUB")

    with pytest.raises(ResidueRenameError, match="already used"):
        rename_residues(
            structure,
            [
                {
                    "chain": target.id.chain_id,
                    "resname": "SUB",
                    "resid": target.id.resid,
                    "new_resname": "COF",
                }
            ],
        )


def test_a_residue_that_is_not_there_is_refused():
    with pytest.raises(ResidueRenameError, match="Not in this structure"):
        rename_residues(
            read_pdb(FIXTURE),
            [{"chain": "Z", "resname": "SUB", "resid": 999, "new_resname": "LIG"}],
        )


def test_the_suggested_name_avoids_everything_in_the_structure():
    taken = {"LIG", "LG1", "LG2", "ALA"}

    suggestion = free_residue_name(taken)

    assert suggestion not in taken
    assert free_residue_name({"ALA"}) == "LIG"


def test_the_suggested_name_is_never_one_a_force_field_defines():
    from mdprep.structure.classify import LEAP_RESERVED_RESIDUE_NAMES

    assert free_residue_name(set()) not in LEAP_RESERVED_RESIDUE_NAMES
