"""Grouping atoms into residues, stated once.

Seven modules had a private ``_build_residues``, in three spellings, all
doing the same thing: group by ``atom.residue_key`` in first-appearance
order, carry the record names, number from zero. Every stage that rewrites a
structure's atoms -- protonation, repair, metal hydrogenation, MCPB, the leap
input, the refinement coordinates -- has to rebuild its residues afterwards,
and each had grown its own copy.

Nothing had drifted yet. The point is that nothing could tell you if it had:
``original_index`` is what downstream code orders residues by, and a copy
that numbered from one, or grouped without the insertion code, would have
produced a structure that looked right and mapped wrong.
"""

import re
from pathlib import Path

import pytest

from mdprep.structure.models import build_residues
from mdprep.structure.pdb import read_pdb


SOURCE = Path("src/mdprep")
FIXTURES = [
    "tests/data/protein_two_ligands.pdb",
    "tests/data/protein_histidine_zinc.pdb",
    "tests/data/protein_with_hydrogenated_free_lysine.pdb",
]


def test_there_is_exactly_one_definition():
    """_build_residues_like in structure/repair.py is a different function:
    it rebuilds residues against an existing residue list, not from atoms."""

    found = [
        f"{path}:{number}"
        for path in SOURCE.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if re.match(r"def _?build_residues\(", line)
    ]

    assert found == ["src/mdprep/structure/models.py:" + found[0].split(":")[1]], found


@pytest.mark.parametrize("fixture", FIXTURES)
def test_it_reproduces_what_the_reader_built(fixture):
    """The reader is the reference: whatever it says a residue list is, a
    stage that rewrites atoms and rebuilds must say the same."""

    structure = read_pdb(fixture)

    rebuilt = build_residues(structure.atoms)

    assert [residue.id for residue in rebuilt] == [
        residue.id for residue in structure.residues
    ]
    assert [residue.original_index for residue in rebuilt] == [
        residue.original_index for residue in structure.residues
    ]
    assert [residue.record_names for residue in rebuilt] == [
        residue.record_names for residue in structure.residues
    ]
    assert [residue.atom_names() for residue in rebuilt] == [
        residue.atom_names() for residue in structure.residues
    ]


def test_residues_are_numbered_from_zero_in_first_appearance_order():
    """What original_index means, pinned, because it is what orders
    residues everywhere downstream."""

    structure = read_pdb(FIXTURES[0])

    rebuilt = build_residues(structure.atoms)

    assert [residue.original_index for residue in rebuilt] == list(
        range(len(rebuilt))
    )


def test_the_insertion_code_separates_residues():
    """Two residues at the same number are two residues. A copy that grouped
    without the icode would merge them and renumber everything after."""

    from dataclasses import replace

    structure = read_pdb(FIXTURES[0])
    first = structure.residues[0]
    inserted = [replace(atom, icode="A") for atom in first.atoms]

    rebuilt = build_residues(list(first.atoms) + inserted)

    assert len(rebuilt) == 2
    assert [residue.id.icode for residue in rebuilt] == [None, "A"]
