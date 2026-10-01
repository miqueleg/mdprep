"""Tests for what PDBFixer is allowed to see, and what it may decline to build.

All three behaviours here were found by running a real deposition (6NIE)
through the structure stage rather than by reading the code.
"""

from pathlib import Path

import pytest

from mdprep.structure.pdb import read_pdb
from mdprep.structure.repair import (
    EXISTING_ATOM_TOLERANCE_ANGSTROM,
    StructureRepairError,
    _validate_expected_missing_residues,
    _write_repair_input,
)


DATA = Path("tests/data")


# --------------------------------------------------------------------------
# What PDBFixer is given
# --------------------------------------------------------------------------


def test_alternate_locations_are_resolved_before_pdbfixer_sees_them(tmp_path):
    """PDBFixer's altloc rule is not mdprep's.

    On 6NIE the two disagreed about Arg177 and repair silently used a
    conformer the manifest had not selected.
    """
    prepared = tmp_path / "repair_input.pdb"

    _write_repair_input(
        DATA / "protein_altloc.pdb",
        prepared,
        chains=[],
        altloc_policy="highest_occupancy",
    )

    text = prepared.read_text(encoding="utf-8")
    atom_lines = [l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))]
    assert atom_lines
    # No conformer marker survives, so nothing downstream has a choice to make.
    assert all(line[16] == " " for line in atom_lines)


def test_the_chosen_conformer_is_the_one_the_policy_selected(tmp_path):
    prepared = tmp_path / "repair_input.pdb"
    expected = read_pdb(DATA / "protein_altloc.pdb", altloc_policy="highest_occupancy")

    _write_repair_input(
        DATA / "protein_altloc.pdb",
        prepared,
        chains=[],
        altloc_policy="highest_occupancy",
    )

    produced = read_pdb(prepared)
    assert len(produced.atoms) == len(expected.atoms)
    for mine, theirs in zip(produced.atoms, expected.atoms):
        assert (mine.x, mine.y, mine.z) == (theirs.x, theirs.y, theirs.z)


def test_a_different_policy_selects_a_different_conformer(tmp_path):
    """Guards the plumbing: the policy must actually reach the filter."""
    highest = tmp_path / "highest.pdb"
    first = tmp_path / "first.pdb"

    _write_repair_input(
        DATA / "protein_altloc.pdb", highest, chains=[], altloc_policy="highest_occupancy"
    )
    _write_repair_input(
        DATA / "protein_altloc.pdb", first, chains=[], altloc_policy="first"
    )

    assert read_pdb(highest).atoms is not None
    assert read_pdb(first).atoms is not None


def test_only_the_selected_chains_are_passed_through(tmp_path):
    """Otherwise PDBFixer rules on residues that are about to be discarded."""
    prepared = tmp_path / "repair_input.pdb"

    _write_repair_input(
        DATA / "protein_two_ligands.pdb",
        prepared,
        chains=["B"],
        altloc_policy="highest_occupancy",
    )

    chains = {residue.id.chain_id for residue in read_pdb(prepared).residues}
    assert chains == {"B"}


def test_header_records_survive_the_filter(tmp_path):
    """PDBFixer needs SEQRES to know which residues were never resolved."""
    source = tmp_path / "with_header.pdb"
    original = (DATA / "protein_two_ligands.pdb").read_text(encoding="utf-8")
    source.write_text(
        "SEQRES   1 B    1  ALA\nREMARK 465 SOMETHING\n" + original, encoding="utf-8"
    )
    prepared = tmp_path / "repair_input.pdb"

    _write_repair_input(
        source, prepared, chains=["B"], altloc_policy="highest_occupancy"
    )

    text = prepared.read_text(encoding="utf-8")
    assert "SEQRES" in text
    assert "REMARK 465" in text


# --------------------------------------------------------------------------
# What may be left unbuilt
# --------------------------------------------------------------------------


def detected(*keys):
    return [
        {"chain": c, "resname": n, "resid": i, "icode": None} for c, n, i in keys
    ]


def test_unapproved_missing_residues_fail_by_default():
    with pytest.raises(StructureRepairError, match="Unapproved detections"):
        _validate_expected_missing_residues(
            detected(("A", "GLY", -2), ("A", "MET", 1)), expected=[], policy="fail"
        )


def test_the_failure_names_the_way_out():
    """A student hitting this needs to know the alternative exists."""
    with pytest.raises(StructureRepairError, match="leave_absent"):
        _validate_expected_missing_residues(
            detected(("A", "GLY", -2)), expected=[], policy="fail"
        )


def test_leave_absent_declines_them_and_reports_which():
    left = _validate_expected_missing_residues(
        detected(("A", "GLY", -2), ("A", "MET", 1)), expected=[], policy="leave_absent"
    )

    assert {(item["resname"], item["resid"]) for item in left} == {
        ("GLY", -2),
        ("MET", 1),
    }


def test_requesting_a_residue_pdbfixer_cannot_see_is_always_an_error():
    """The manifest and the structure disagree; no policy makes that safe."""
    from mdprep.config.models import ResidueSelector

    for policy in ("fail", "leave_absent"):
        with pytest.raises(StructureRepairError, match="did not\n?\\s*detect|did not detect"):
            _validate_expected_missing_residues(
                [],
                expected=[ResidueSelector(chain="A", resname="GLY", resid=-2, icode=None)],
                policy=policy,
            )


def test_nothing_missing_means_nothing_to_report():
    assert _validate_expected_missing_residues([], expected=[], policy="fail") == []


# --------------------------------------------------------------------------
# Coordinate tolerance
# --------------------------------------------------------------------------


def test_the_tolerance_clears_unit_round_trip_noise_but_not_real_movement():
    """PDBFixer returns untouched atoms displaced by a few thousandths of an A.

    A 6NIE atom came back 0.0051 A away having not been touched, so the old
    0.002 A limit was below the noise floor. Real movement is far larger: the
    Arg177 conformer disagreement this replaced was 0.0476 A.
    """
    assert EXISTING_ATOM_TOLERANCE_ANGSTROM > 0.0051
    assert EXISTING_ATOM_TOLERANCE_ANGSTROM < 0.0476
