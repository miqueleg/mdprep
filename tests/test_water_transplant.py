"""Crystal waters for a predicted model.

The superposition, the metal-following and the filtering are plain numpy, so
these run without PyMOL; the PyMOL pairing has its own external test
(test_external_water_transplant.py).
"""

from pathlib import Path

import numpy as np
import pytest

from mdprep.structure import waters as transplant
from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.waters import (
    ReferenceMetal,
    ReferenceWater,
    Superposition,
    WaterTransplantError,
    add_waters,
    fit,
    follow_metals,
    kabsch,
    select_waters,
    tm_score,
)


def _rotation(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def _atom(name, resname, chain, resid, xyz, element, record="ATOM"):
    return AtomRecord(
        serial=None, name=name, altloc="", resname=resname, chain_id=chain, resid=resid,
        icode="", x=xyz[0], y=xyz[1], z=xyz[2], occupancy=1.0, bfactor=0.0,
        element=element, record_name=record, original_line="",
    )


def _structure(atoms):
    return PdbStructure(path=Path("model.pdb"), atoms=atoms, residues=build_residues(atoms), model_count=1)


# --------------------------------------------------------------------------
# Superposition
# --------------------------------------------------------------------------


def test_kabsch_recovers_a_known_motion():
    rng = np.random.default_rng(7)
    mobile = rng.normal(size=(20, 3)) * 10
    rotation = _rotation(37.0)
    target = mobile @ rotation.T + np.array([3.0, -2.0, 5.0])

    found_rotation, found_translation = kabsch(mobile, target)

    assert np.allclose(found_rotation, rotation, atol=1e-8)
    assert np.allclose(found_translation, [3.0, -2.0, 5.0], atol=1e-8)


def test_kabsch_never_returns_a_reflection():
    rng = np.random.default_rng(3)
    mobile = rng.normal(size=(12, 3))
    mirrored = mobile * np.array([1.0, 1.0, -1.0])

    rotation, _ = kabsch(mobile, mirrored)

    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_a_loop_placed_differently_does_not_tilt_the_core():
    """A predicted loop 6 A away from the crystal's must be left out of the
    fit, as PyMOL's align leaves it out, or every water moves with it."""

    rng = np.random.default_rng(11)
    core = rng.normal(size=(60, 3)) * 8
    loop = rng.normal(size=(8, 3)) * 3 + np.array([20.0, 0.0, 0.0])
    mobile = np.vstack([core, loop])
    target = mobile @ _rotation(20.0).T + np.array([1.0, 2.0, 3.0])
    target[60:] += np.array([6.0, 0.0, 0.0])

    result = fit(mobile, target, length=68)

    assert result.aligned == 60
    assert result.rmsd == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(result.apply(core[0]), target[0], atol=1e-6)


def test_tm_score_is_one_for_identity_and_normalised_by_the_model():
    assert tm_score(np.zeros(100), 100) == pytest.approx(1.0)
    # Half the model paired, perfectly: half the score.
    assert tm_score(np.zeros(50), 100) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Moving a bound water with its metal
# --------------------------------------------------------------------------


def test_a_bound_water_moves_with_its_metal():
    reference_metal = ReferenceMetal("A:FE502", "FE", (0.0, 0.0, 0.0), model_chain="A")
    model_iron = _atom("FE", "FE", "F", 1, (0.5, 0.0, 0.0), "FE", record="HETATM")
    bound = ReferenceWater("A:HOH702", (2.4, 0.0, 0.0), bound_to="A:FE502", model_chain="A")
    free = ReferenceWater("A:HOH800", (9.0, 0.0, 0.0), model_chain="A")

    moved, records = follow_metals([bound, free], [reference_metal], [model_iron])

    assert moved[0].xyz == pytest.approx((2.9, 0.0, 0.0))
    assert moved[1] == free
    assert records == [
        {
            "water": "A:HOH702",
            "reference_metal": "A:FE502",
            "model_metal": "F:FE1:FE",
            "shift_angstrom": 0.5,
        }
    ]


def test_a_metal_too_far_away_is_not_the_same_site():
    reference_metal = ReferenceMetal("A:FE502", "FE", (0.0, 0.0, 0.0), model_chain="A")
    model_iron = _atom("FE", "FE", "F", 1, (5.0, 0.0, 0.0), "FE", record="HETATM")
    bound = ReferenceWater("A:HOH702", (2.4, 0.0, 0.0), bound_to="A:FE502", model_chain="A")

    moved, records = follow_metals([bound], [reference_metal], [model_iron])

    assert moved == [bound] and records == []


def test_a_dimer_model_moves_each_copy_with_its_own_metal():
    """A monomeric reference fitted to both chains of a dimer yields the same
    labels twice; each copy follows the metal of its own chain."""

    metals = [
        ReferenceMetal("A:FE502", "FE", (0.0, 0.0, 0.0), model_chain="A"),
        ReferenceMetal("A:FE502", "FE", (50.0, 0.0, 0.0), model_chain="B"),
    ]
    model = [
        _atom("FE", "FE", "F", 1, (0.5, 0.0, 0.0), "FE", record="HETATM"),
        _atom("FE", "FE", "G", 1, (49.0, 0.0, 0.0), "FE", record="HETATM"),
    ]
    waters = [
        ReferenceWater("A:HOH702", (2.4, 0.0, 0.0), bound_to="A:FE502", model_chain="A"),
        ReferenceWater("A:HOH702", (52.4, 0.0, 0.0), bound_to="A:FE502", model_chain="B"),
    ]

    moved, _ = follow_metals(waters, metals, model)

    assert moved[0].xyz == pytest.approx((2.9, 0.0, 0.0))
    assert moved[1].xyz == pytest.approx((51.4, 0.0, 0.0))


# --------------------------------------------------------------------------
# Which waters fit
# --------------------------------------------------------------------------


def _site():
    """A histidine nitrogen, a carbon and an iron, far enough apart to test."""

    return _structure(
        [
            _atom("NE2", "HIS", "A", 10, (0.0, 0.0, 0.0), "N"),
            _atom("CB", "HIS", "A", 10, (10.0, 0.0, 0.0), "C"),
            _atom("FE", "FE", "F", 1, (0.0, 10.0, 0.0), "FE", record="HETATM"),
        ]
    )


def test_polar_and_apolar_neighbours_have_different_clash_distances():
    waters = [
        ReferenceWater("hbond", (2.7, 0.0, 0.0)),        # 2.7 A from N: an H-bond
        ReferenceWater("too_close_n", (2.3, 0.0, 0.0)),  # inside the nitrogen
        ReferenceWater("touching_c", (7.3, 0.0, 0.0)),   # 2.7 A from C: a clash
        ReferenceWater("beside_c", (13.3, 0.0, 0.0)),    # 3.3 A from C: fine
    ]

    selection = select_waters(_site(), waters)

    assert [water.label for water in selection.kept] == ["hbond", "beside_c"]
    assert {entry["water"] for entry in selection.clashing} == {"too_close_n", "touching_c"}


def test_a_water_on_the_metal_is_not_a_clash_and_is_reported_bound():
    selection = select_waters(_site(), [ReferenceWater("aqua", (0.0, 12.2, 0.0))])

    assert [water.label for water in selection.kept] == ["aqua"]
    assert selection.metal_bound == [
        {"water": "aqua", "metal": "F:FE1:FE", "distance_angstrom": 2.2}
    ]


def test_bulk_and_overlapping_waters_are_dropped():
    waters = [
        ReferenceWater("bulk", (30.0, 30.0, 30.0)),
        ReferenceWater("first", (0.0, 3.0, 0.0)),
        ReferenceWater("overlapping", (0.0, 3.0, 1.5)),
    ]

    selection = select_waters(_site(), waters)

    assert selection.distant == ["bulk"]
    assert [water.label for water in selection.kept] == ["first"]
    assert selection.clashing[0]["nearest"] == "another transplanted water"


def test_a_bound_water_wins_over_a_crystal_water_next_to_it():
    """Tried first, so a neighbour cannot crowd out the one that matters."""

    waters = [
        ReferenceWater("neighbour", (0.0, 12.0, 1.0)),
        ReferenceWater("aqua", (0.0, 12.2, 0.0), bound_to="A:FE502"),
    ]

    selection = select_waters(_site(), waters)

    assert [water.label for water in selection.kept] == ["aqua"]


def test_around_metal_keeps_only_the_active_site():
    waters = [ReferenceWater("aqua", (0.0, 12.2, 0.0)), ReferenceWater("far", (2.7, 0.0, 0.0))]

    selection = select_waters(_site(), waters, around_metal=5.0)

    assert [water.label for water in selection.kept] == ["aqua"]
    assert selection.outside_metal_radius == ["far"]


def test_waters_join_the_nearest_protein_chain_after_its_last_residue(tmp_path):
    from mdprep.structure.writer import write_pdb

    model = _structure(
        [
            _atom("CA", "ALA", "A", 7, (0.0, 0.0, 0.0), "C"),
            _atom("CA", "ALA", "B", 300, (40.0, 0.0, 0.0), "C"),
            _atom("FE", "FE", "F", 1, (2.0, 2.0, 0.0), "FE", record="HETATM"),
        ]
    )
    waters = [ReferenceWater("w1", (3.0, 0.0, 0.0)), ReferenceWater("w2", (37.0, 0.0, 0.0))]

    write_pdb(add_waters(model, waters), tmp_path / "out.pdb")

    added = [atom for atom in read_pdb(tmp_path / "out.pdb").atoms if atom.resname == "HOH"]
    assert [(atom.chain_id, atom.resid, atom.name) for atom in added] == [("A", 8, "O"), ("B", 301, "O")]


# --------------------------------------------------------------------------
# The whole operation, with the PyMOL part replaced
# --------------------------------------------------------------------------


def _identity_fit(chain="A", tm=0.95, identity=0.9):
    return transplant.ChainFit(
        model_chain="A",
        reference_chain=chain,
        superposition=Superposition(np.eye(3), np.zeros(3), 0.5, 200, 210, tm),
        identity=identity,
        model_ca=220,
    )


def _fake_reference(**overrides):
    values = dict(
        fits=[_identity_fit()],
        waters=[
            ReferenceWater("A:HOH702", (5.6, 5.4, 3.5), bound_to="A:ZN500", model_chain="A"),
            ReferenceWater("A:HOH800", (8.0, 1.0, 0.0), model_chain="A"),
        ],
        metals=[ReferenceMetal("A:ZN500", "ZN", (5.6, 5.4, 1.4), model_chain="A")],
        total_waters=2,
        bound_by_chain={"A": ["A:HOH702 on A:ZN500"]},
    )
    values.update(overrides)
    return transplant._Reference(**values)


def _run(tmp_path, monkeypatch, reference, **options):
    monkeypatch.setattr(transplant, "_superpose", lambda *args: reference)
    return transplant.transplant_waters(
        "tests/data/protein_histidine_zinc.pdb", tmp_path / "ref.pdb", tmp_path / "out.pdb", **options
    )


def test_the_transplanted_model_carries_the_bound_water(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, _fake_reference())

    assert [entry["water"] for entry in result.selection.metal_bound] == ["A:HOH702"]
    assert result.moved_with_metal[0]["shift_angstrom"] == pytest.approx(0.3, abs=0.01)
    written = read_pdb(tmp_path / "out.pdb")
    assert sum(1 for residue in written.residues if residue.id.resname == "HOH") == 2


def test_a_reference_of_another_fold_is_refused(tmp_path, monkeypatch):
    reference = _fake_reference(fits=[_identity_fit(tm=0.42)])

    with pytest.raises(WaterTransplantError, match="does not share a fold"):
        _run(tmp_path, monkeypatch, reference)
    assert not (tmp_path / "out.pdb").exists()


def test_a_distant_homologue_is_warned_about(tmp_path, monkeypatch):
    reference = _fake_reference(fits=[_identity_fit(tm=0.6, identity=0.2)])

    result = _run(tmp_path, monkeypatch, reference)

    assert any("TM-score of 0.60" in warning for warning in result.warnings)
    assert any("20% of the paired residues" in warning for warning in result.warnings)


def test_the_other_copy_with_a_bound_water_is_pointed_out(tmp_path, monkeypatch):
    """7E06 has a water on the iron of chain A and none on chain B's, and
    the better-fitting chain was B."""

    reference = _fake_reference(
        fits=[_identity_fit(chain="B")],
        waters=[ReferenceWater("B:HOH900", (8.0, 1.0, 0.0), model_chain="A")],
        bound_by_chain={"A": ["A:HOH702 on A:ZN500"]},
    )

    result = _run(tmp_path, monkeypatch, reference)

    assert any("other copies in the crystal do (chain A" in warning for warning in result.warnings)


def test_a_bound_water_that_no_longer_fits_is_explained(tmp_path, monkeypatch):
    """The model may put a ligand where the crystal has the water."""

    reference = _fake_reference(
        waters=[
            # 1.0 A from HIS2 NE2 at (3.6, 5.4, 1.1) once moved with the zinc.
            ReferenceWater("A:HOH702", (4.6, 5.4, 1.1), bound_to="A:ZN500", model_chain="A"),
        ],
        metals=[ReferenceMetal("A:ZN500", "ZN", (5.6, 5.4, 1.1), model_chain="A")],
    )

    result = _run(tmp_path, monkeypatch, reference)

    assert any(
        "A:HOH702 is bound to a metal in the reference but would sit" in warning
        for warning in result.warnings
    )


def test_assigned_names_survive_the_transplant(tmp_path, monkeypatch):
    """A predicted iron called LIG1 is read as FE, an assigned name. Read
    back cold, FE is the CCD code for Fe(III), and the planner proposed +3
    for an Fe(II) enzyme."""

    from mdprep.config.decisions import plan_decisions

    model = tmp_path / "model.pdb"
    lines = Path("tests/data/protein_histidine_zinc.pdb").read_text(encoding="utf-8").splitlines(True)
    model.write_text(
        "".join(
            line.replace("ZN    ZN Z 500", "FE1  LIG1Z   1").replace("          ZN", "          FE")
            if line.startswith("HETATM")
            else line
            for line in lines
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        transplant,
        "_superpose",
        lambda *args: _fake_reference(
            metals=[ReferenceMetal("A:FE500", "FE", (5.6, 5.4, 1.4), model_chain="A")],
            waters=[ReferenceWater("A:HOH800", (8.0, 1.0, 0.0), model_chain="A")],
            bound_by_chain={},
        ),
    )

    transplant.transplant_waters(model, tmp_path / "ref.pdb", tmp_path / "out.pdb")

    written = read_pdb(tmp_path / "out.pdb")
    assert "FE" in written.assigned_resnames
    charge = plan_decisions(tmp_path / "out.pdb").by_id("metal.z_fe_1_site.charge")
    assert charge.default is None
