"""A bonded site fitted where the provisional force field puts it.

A Seminario fit needs a stationary point. An AlphaFold iron site fitted as a
singlet with only its hydrogens relaxed gave three Fe-O bonds a negative
force constant; minimising the site with the provisional 12-6-4 ion first
gave all six a positive one. ``geometry_source: ff_minimized_site`` does that:
it minimises the site, refuses to fit one that moved too far, fits it there,
and builds the parameters onto the original geometry.

These tests pin each part without AmberTools: the settings, the check that
the site is still the site, the original coordinates going back, the 12-6-4
table, and the minimiser holding everything it should.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig, McpbPySCFConfig, McpbSiteMinimizationConfig
from mdprep.metals.c4 import complete_polarizability_table
from mdprep.metals.coordination import ResolvedMetalBond, ResolvedMetalIon, ResolvedMetalSite
from mdprep.metals.site_minimization import _site_changes, put_back_original_coordinates
from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


def atom(serial, name, resname, resid, x, y=0.0, z=0.0, element=None):
    return AtomRecord(
        serial=serial, name=name, altloc=None, resname=resname, chain_id="A", resid=resid,
        icode=None, x=x, y=y, z=z, occupancy=1.0, bfactor=0.0,
        element=element or name[:1], record_name="HETATM", original_line="",
    )


def structure_of(atoms) -> PdbStructure:
    atoms = list(atoms)
    return PdbStructure(path=Path("site.pdb"), atoms=atoms, residues=build_residues(atoms), model_count=1)


# An iron, a water on it, and an acetate oxygen 3.2 A away.
SITE = structure_of(
    [
        atom(1, "FE", "FE2", 1, 0.0, element="Fe"),
        atom(2, "O", "HOH", 2, 2.10, element="O"),
        atom(3, "O1", "ACT", 3, 0.0, 3.2, element="O"),
        atom(4, "C1", "ACT", 3, 0.0, 4.4, element="C"),
    ]
)


def resolved(structure: PdbStructure) -> ResolvedMetalSite:
    from mdprep.config.models import MetalSiteConfig

    metal, water = structure.atoms[0], structure.atoms[1]
    by_key = {residue.id.resid: residue for residue in structure.residues}
    ion = ResolvedMetalIon(site_id="site", element="Fe", charge=2, atom=metal, residue=by_key[1])
    config = MetalSiteConfig.model_validate(
        {
            "id": "site",
            "model": "nonbonded",
            "ions": [
                {
                    "selector": {"chain": "A", "resname": "FE2", "resid": 1, "atom_name": "FE"},
                    "element": "Fe",
                    "charge": 2,
                }
            ],
            "nonbonded": {"parameter_set": "12_6_4"},
        }
    )
    return ResolvedMetalSite(
        config=config,
        ions=(ion,),
        bonds=(
            ResolvedMetalBond(
                ion=ion, coordinator=water, coordinator_residue=by_key[2],
                distance_angstrom=2.10, discovered_by_cutoff=False,
            ),
        ),
        additional_residues=(),
    )


def moved(structure: PdbStructure, changes: dict[int, tuple[float, float, float]]) -> PdbStructure:
    return structure_of(
        replace(a, x=changes[a.serial][0], y=changes[a.serial][1], z=changes[a.serial][2])
        if a.serial in changes
        else a
        for a in structure.atoms
    )


def changes_for(after: PdbStructure, **limits) -> dict[str, object]:
    index_of = {a.atom_identity: index for index, a in enumerate(SITE.atoms)}
    return _site_changes(
        resolved(SITE), SITE, after, index_of, McpbSiteMinimizationConfig(**limits), 2.6
    )


# --------------------------------------------------------------------------
# Is it still the site?
# --------------------------------------------------------------------------


def test_a_site_that_settles_a_little_is_fitted():
    report = changes_for(moved(SITE, {2: (2.02, 0.0, 0.0)}))

    assert report["problems"] == []
    assert report["bonds"] == [
        {"bond": "A:FE21@FE-A:HOH2@O", "before_angstrom": 2.1, "after_angstrom": 2.02}
    ]


def test_a_bond_that_changes_too_much_stops_the_fit():
    report = changes_for(moved(SITE, {2: (1.50, 0.0, 0.0)}))

    assert report["problems"] == ["A:FE21@FE-A:HOH2@O went from 2.10 to 1.50 A"]


def test_a_donor_that_leaves_stops_the_fit():
    report = changes_for(moved(SITE, {2: (2.80, 0.0, 0.0)}), max_bond_change_angstrom=1.0)

    assert report["problems"] == ["A:FE21@FE-A:HOH2@O is 2.80 A, outside the 2.60 A bond cutoff"]


def test_an_atom_that_joins_the_first_shell_stops_the_fit():
    report = changes_for(moved(SITE, {3: (0.0, 2.3, 0.0), 4: (0.0, 3.5, 0.0)}))

    assert report["problems"] == ["A:ACT3@O1 joined the first shell of A:FE21@FE at 2.30 A"]


def test_a_site_that_moves_too_far_stops_the_fit():
    report = changes_for(
        moved(SITE, {1: (0.5, 0.0, 0.0), 2: (2.6, 0.0, 0.0)}), max_rmsd_angstrom=0.3
    )

    assert report["problems"] == ["the site's heavy atoms moved 0.50 A RMS"]


def test_the_original_coordinates_go_back_after_the_fit(tmp_path):
    fitted = tmp_path / "fitted.pdb"
    write_pdb(moved(SITE, {1: (0.3, 0.1, 0.0), 2: (2.0, 0.2, 0.0)}), fitted)

    assert put_back_original_coordinates(fitted, SITE) == 2

    assert [(a.x, a.y, a.z) for a in read_pdb(fitted).atoms] == [
        (a.x, a.y, a.z) for a in SITE.atoms
    ]


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def pyscf(**fields) -> McpbPySCFConfig:
    return McpbPySCFConfig.model_validate({"method": "B3LYP", "basis": "6-31G*", **fields})


def test_the_minimised_site_comes_with_its_limits():
    config = pyscf(geometry_source="ff_minimized_site")

    assert config.site_minimization == McpbSiteMinimizationConfig()
    assert config.site_minimization.radius_angstrom == 5.0


def test_limits_are_only_for_the_minimised_site():
    with pytest.raises(ValueError, match="site_minimization may only be supplied when"):
        pyscf(geometry_source="qmmm_refinement", site_minimization={"radius_angstrom": 4.0})


def test_the_builder_fits_a_bonded_site_where_the_force_field_puts_it():
    from mdprep.config.decisions import build_manifest, plan_decisions
    from tests.test_oligomers import DEPOSITION, SHARED

    plan = plan_decisions(DEPOSITION)
    data = build_manifest(
        plan,
        {**SHARED, "metal.a_fe2_900_site.model": "bonded_mcpb", "structure.chains": "A"},
    )

    (site,) = [block for block in data["metals"] if block["model"] == "bonded_mcpb"]
    assert site["mcpb"]["pyscf"]["geometry_source"] == "ff_minimized_site"
    assert site["mcpb"]["provisional_nonbonded_parameter_set"] == "12_6_4"
    ManifestConfig.model_validate(data)


# --------------------------------------------------------------------------
# The 12-6-4 polarizability table
# --------------------------------------------------------------------------


def test_a_gaff2_type_the_table_lacks_takes_its_gaff_parents_value(tmp_path):
    table = tmp_path / "pol.dat"
    table.write_text("c3    1.061    Miller\nos    0.637    Miller\n", encoding="utf-8")

    completed = complete_polarizability_table(table, ["c3", "c6", "op", "Fe2+"], tmp_path / "out.dat")

    lines = completed.read_text(encoding="utf-8").splitlines()
    assert "c6    1.061    Taken from GAFF type c3 by mdprep" in lines
    assert "op    0.637    Taken from GAFF type os by mdprep" in lines


def test_a_complete_table_is_used_as_distributed(tmp_path):
    table = tmp_path / "pol.dat"
    table.write_text("c3    1.061    Miller\n", encoding="utf-8")

    assert complete_polarizability_table(table, ["c3"], tmp_path / "out.dat") == table


# --------------------------------------------------------------------------
# The minimiser holds what it should
# --------------------------------------------------------------------------


def test_the_minimiser_moves_only_the_atoms_it_is_given():
    pytest.importorskip("openmm", reason="OpenMM is required to minimise")
    from mdprep.refinement.openmm_minimizer import run_openmm_site_minimization

    amber = Path("tests/data/amber")
    import openmm.app as app

    inpcrd = app.AmberInpcrdFile(str(amber / "ace_ala_gly_nme.inpcrd"))
    before = [tuple(v) for v in inpcrd.positions.value_in_unit(__import__("openmm").unit.angstrom)]
    free = list(range(6, 16))

    run = run_openmm_site_minimization(
        prmtop_path=amber / "ace_ala_gly_nme.prmtop",
        inpcrd_path=amber / "ace_ala_gly_nme.inpcrd",
        movable=free,
        platform="CPU",
    )

    after = run.coordinates_angstrom
    held = [index for index in range(len(before)) if index not in free]
    assert all(
        max(abs(a - b) for a, b in zip(after[index], before[index])) < 1e-4 for index in held
    )
    assert any(
        max(abs(a - b) for a, b in zip(after[index], before[index])) > 1e-3 for index in free
    )
    assert run.final_energy_kj_mol < run.initial_energy_kj_mol
    assert run.movable_atoms == len(free)


def overlapping(tmp_path, first: int, second: int) -> Path:
    """The test peptide with atom ``second`` put exactly on atom ``first``."""

    parmed = pytest.importorskip("parmed", reason="ParmEd writes the test coordinates")
    amber = Path("tests/data/amber")
    system = parmed.load_file(str(amber / "ace_ala_gly_nme.prmtop"), str(amber / "ace_ala_gly_nme.inpcrd"))
    coordinates = system.coordinates
    coordinates[second] = coordinates[first]
    system.coordinates = coordinates
    path = tmp_path / "overlap.inpcrd"
    system.save(str(path), format="rst7", overwrite=True)
    return path


def test_two_held_atoms_on_each_other_do_not_stop_the_minimisation(tmp_path):
    """tleap rebuilt a lysine NZ onto a water hydrogen in SyrB2, far from the
    iron: the energy was NaN, though neither atom was ever going to move."""

    pytest.importorskip("openmm", reason="OpenMM is required to minimise")
    from mdprep.refinement.openmm_minimizer import run_openmm_site_minimization

    run = run_openmm_site_minimization(
        prmtop_path=Path("tests/data/amber/ace_ala_gly_nme.prmtop"),
        inpcrd_path=overlapping(tmp_path, 0, 28),
        movable=list(range(8, 16)),
        platform="CPU",
    )

    assert run.final_energy_kj_mol == run.final_energy_kj_mol  # not NaN


def test_an_atom_on_top_of_another_inside_the_site_stops_it(tmp_path):
    pytest.importorskip("openmm", reason="OpenMM is required to minimise")
    from mdprep.refinement.openmm_minimizer import (
        OpenmmRefinementError,
        run_openmm_site_minimization,
    )

    with pytest.raises(OpenmmRefinementError, match="apart inside the region being minimised"):
        run_openmm_site_minimization(
            prmtop_path=Path("tests/data/amber/ace_ala_gly_nme.prmtop"),
            inpcrd_path=overlapping(tmp_path, 8, 28),
            movable=list(range(8, 16)),
            platform="CPU",
        )
