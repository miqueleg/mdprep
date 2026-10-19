"""A water bound to a bonded metal.

The resting iron of an alpha-ketoglutarate enzyme often carries one: in
SyrB2 (PDB 2FCT) it sits where O2 later binds, beside the chloride. MCPB.py
makes it part of the site, which needs four things the protein path never
did: a mol2 template for the water, an H-O-H angle (TIP3P is rigid and has
none), a neutral charge component in the QM/MM refinement, and a three-point
water model. None of these need AmberTools.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mdprep.config.decisions import (
    DecisionError,
    build_manifest,
    plan_decisions,
    proposed_answers,
)
from mdprep.metals.mcpb import (
    McpbWorkflowError,
    _stage_bound_water_frcmod,
    _write_mcpb_water_mol2,
)
from mdprep.structure.pdb import read_pdb


DATA = Path("tests/data")


def _water(resname="HOH", names=("O", "H1", "H2")):
    path = DATA / "protein_zinc_aqua.pdb"
    oxygen = next(r for r in read_pdb(path).residues if r.id.resname == "HOH").atoms[0]
    atoms = [
        SimpleNamespace(name=name, x=oxygen.x + 0.1 * i, y=oxygen.y, z=oxygen.z)
        for i, name in enumerate(names)
    ]
    return SimpleNamespace(
        id=SimpleNamespace(resname=resname, display=lambda: f"W:{resname}601"),
        atoms=atoms,
    )


def _site(*coordinators):
    return SimpleNamespace(
        config=SimpleNamespace(id="fe_site"),
        bonds=[SimpleNamespace(coordinator_residue=residue) for residue in coordinators],
    )


def _manifest(water_model="TIP3P"):
    return SimpleNamespace(protein=SimpleNamespace(water_model=water_model))


# --------------------------------------------------------------------------
# What MCPB.py is given
# --------------------------------------------------------------------------


def test_the_water_template_carries_tip3p_types_and_charges(tmp_path):
    [path] = _write_mcpb_water_mol2(_site(_water()), manifest=_manifest(), work_dir=tmp_path)

    text = path.read_text(encoding="utf-8")
    atoms = text.split("@<TRIPOS>ATOM\n")[1].split("@<TRIPOS>BOND")[0].splitlines()
    assert [line.split()[1] for line in atoms] == ["O", "H1", "H2"]
    assert [line.split()[5] for line in atoms] == ["OW", "HW", "HW"]
    charges = [float(line.split()[-1]) for line in atoms]
    assert charges == [-0.834, 0.417, 0.417]
    assert sum(charges) == pytest.approx(0.0)
    assert path.name == "HOH.mol2"


def test_a_site_without_a_bound_water_gets_no_water_files(tmp_path):
    site = _site()

    assert _write_mcpb_water_mol2(site, manifest=_manifest(), work_dir=tmp_path) == []
    assert _stage_bound_water_frcmod(site, work_dir=tmp_path) == []


def test_a_four_point_water_cannot_join_the_quantum_model(tmp_path):
    with pytest.raises(McpbWorkflowError, match="three-point"):
        _write_mcpb_water_mol2(_site(_water()), manifest=_manifest("OPC"), work_dir=tmp_path)


def test_a_bound_water_without_hydrogens_fails_clearly(tmp_path):
    """Hydrogens are added before MCPB; one missing here is a pipeline fault,
    not something to type a template around."""

    with pytest.raises(McpbWorkflowError, match="no H1, H2"):
        _write_mcpb_water_mol2(
            _site(_water(names=("O",))), manifest=_manifest(), work_dir=tmp_path
        )


def test_the_bound_water_gets_the_angle_tip3p_never_had(tmp_path):
    """tleap stopped on HW-Y6-HW: MCPB.py copies the oxygen's terms to its
    new type, and rigid TIP3P has a bond but no angle to copy."""

    [path] = _stage_bound_water_frcmod(_site(_water("WAT")), work_dir=tmp_path)

    angle = path.read_text(encoding="utf-8").split("ANGLE\n")[1].splitlines()[0]
    assert angle.split()[:3] == ["HW-OW-HW", "100.00", "104.52"]


# --------------------------------------------------------------------------
# What the planner lets through
# --------------------------------------------------------------------------


BONDED_AQUA = {
    "structure.keep_crystal_waters": True,
    "structure.repair.add_missing_heavy_atoms": False,
    "metal.z_zn_500_site.model": "bonded_mcpb",
    "metal.z_zn_500_site.charge": 2,
    "metal.z_zn_500_site.confirm_donors": True,
    "metal.z_zn_500_site.multiplicity": 1,
    "metal.z_zn_500_site.model_charge": 2,
    "refinement.total_qm_multiplicity": 1,
}


def test_opc_with_a_bound_water_is_refused_before_colab():
    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")

    with pytest.raises(DecisionError, match="W:HOH601 is a water bound to a bonded metal"):
        build_manifest(
            plan,
            {**BONDED_AQUA, "protein.forcefield": "ff19SB", "protein.water_model": "OPC"},
        )


def test_opc_is_fine_when_the_metal_holds_no_water():
    plan = plan_decisions(DATA / "protein_histidine_zinc.pdb")
    answers = {
        "structure.repair.add_missing_heavy_atoms": False,
        "metal.z_zn_500_site.model": "bonded_mcpb",
        "metal.z_zn_500_site.charge": 2,
        "metal.z_zn_500_site.confirm_donors": True,
        "metal.z_zn_500_site.multiplicity": 1,
        "metal.z_zn_500_site.model_charge": 2,
        "refinement.total_qm_multiplicity": 1,
        "protein.forcefield": "ff19SB",
        "protein.water_model": "OPC",
    }

    assert build_manifest(plan, answers)["protein"]["water_model"] == "OPC"


def test_the_spin_proposal_counts_a_model_accepted_by_default():
    """The bonded model is the default for a metal with donors. Accepting it
    without typing it used to make every bonded-only question look
    unreachable, so 2FCT's iron got no spin proposal at all."""

    plan = plan_decisions(DATA / "protein_zinc_aqua.pdb")
    typed = {key: value for key, value in BONDED_AQUA.items() if not key.endswith(".model")}
    typed["metal.z_zn_500_site.multiplicity"] = 5

    assert proposed_answers(plan, typed)["refinement.total_qm_multiplicity"] == "5"


# --------------------------------------------------------------------------
# The QM/MM refinement's charge bookkeeping
# --------------------------------------------------------------------------


def test_a_bound_water_is_a_neutral_component_not_an_error():
    """It used to stop the refinement: "A non-protein metal coordinator must
    be configured as a ligand/cofactor"."""

    from mdprep.refinement.mapping import MappedResidue
    from mdprep.refinement.selection import _charge_components

    residues = read_pdb(DATA / "protein_zinc_aqua.pdb").residues
    pairs = [
        MappedResidue(
            input_index=index,
            topology_index=index,
            input_residue=residue,
            topology_residue=residue,
            anchor_rmsd_angstrom=0.0,
        )
        for index, residue in enumerate(residues)
    ]
    water = next(i for i, r in enumerate(residues) if r.id.display() == "W:HOH601")
    histidine = next(i for i, r in enumerate(residues) if r.id.resname == "HIS")

    components = _charge_components(
        pairs=pairs,
        manifest=SimpleNamespace(ligands=[], metals=[]),
        selected_ligand_residues={},
        selected_metal_residues={},
        selected_coordinating_ions={},
        coordinator_residue_indices={water, histidine},
    )

    by_label = {component.label: component for component in components}
    assert by_label["water:W:HOH601"].kind == "water"
    assert by_label["water:W:HOH601"].charge == 0
    assert "protein:A:HIS2" in by_label
