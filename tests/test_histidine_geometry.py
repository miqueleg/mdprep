import pytest
from dataclasses import replace

from mdprep.protonation.histidine_geometry import (
    HistidineGeometryError,
    build_tautomer_cluster_model,
    build_tautomer_xyz_atoms,
    place_histidine_tautomer_hydrogen,
    write_xcontrol_fix_file,
)
from mdprep.structure.models import AtomRecord, ResidueRecord
from mdprep.structure.pdb import read_pdb


def histidine():
    structure = read_pdb("tests/data/protein_histidine_ring_hydrogenated.pdb")
    return next(residue for residue in structure.residues if residue.id.resname == "HIS")


def distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2) ** 0.5


def test_places_hid_hydrogen_on_nd1_at_expected_distance():
    residue = histidine()
    hydrogen = place_histidine_tautomer_hydrogen(residue, tautomer="HID")
    nd1 = next(atom for atom in residue.atoms if atom.name == "ND1")

    assert hydrogen.name == "HD1"
    assert distance(hydrogen, nd1) == pytest.approx(1.01)


def test_places_hie_hydrogen_on_ne2_at_expected_distance():
    residue = histidine()
    hydrogen = place_histidine_tautomer_hydrogen(residue, tautomer="HIE")
    ne2 = next(atom for atom in residue.atoms if atom.name == "NE2")

    assert hydrogen.name == "HE2"
    assert distance(hydrogen, ne2) == pytest.approx(1.01)


def test_degenerate_geometry_fails_clearly():
    residue = histidine()
    cg = next(atom for atom in residue.atoms if atom.name == "CG")
    ce1 = next(atom for atom in residue.atoms if atom.name == "CE1")
    nd1_index = next(index for index, atom in enumerate(residue.atoms) if atom.name == "ND1")
    residue.atoms[nd1_index] = replace(
        residue.atoms[nd1_index],
        x=(cg.x + ce1.x) / 2,
        y=(cg.y + ce1.y) / 2,
        z=(cg.z + ce1.z) / 2,
    )

    with pytest.raises(HistidineGeometryError):
        place_histidine_tautomer_hydrogen(residue, tautomer="HID")


def test_generated_cluster_preserves_input_hydrogens_adds_caps_and_tautomer_h():
    residue = histidine()
    model = build_tautomer_cluster_model([residue], residue, tautomer="HID")
    names = [atom.name for atom in model.atoms]

    assert "HA" in names
    assert "HB2" in names
    assert "HB3" in names
    assert "HD1" in names
    assert "HE2" not in names
    assert "HCA_NCAP" in names
    assert "HCA_CCAP" in names
    fixed_names = [model.atoms[index - 1].name for index in model.fixed_atom_indices]
    assert "CA" in fixed_names
    assert "CG" in fixed_names
    assert "ND1" in fixed_names
    assert "NE2" in fixed_names
    assert "HCA_NCAP" in fixed_names
    assert "HCA_CCAP" in fixed_names
    assert len(model.cap_atom_indices) == 2
    assert len(model.anchor_atom_indices) == 1


def test_target_histidine_input_imidazole_hydrogens_are_replaced_by_candidate():
    residue = _with_histidine_n_hydrogens(histidine())

    hid = build_tautomer_cluster_model([residue], residue, tautomer="HID")
    hie = build_tautomer_cluster_model([residue], residue, tautomer="HIE")

    assert [atom.name for atom in hid.atoms].count("HD1") == 1
    assert "HE2" not in [atom.name for atom in hid.atoms]
    assert [atom.name for atom in hie.atoms].count("HE2") == 1
    assert "HD1" not in [atom.name for atom in hie.atoms]


def test_missing_histidine_ring_carbon_hydrogen_is_added_temporarily():
    residue = histidine()
    residue = ResidueRecord(
        id=residue.id,
        atoms=[atom for atom in residue.atoms if atom.name != "HE1"],
        record_names=residue.record_names,
        original_index=residue.original_index,
    )

    model = build_tautomer_cluster_model([residue], residue, tautomer="HID")
    temporary = [atom for atom in model.atoms if atom.name == "HE1"]

    assert len(temporary) == 1
    assert temporary[0].source == "temporary_histidine_hydrogen"


def test_neighbor_histidine_assigned_state_ignores_ambiguous_input_n_hydrogens():
    target = histidine()
    neighbor = _shift_residue(_with_histidine_n_hydrogens(histidine()), resid=3, dx=2.0)

    model = build_tautomer_cluster_model(
        [target, neighbor],
        target,
        tautomer="HID",
        residue_states={id(neighbor): "HID"},
    )

    names = [atom.name for atom in model.atoms]
    assert names.count("HD1") == 2
    assert "HE2" not in names


def test_generated_xyz_uses_saturated_cluster_model():
    residue = histidine()
    atoms = build_tautomer_xyz_atoms([residue], residue, tautomer="HIE")
    names = [atom.name for atom in atoms]

    assert "HE2" in names
    assert "HD1" not in names
    assert "HCA_NCAP" in names
    assert "HCA_CCAP" in names


def test_backbone_hydrogen_named_hn_is_removed_with_truncated_backbone():
    structure = read_pdb("tests/data/protein_histidine_ring_hydrogenated.pdb")
    residue = next(residue for residue in structure.residues if residue.id.resname == "HIS")
    h_index = next(index for index, atom in enumerate(residue.atoms) if atom.name == "H")
    residue.atoms[h_index] = replace(residue.atoms[h_index], name="HN")

    model = build_tautomer_cluster_model([residue], residue, tautomer="HID")
    names = [atom.name for atom in model.atoms]

    assert "HN" not in names
    assert "HA" in names


def test_dehydrogenated_cluster_fails_clearly():
    structure = read_pdb("tests/data/protein_histidine_ring.pdb")
    residue = next(residue for residue in structure.residues if residue.id.resname == "HIS")

    with pytest.raises(HistidineGeometryError, match="requires a hydrogenated protein model"):
        build_tautomer_cluster_model([residue], residue, tautomer="HID")


def test_protonated_acid_state_without_carboxyl_h_fails_before_xtb():
    histidine_residue = histidine()
    structure = read_pdb("tests/data/protein_with_waters.pdb")
    asp = next(residue for residue in structure.residues if residue.id.resname == "ASP")

    with pytest.raises(HistidineGeometryError, match="assigned ASH"):
        build_tautomer_cluster_model(
            [histidine_residue, asp],
            histidine_residue,
            tautomer="HID",
            residue_states={id(asp): "ASH"},
        )


def test_xcontrol_fix_file_contains_fixed_atoms(tmp_path):
    path = tmp_path / "xtb.inp"
    write_xcontrol_fix_file([3, 17, 18], path)

    assert path.read_text(encoding="utf-8") == "$fix\n atoms: 3,17,18\n$end\n"


def _with_histidine_n_hydrogens(residue: ResidueRecord) -> ResidueRecord:
    nd1 = next(atom for atom in residue.atoms if atom.name == "ND1")
    ne2 = next(atom for atom in residue.atoms if atom.name == "NE2")
    atoms = list(residue.atoms)
    atoms.extend(
        [
            _hydrogen_from_anchor(nd1, "HD1", serial=9001),
            _hydrogen_from_anchor(ne2, "HE2", serial=9002),
        ]
    )
    return ResidueRecord(
        id=residue.id,
        atoms=atoms,
        record_names=residue.record_names,
        original_index=residue.original_index,
    )


def _hydrogen_from_anchor(anchor: AtomRecord, name: str, *, serial: int) -> AtomRecord:
    return replace(
        anchor,
        serial=serial,
        name=name,
        x=anchor.x + 1.0,
        y=anchor.y,
        z=anchor.z,
        element="H",
        original_line="",
    )


def _shift_residue(residue: ResidueRecord, *, resid: int, dx: float) -> ResidueRecord:
    shifted_atoms = [
        replace(atom, resid=resid, x=atom.x + dx)
        for atom in residue.atoms
    ]
    return ResidueRecord(
        id=replace(residue.id, resid=resid),
        atoms=shifted_atoms,
        record_names=residue.record_names,
        original_index=residue.original_index,
    )
