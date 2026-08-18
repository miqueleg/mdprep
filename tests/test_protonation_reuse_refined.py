from dataclasses import replace

from mdprep.protonation.apply import (
    ProtonationRecord,
    ProtonationResult,
    reuse_protonation_assignments_after_refinement,
)
from mdprep.protonation.histidine_geometry import place_histidine_tautomer_hydrogen
from mdprep.structure.models import AtomRecord, PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb


def test_reuse_initial_protonation_preserves_refined_hydrogen_coordinates(tmp_path):
    source = read_pdb("tests/data/protein_histidine_ring_hydrogenated.pdb")
    histidine = source.residues[0]
    placed = place_histidine_tautomer_hydrogen(histidine, tautomer="HID")
    hd1 = replace(
        histidine.atoms[0],
        serial=17,
        name="HD1",
        x=placed.x,
        y=placed.y,
        z=placed.z,
        element="H",
        original_line="",
    )
    atoms = [*histidine.atoms, hd1]
    refined = PdbStructure(
        path=tmp_path / "refined.pdb",
        atoms=atoms,
        residues=[
            ResidueRecord(
                id=histidine.id,
                atoms=atoms,
                record_names={"ATOM"},
                original_index=0,
            )
        ],
        model_count=1,
    )
    previous = ProtonationResult(
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "initial.pdb",
        method="propka_xtb_his",
        ph=7.0,
        structure=source,
        xtb_assignments_applied=[
            ProtonationRecord(
                chain="A",
                resid=2,
                icode=None,
                original_resname="HIS",
                final_resname="HID",
                source="propka_xtb_his",
                reason="test assignment",
            )
        ],
    )

    result = reuse_protonation_assignments_after_refinement(
        refined,
        previous,
        input_refined_pdb_path=refined.path,
        output_protonation_pdb_path=tmp_path / "final_protonation.pdb",
    )

    assert result.structure.residues[0].id.resname == "HID"
    preserved = next(atom for atom in result.structure.atoms if atom.name == "HD1")
    assert (preserved.x, preserved.y, preserved.z) == (placed.x, placed.y, placed.z)
    assert result.hydrogen_atoms_removed == 0
    assert result.reused_initial_assignments_after_refinement is True
    assert result.preserved_refined_hydrogen_count == sum(
        atom.element == "H" for atom in atoms
    )


def test_reuse_initial_accepts_normal_cysteine_s_h_bond_length(tmp_path):
    atoms = [
        AtomRecord(
            serial=1,
            name="SG",
            altloc=None,
            resname="CYS",
            chain_id="A",
            resid=65,
            icode=None,
            x=0.0,
            y=0.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="S",
            record_name="ATOM",
            original_line="",
        ),
        AtomRecord(
            serial=2,
            name="HG",
            altloc=None,
            resname="CYS",
            chain_id="A",
            resid=65,
            icode=None,
            x=1.34,
            y=0.0,
            z=0.0,
            occupancy=1.0,
            bfactor=0.0,
            element="H",
            record_name="ATOM",
            original_line="",
        ),
    ]
    refined = PdbStructure(
        path=tmp_path / "refined_cys.pdb",
        atoms=atoms,
        residues=[
            ResidueRecord(
                id=ResidueId(chain_id="A", resname="CYS", resid=65),
                atoms=atoms,
                record_names={"ATOM"},
                original_index=0,
            )
        ],
        model_count=1,
    )
    previous = ProtonationResult(
        input_normalized_pdb_path=tmp_path / "normalized.pdb",
        output_protonation_pdb_path=tmp_path / "initial.pdb",
        method="propka_xtb_his",
        ph=7.0,
        structure=refined,
    )

    result = reuse_protonation_assignments_after_refinement(
        refined,
        previous,
        input_refined_pdb_path=refined.path,
        output_protonation_pdb_path=tmp_path / "final_protonation.pdb",
    )

    assert result.preserved_refined_hydrogen_count == 1
