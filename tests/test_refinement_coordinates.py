from dataclasses import replace

from mdprep.refinement.coordinates import transfer_optimized_coordinates
from mdprep.structure.models import PdbStructure, ResidueId, ResidueRecord
from mdprep.structure.pdb import read_pdb


def _residue(structure, resname):
    return next(item for item in structure.residues if item.id.resname == resname)


def _structure_with_atoms(path, atoms):
    grouped = {}
    for atom in atoms:
        grouped.setdefault(atom.residue_key, []).append(atom)
    residues = [
        ResidueRecord(
            id=ResidueId(chain_id=chain, resname=resname, resid=resid, icode=icode),
            atoms=residue_atoms,
            record_names={atom.record_name for atom in residue_atoms},
            original_index=index,
        )
        for index, ((chain, resname, resid, icode), residue_atoms) in enumerate(grouped.items())
    ]
    return PdbStructure(
        path=path,
        atoms=atoms,
        residues=residues,
        model_count=1,
        conect_bonds=structure_bonds(atoms),
    )


def structure_bonds(atoms):
    ligand = [atom for atom in atoms if atom.resname == "SUB"]
    if len(ligand) >= 2 and all(atom.serial is not None for atom in ligand[:2]):
        return {(ligand[0].serial, ligand[1].serial)}
    return set()


def test_coordinate_transfer_restores_generic_protonation_names_and_keeps_tleap_hydrogens(
    tmp_path,
):
    base = read_pdb("tests/data/protein_two_ligands.pdb")
    his_source = base.residues[0]
    ligand_source = _residue(base, "SUB")
    reference_atoms = [
        replace(atom, resname="HIS", chain_id="A", resid=10, serial=index + 1)
        for index, atom in enumerate(his_source.atoms[:2])
    ] + [
        replace(atom, serial=index + 3)
        for index, atom in enumerate(ligand_source.atoms)
    ]
    reference = _structure_with_atoms(tmp_path / "reference.pdb", reference_atoms)
    leap_atoms = [
        replace(atom, resname="HID") if atom.resname == "HIS" else atom
        for atom in reference_atoms
    ]
    leap_input = _structure_with_atoms(tmp_path / "leap_input.pdb", leap_atoms)
    topology_atoms = [
        replace(leap_atoms[0], chain_id="", resid=1, serial=1),
        replace(leap_atoms[1], chain_id="", resid=1, serial=2),
        replace(
            leap_atoms[0],
            name="HD1",
            element="H",
            chain_id="",
            resid=1,
            serial=3,
        ),
        *[
            replace(atom, chain_id="", resid=2, serial=index + 4)
            for index, atom in enumerate(leap_atoms[2:])
        ],
    ]
    topology = _structure_with_atoms(tmp_path / "topology.pdb", topology_atoms)
    coordinates = tuple(
        (float(index), float(index + 1), float(index + 2))
        for index in range(len(topology_atoms))
    )

    result = transfer_optimized_coordinates(
        reference_structure=reference,
        leap_input_structure=leap_input,
        topology_structure=topology,
        coordinates_angstrom=coordinates,
        output_path=tmp_path / "refined.pdb",
    )

    restored_his = result.structure.residues[0]
    restored_ligand = result.structure.residues[1]
    assert restored_his.id.resname == "HIS"
    assert restored_his.atom_names() == [reference_atoms[0].name, reference_atoms[1].name, "HD1"]
    assert result.generated_hydrogen_count == 1
    assert result.restored_residue_names[0]["optimized_provisional_resname"] == "HID"
    assert restored_ligand.atom_names() == [atom.name for atom in ligand_source.atoms]
    assert (restored_ligand.atoms[0].x, restored_ligand.atoms[0].y) == (3.0, 4.0)
