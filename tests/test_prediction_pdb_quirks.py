"""What a prediction converted to PDB does that a deposited structure does not.

Every defect here came out of one real file: a Boltz model written to PDB,
run through the web builder, and stopped in Colab four times in a row, each
failure revealing the next. They are not specific to that file -- any
prediction with four-character residue names reaches all four -- but none
of them appeared in a fixture built by hand, because the hand-built ones
had three-character names and no CONECT.
"""

from pathlib import Path

import pytest

from mdprep.structure.pdb import read_pdb

DATA = Path("tests/data")


def write(tmp_path, lines: list[str], name: str = "pred.pdb") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\nEND\n", encoding="utf-8")
    return path


# A prediction names its non-polymers by position and writes the fourth
# character of the name into column 21, which the format leaves blank.
FOUR_CHAR = [
    "HETATM 2245  O5  LIG2A   1       5.400  -0.557   3.929  1.00 95.16          O",
    "HETATM 2250  C10 LIG2A   1       4.903  -1.694   3.788  1.00 98.54          C",
    "HETATM 2255 FE1  LIG1F   1       4.575   0.617   5.367  1.00 98.88          FE",
]


def test_a_four_character_residue_name_is_read_whole(tmp_path):
    """Reading three characters drops the one that tells them apart, and a
    prediction's LIG1 and LIG2 become the same residue name. When each is
    also residue 1 of its own chain, tleap merges them into one residue and
    the parameters stop meaning anything."""

    structure = read_pdb(write(tmp_path, FOUR_CHAR))

    names = {residue.id.resname for residue in structure.residues}
    assert len(names) == 2, names
    # And each is fitted to something a PDB can hold.
    assert all(len(name) <= 3 for name in names)


def test_a_lone_metal_keeps_its_element_through_the_renaming(tmp_path):
    """LIG1 holding one iron is the iron, whatever the predictor called it."""

    structure = read_pdb(write(tmp_path, FOUR_CHAR))

    iron = next(r for r in structure.residues if len(r.atoms) == 1)
    assert iron.id.resname == "FE"
    assert iron.atoms[0].element == "Fe"


def test_a_one_atom_ion_takes_its_element_from_its_residue_name(tmp_path):
    """tleap writes the element column from the Amber atom type, so Fe(II)
    comes back as " F", and the atom is called FE1 so the column convention
    agrees with it. Both halves say fluorine. The residue name is the only
    thing left that says iron, and a residue holding one atom cannot be
    describing anything else."""

    line = (
        "ATOM   4465  FE1 FE    294       4.575   0.617   5.367"
        "  1.00  0.00           F2+"
    )
    structure = read_pdb(write(tmp_path, [line]))

    assert structure.atoms[0].element == "Fe"


def test_the_mcpb_residue_key_is_the_whole_identity():
    """A prediction can put a ligand at residue 1 of the protein's own
    chain. Keyed on chain and number alone it is the protein's first
    residue, and the build stopped on an ambiguity that was only in the
    key -- ResidueId has carried the residue name all along."""

    from mdprep.metals.mcpb import write_mcpb_numbered_pdb
    import tempfile

    structure = read_pdb(DATA / "protein_histidine_zinc.pdb")
    with tempfile.TemporaryDirectory() as directory:
        mapping = write_mcpb_numbered_pdb(
            structure, Path(directory) / "numbered.pdb"
        )

    for key in mapping.mcpb_resid_by_original_key:
        assert len(key) == 4, key
    # Every residue of the structure is in the map, none lost to a collision.
    assert len(mapping.mcpb_resid_by_original_key) == len(structure.residues)


def test_partial_connectivity_is_dropped_rather_than_half_believed(tmp_path):
    """CONECT comes from the input file, written before mdprep added any
    hydrogens, so for a ligand that needed them it describes the heavy
    atoms and nothing else. antechamber believes it, types the hydrogens DU
    and the carbons they belonged to as sp -- a carbon that has lost its
    hydrogens looks triple-bonded. All of the ligand, or none of it."""

    from mdprep.ligands.extract import _ligand_conect_lines
    from mdprep.structure.models import AtomRecord

    def atom(serial, name, element):
        return AtomRecord(
            serial=serial, name=name, altloc="", resname="LIG", chain_id="A",
            resid=1, icode="", x=0.0, y=0.0, z=0.0, occupancy=1.0,
            bfactor=0.0, element=element, record_name="HETATM",
            original_line="",
        )

    source = tmp_path / "input.pdb"
    source.write_text("CONECT    1    2\nCONECT    2    1\nEND\n", encoding="utf-8")

    covered = [atom(1, "C1", "C"), atom(2, "O1", "O")]
    assert _ligand_conect_lines(source, covered), "complete records must be kept"

    # The same records, but the ligand now has a hydrogen mdprep placed and
    # nothing in the input file can be describing its bonds.
    with_hydrogen = covered + [atom(None, "H1", "H")]
    assert _ligand_conect_lines(source, with_hydrogen) == []
