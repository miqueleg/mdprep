"""What a structure file declares about its own bonds and modified residues.

A deposition says which atoms are bonded across residues -- LINK records in a
PDB file, ``_struct_conn`` in an mmCIF -- and which polymer residues are
modified forms of a standard one -- MODRES, ``_pdbx_struct_mod_residue``.
Coordinates cannot: a predicted complex has clashes as short as a bond. The
readers keep both, under the names the reader settles on.
"""

from pathlib import Path

from mdprep.structure.links import parse_mmcif_links, parse_pdb_links, read_category_rows
from mdprep.structure.mmcif import read_mmcif
from mdprep.structure.pdb import read_pdb

DATA = Path(__file__).parent / "data" / "heterogens"


def _atom(serial, name, resname, resid, element, x, chain="A", record="ATOM", altloc=" ", occupancy=1.0):
    return (
        f"{record:<6}{serial:>5} {name:<4}{altloc}{resname:>3} {chain}{resid:>4}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f} 10.00          {element:>2}\n"
    )


def _link(name1, resname1, chain1, resid1, name2, resname2, chain2, resid2, sym2="1555"):
    return (
        f"LINK        {name1:<4} {resname1:>3} {chain1}{resid1:>4}                "
        f"{name2:<4} {resname2:>3} {chain2}{resid2:>4}     1555 {sym2:>6}  1.45  \n"
    )


def test_a_link_record_is_read_by_its_columns():
    (link,) = parse_pdb_links(
        [_link("ND2", "ASN", "A", 11, "C1", "NAG", "A", 404).rstrip("\n")]
    )

    assert link.first == ("A", "ASN", 11, None, "ND2")
    assert link.second == ("A", "NAG", 404, None, "C1")


def test_a_bond_to_a_symmetry_mate_is_not_a_bond_in_this_molecule():
    assert parse_pdb_links(
        [_link("O", "HOH", "A", 301, "NA", "NA", "A", 201, sym2="3655").rstrip("\n")]
    ) == []


def test_a_deposited_link_reaches_the_structure(tmp_path):
    structure = read_pdb(DATA / "covalent_intermediate_1zrn.pdb")

    (link,) = structure.links
    assert link.kind == "covalent"
    assert {link.first, link.second} == {
        ("A", "ASP", 10, None, "OD2"),
        ("A", "ACY", 300, None, "CH3"),
    }


def test_modres_names_the_parent_of_a_modified_residue():
    structure = read_pdb(DATA / "selenomethionine_1rqr.pdb")

    assert structure.modified_residue_parents == {("A", "MSE", 14, None): "MET"}


def test_a_link_to_a_metal_is_coordination(tmp_path):
    path = tmp_path / "zinc.pdb"
    path.write_text(
        _link("NE2", "HIS", "A", 1, "ZN", "ZN", "A", 101)
        + _atom(1, "NE2", "HIS", 1, "N", 0.0)
        + _atom(2, "ZN", "ZN", 101, "ZN", 2.1, record="HETATM")
        + "END\n",
        encoding="utf-8",
    )

    (link,) = read_pdb(path).links

    assert link.kind == "metal"


def test_a_link_follows_the_atom_the_altloc_policy_kept(tmp_path):
    """Two conformers share one identity; the link is to whichever stays."""

    path = tmp_path / "altloc.pdb"
    path.write_text(
        _link("ND2", "ASN", "A", 1, "C1", "NAG", "A", 2)
        + _atom(1, "ND2", "ASN", 1, "N", 0.0, altloc="A", occupancy=0.4)
        + _atom(2, "ND2", "ASN", 1, "N", 0.2, altloc="B", occupancy=0.6)
        + _atom(3, "C1", "NAG", 2, "C", 1.45, record="HETATM")
        + "END\n",
        encoding="utf-8",
    )

    (link,) = read_pdb(path).links

    assert link.first == ("A", "ASN", 1, None, "ND2")


def test_a_link_to_an_atom_that_is_not_there_is_dropped(tmp_path):
    path = tmp_path / "missing.pdb"
    path.write_text(
        _link("ND2", "ASN", "A", 1, "C1", "NAG", "A", 2)
        + _atom(1, "ND2", "ASN", 1, "N", 0.0)
        + "END\n",
        encoding="utf-8",
    )

    assert read_pdb(path).links == ()


MMCIF = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N ND2 . ASN A 11 ? 0.000 0.000 0.000 1.00 10.00 11 ASN A ND2 1
HETATM 2 C C1 . NAG B . ? 1.450 0.000 0.000 1.00 10.00 404 NAG A C1 1
HETATM 3 SE SE . MSE A 12 ? 9.000 0.000 0.000 1.00 10.00 12 MSE A SE 1
#
"""

STRUCT_CONN_LOOP = """loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.pdbx_ptnr1_PDB_ins_code
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.pdbx_ptnr2_PDB_ins_code
_struct_conn.ptnr1_auth_asym_id
_struct_conn.ptnr1_auth_seq_id
_struct_conn.ptnr2_auth_asym_id
_struct_conn.ptnr2_auth_seq_id
_struct_conn.ptnr2_symmetry
covale1 covale A ASN 11 ND2 ? 1_555 B NAG . C1 ? A 11 A 404 1_555
metalc1 metalc A ASN 11 ND2 ? 1_555 B NAG . C1 ? A 11 A 404 2_655
#
"""

# A deposition with one link writes the category as key-value pairs.
STRUCT_CONN_SINGLE = """_struct_conn.id covale1
_struct_conn.conn_type_id covale
_struct_conn.ptnr1_label_asym_id A
_struct_conn.ptnr1_label_comp_id ASN
_struct_conn.ptnr1_label_seq_id 11
_struct_conn.ptnr1_label_atom_id ND2
_struct_conn.ptnr2_label_asym_id B
_struct_conn.ptnr2_label_comp_id NAG
_struct_conn.ptnr2_label_seq_id .
_struct_conn.ptnr2_label_atom_id C1
_struct_conn.ptnr1_auth_asym_id A
_struct_conn.ptnr1_auth_seq_id 11
_struct_conn.ptnr2_auth_asym_id A
_struct_conn.ptnr2_auth_seq_id 404
#
"""

MOD_RESIDUE = """loop_
_pdbx_struct_mod_residue.id
_pdbx_struct_mod_residue.label_asym_id
_pdbx_struct_mod_residue.label_comp_id
_pdbx_struct_mod_residue.label_seq_id
_pdbx_struct_mod_residue.auth_asym_id
_pdbx_struct_mod_residue.auth_comp_id
_pdbx_struct_mod_residue.auth_seq_id
_pdbx_struct_mod_residue.PDB_ins_code
_pdbx_struct_mod_residue.parent_comp_id
_pdbx_struct_mod_residue.details
1 A MSE 12 A MSE 12 ? MET SELENOMETHIONINE
#
"""


def test_struct_conn_is_read_by_author_identifiers(tmp_path):
    path = tmp_path / "linked.cif"
    path.write_text(MMCIF + STRUCT_CONN_LOOP + MOD_RESIDUE, encoding="utf-8")

    structure = read_mmcif(path)

    # The second row is to a symmetry mate, so only one bond is in here.
    (link,) = structure.links
    assert link.kind == "covalent"
    assert {link.first, link.second} == {
        ("A", "ASN", 11, None, "ND2"),
        ("A", "NAG", 404, None, "C1"),
    }
    assert structure.modified_residue_parents == {("A", "MSE", 12, None): "MET"}


def test_a_single_struct_conn_written_as_key_values_is_read():
    (link,) = parse_mmcif_links(MMCIF + STRUCT_CONN_SINGLE)

    assert link.declared_kind == "covale"
    assert link.second == ("A", "NAG", 404, None, "C1")


def test_a_category_that_is_absent_has_no_rows():
    assert read_category_rows(MMCIF, "_struct_conn.") == []
