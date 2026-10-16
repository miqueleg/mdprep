"""Reading mmCIF, which is what structure prediction writes.

Boltz emits ModelCIF and the RCSB serves mmCIF, so a preparation tool that
only reads PDB makes a conversion step out of every predicted structure.

The reader is deliberately hand-written rather than built on a CIF library:
the web builder runs mdprep under Pyodide with pydantic and PyYAML and
nothing else, and a reader needing a compiled dependency would work
everywhere except where a student is most likely to be.
"""

from pathlib import Path

import pytest

from mdprep.structure.mmcif import MmcifParseError, read_mmcif, split_cif_row
from mdprep.structure.reader import read_structure

DATA = Path(__file__).parent / "data"
BOLTZ = DATA / "boltz_model.cif"


def test_a_boltz_model_is_read_into_the_same_structure_a_pdb_would_give():
    structure = read_mmcif(BOLTZ)

    assert len(structure.atoms) == 8
    assert [residue.id.resname for residue in structure.residues] == [
        "ALA",
        "FE2",
        "AKG",
    ]
    assert {residue.id.chain_id for residue in structure.residues} == {"A", "B", "C"}


def test_the_author_numbering_is_used_where_the_label_one_is_absent():
    """label_seq_id is null for every non-polymer, so the iron and the
    cofactor would have no residue number if it were trusted."""

    structure = read_mmcif(BOLTZ)
    iron = next(r for r in structure.residues if r.id.resname == "FE2")

    assert iron.id.resid == 1
    assert iron.atoms[0].name == "FE"
    assert iron.atoms[0].record_name == "HETATM"


def test_the_element_column_is_trusted_and_matches_the_pdb_reader():
    """A PDB file has to be read by column alignment to tell FE from F+E --
    which is how a deposited iron once came back as fluorine. mmCIF states
    the element, so for ordinary atoms there is nothing to infer.

    The one exception is a one-atom residue named after its element, where
    both readers settle the identity from the names rather than from a column
    a writer may have mangled. Either way the spelling has to match what the
    PDB reader produces for the same atom, or the same structure in two
    formats behaves differently downstream.
    """

    from mdprep.structure.pdb import read_pdb

    elements = {atom.name: atom.element for atom in read_mmcif(BOLTZ).atoms}
    assert elements["CA"] == "C"

    deposited = read_pdb(Path("examples/tutorials/besd_6nie/6NIE.pdb"))
    iron = next(atom for atom in deposited.atoms if atom.resname == "FE2")
    assert iron.element == elements["FE"]
    assert iron.element.upper() == "FE"


def test_quoted_values_survive_splitting():
    """A nucleotide atom name carries a prime and is quoted, so splitting on
    whitespace alone would lose it."""

    row = split_cif_row("""ATOM 1 C "O5'" . DA 1 ? 'a b' 1.0""")

    assert row == ["ATOM", "1", "C", "O5'", ".", "DA", "1", "?", "a b", "1.0"]


def test_an_unterminated_quote_is_refused():
    with pytest.raises(MmcifParseError, match="Unterminated quote"):
        split_cif_row('ATOM 1 C "O5 . DA')


def test_a_file_without_an_atom_site_loop_says_so(tmp_path):
    path = tmp_path / "no_atoms.cif"
    path.write_text("data_x\n_entry.id x\n#\n", encoding="utf-8")

    with pytest.raises(MmcifParseError, match="No _atom_site loop"):
        read_mmcif(path)


def test_a_ragged_row_is_refused(tmp_path):
    """Silently zipping a short row against the columns would misassign every
    value after the gap."""

    text = BOLTZ.read_text(encoding="utf-8").replace(
        "ATOM 1 N N . ALA 1 1 ? A 0.000 0.000 0.000 1 1 A ALA 91.5 1",
        "ATOM 1 N N . ALA 1 1 ? A 0.000 0.000",
    )
    path = tmp_path / "ragged.cif"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(MmcifParseError, match="values for"):
        read_mmcif(path)


def test_only_the_first_model_is_used_and_it_is_reported(tmp_path):
    lines = BOLTZ.read_text(encoding="utf-8").rstrip().splitlines()
    second = []
    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        fields = line.split()
        fields[-1] = "2"  # pdbx_PDB_model_num is the last column here
        second.append(" ".join(fields))
    assert second, "the fixture should have atom rows to duplicate"
    path = tmp_path / "two_models.cif"
    path.write_text("\n".join(lines[:-1] + second) + "\n#\n", encoding="utf-8")

    structure = read_mmcif(path)

    assert structure.model_count == 2
    assert len(structure.atoms) == 8
    assert any("using the first" in warning for warning in structure.warnings)


def test_the_reader_is_chosen_by_extension():
    from mdprep.structure.pdb import PdbParseError

    assert len(read_structure(BOLTZ).atoms) == 8
    assert read_structure(DATA / "protein_two_ligands.pdb").atoms

    with pytest.raises(PdbParseError, match="Unrecognised structure format"):
        read_structure(DATA / "boltz_model.xyz")


def test_a_predicted_structure_plans_the_same_way_a_deposited_one_does():
    """The point of reading mmCIF: a Boltz model goes into the decision
    engine with no conversion step, and the metal site is found from its
    coordinates like any other."""

    from mdprep.config.decisions import plan_decisions

    plan = plan_decisions(BOLTZ)

    assert plan.findings["ligands"]
    assert "FE2" in " ".join(plan.findings["metal_ions"])
    # The iron is proposed as a bonded site, as it would be from a PDB.
    model = next(
        decision
        for decision in plan.decisions
        if decision.section == "metals" and decision.id.endswith(".model")
    )
    assert model.default == "bonded_mcpb"


def test_the_same_structure_in_both_formats_reads_identically(tmp_path):
    """mmCIF and PDB must not be two different front doors to two different
    structures: everything downstream compares names, elements and numbers."""

    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.writer import write_pdb

    from_cif = read_mmcif(BOLTZ)
    round_trip = tmp_path / "from_cif.pdb"
    write_pdb(from_cif, round_trip)
    from_pdb = read_pdb(round_trip)

    def identity(structure):
        return [
            (a.record_name, a.name, a.resname, a.chain_id, a.resid, a.element,
             round(a.x, 3), round(a.y, 3), round(a.z, 3))
            for a in structure.atoms
        ]

    assert identity(from_cif) == identity(from_pdb)


# --------------------------------------------------------------------------
# Names that do not fit a PDB column
#
# mmCIF puts no width on a chain id or a residue name, and a PDB record gives
# them one column and three. Boltz uses both freedoms: it names a second
# SMILES ligand LIG2 and names chains after the entity. Everything downstream
# of mdprep reads fixed columns, so these have to be fitted on the way in.
# --------------------------------------------------------------------------

WIDE = DATA / "boltz_wide_names.cif"


def test_a_residue_name_too_wide_for_a_pdb_column_is_renamed():
    structure = read_mmcif(WIDE)

    names = {residue.id.resname for residue in structure.residues}
    assert all(len(name) <= 3 for name in names)
    # The two ligands stay distinguishable. Truncating both to LIG would be
    # the obvious way to make them fit and would merge the iron with the
    # cofactor.
    ligands = [
        residue.id.resname
        for residue in structure.residues
        if residue.id.resname not in {"HIS", "ASP"}
    ]
    assert len(ligands) == 2
    assert len(set(ligands)) == 2


def test_a_chain_id_too_wide_for_a_pdb_column_is_renamed():
    structure = read_mmcif(WIDE)

    chains = {residue.id.chain_id for residue in structure.residues}
    assert all(len(chain) == 1 for chain in chains)
    # The protein was already A and keeps it, so the renamed chains went
    # somewhere else rather than landing on top of it.
    assert len(chains) == 3


def test_the_renaming_is_reported_rather_than_done_quietly():
    """A selector written against the original file has to be translatable."""

    structure = read_mmcif(WIDE)
    reported = " ".join(structure.warnings)

    assert "LIG2" in reported
    assert "AKG" in reported
    for residue in structure.residues:
        if residue.id.resname not in {"HIS", "ASP"}:
            assert residue.id.resname in reported


def test_a_renamed_residue_is_not_given_a_name_that_means_something():
    """Classification is by residue name: a ligand renamed WAT becomes
    solvent and one renamed FE2 becomes a metal site. (A residue that was
    already an ion is the one exception, and keeps its element -- see
    test_a_wide_ion_name_keeps_the_element_it_names.)"""

    from mdprep.structure.classify import STANDARD_PROTEIN_RESIDUES, WATER_RESIDUES
    from mdprep.structure.pdb_fields import _free_resname, looks_like_ion_resname

    assert _free_resname("WATX", set()) != "WAT"
    assert not looks_like_ion_resname(_free_resname("FE2X", set()))
    for source in ("HISX", "HOHX", "LIG2", "SUBSTRATE"):
        chosen = _free_resname(source, set())
        assert chosen not in STANDARD_PROTEIN_RESIDUES | WATER_RESIDUES
        assert not looks_like_ion_resname(chosen)


def test_a_fitted_structure_survives_being_written_and_read_back():
    """The point of the renaming: what comes out is a PDB file whose columns
    line up, which is what tleap, antechamber and RDKit all read."""

    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.writer import write_pdb
    import tempfile

    structure = read_mmcif(WIDE)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "fitted.pdb"
        write_pdb(structure, output)
        reparsed = read_pdb(output)

    assert [residue.id for residue in reparsed.residues] == [
        residue.id for residue in structure.residues
    ]
    for original, rewritten in zip(structure.atoms, reparsed.atoms):
        assert (rewritten.x, rewritten.y, rewritten.z) == (
            original.x,
            original.y,
            original.z,
        )


def test_a_wide_ion_name_keeps_the_element_it_names():
    """An ion residue name is not a label: Amber matches ZN and FE2 against
    its ion templates, and mdprep reads the metal out of the same name. A
    stand-in that dropped the element would fit the column and lose the
    chemistry."""

    from mdprep.structure.pdb_fields import _free_resname, ion_element

    assert _free_resname("ZN22", set()) == "ZN"
    assert _free_resname("FE234", set()) == "FE"
    # Still unique when the obvious name is already in the structure.
    assert ion_element(_free_resname("ZN22", {"ZN"})) == "ZN"
    assert _free_resname("ZN22", {"ZN"}) != "ZN"


def test_an_mmcif_reaches_the_repair_stage_rather_than_being_refused(tmp_path):
    """The planner reads mmCIF, so the page accepts a prediction and writes a
    manifest for it -- and then `mdprep prepare` refused the same file at the
    repair stage, which reads the input directly. Reported only after the long
    part of a Colab run.
    """

    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.repair import _write_repair_input

    destination = tmp_path / "repair_input.pdb"
    _write_repair_input(
        WIDE,
        destination,
        chains=[],
        altloc_policy="highest_occupancy",
    )

    written = read_pdb(destination)
    source = read_mmcif(WIDE)
    assert len(written.atoms) == len(source.atoms)
    assert [atom.resname for atom in written.atoms] == [
        atom.resname for atom in source.atoms
    ]


def test_the_repair_input_keeps_only_the_chosen_chains(tmp_path):
    from mdprep.structure.pdb import read_pdb
    from mdprep.structure.repair import _write_repair_input

    destination = tmp_path / "chain_a.pdb"
    _write_repair_input(
        WIDE, destination, chains=["A"], altloc_policy="highest_occupancy"
    )

    assert {atom.chain_id for atom in read_pdb(destination).atoms} == {"A"}


# --------------------------------------------------------------------------
# What a prediction calls its non-polymers
#
# Boltz names them by position, not by chemistry: the iron is LIG1 and the
# cofactor beside it is LIG2. Neither fits the three characters a PDB record
# gives a residue name, and shortening them to LIG and L01 would hand the
# metal a generic ligand name -- which is exactly the thing tleap, MCPB and
# mdprep's own charge arithmetic all read the metal out of.
# --------------------------------------------------------------------------


def boltz_like(tmp_path, rows):
    header = (
        "data_model\n_entry.id model\n#\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
        "_atom_site.label_atom_id\n_atom_site.label_alt_id\n"
        "_atom_site.label_comp_id\n_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.auth_asym_id\n"
        "_atom_site.B_iso_or_equiv\n_atom_site.pdbx_PDB_model_num\n"
    )
    path = tmp_path / "prediction.cif"
    path.write_text(header + "\n".join(rows) + "\n#\n", encoding="utf-8")
    return path


def test_a_single_atom_residue_is_named_after_its_element(tmp_path):
    """LIG1 holding one iron is the iron, whatever the predictor called it."""

    path = boltz_like(
        tmp_path,
        [
            "HETATM 1 Fe FE1 . LIG1 1 ? 0.0 0.0 0.0 1 FE 80.0 1",
            "HETATM 2 Cl CL1 . LIG4 1 ? 5.0 0.0 0.0 1 CL 80.0 1",
        ],
    )

    names = {residue.id.resname for residue in read_mmcif(path).residues}

    assert names == {"FE", "CL"}


def test_a_multi_atom_prediction_ligand_keeps_a_distinguishable_name(tmp_path):
    """LIG2 and LIG3 must not both collapse onto LIG, and the digit that
    tells them apart is the part worth keeping."""

    path = boltz_like(
        tmp_path,
        [
            "HETATM 1 C C1 . LIG2 1 ? 0.0 0.0 0.0 1 K 80.0 1",
            "HETATM 2 O O1 . LIG2 1 ? 1.4 0.0 0.0 1 K 80.0 1",
            "HETATM 3 C C1 . LIG3 1 ? 8.0 0.0 0.0 1 S 80.0 1",
            "HETATM 4 O O1 . LIG3 1 ? 9.4 0.0 0.0 1 S 80.0 1",
        ],
    )

    names = [residue.id.resname for residue in read_mmcif(path).residues]

    assert len(set(names)) == 2, names
    assert all(len(name) <= 3 for name in names)
    # The trailing number survives, so a reader can follow which is which.
    assert {name[-1] for name in names} == {"2", "3"}


def test_the_metal_is_not_given_the_generic_ligand_name(tmp_path):
    """The case reported from a real prediction: the iron came out as LIG
    while the cofactor beside it took L01."""

    path = boltz_like(
        tmp_path,
        [
            "HETATM 1 Fe FE1 . LIG1 1 ? 0.0 0.0 0.0 1 FE 80.0 1",
            "HETATM 2 C C1 . LIG2 1 ? 5.0 0.0 0.0 1 K 80.0 1",
            "HETATM 3 O O1 . LIG2 1 ? 6.4 0.0 0.0 1 K 80.0 1",
        ],
    )
    structure = read_mmcif(path)

    iron = next(r for r in structure.residues if r.atoms[0].element == "Fe")
    assert iron.id.resname == "FE"
    assert "element of its only atom" in " ".join(structure.warnings)
