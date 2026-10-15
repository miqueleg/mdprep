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


def test_the_element_column_is_taken_literally_and_matches_the_pdb_reader():
    """A PDB file has to be read by column alignment to tell FE from F+E --
    which is how a deposited iron once came back as fluorine. mmCIF states
    the element, so there is nothing to infer.

    The spelling has to match what the PDB reader produces for the same atom,
    or the same structure in two formats would behave differently downstream.
    """

    from mdprep.structure.pdb import read_pdb

    elements = {atom.name: atom.element for atom in read_mmcif(BOLTZ).atoms}
    assert elements["FE"] == "FE"
    assert elements["CA"] == "C"

    deposited = read_pdb(Path("examples/tutorials/besd_6nie/6NIE.pdb"))
    iron = next(atom for atom in deposited.atoms if atom.resname == "FE2")
    assert iron.element == elements["FE"]


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
