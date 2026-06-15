import pytest

from mdprep.protonation.propka_parser import (
    PropkaParseError,
    PropkaRecord,
    map_propka_records,
    parse_propka_file,
    parse_propka_line,
)
from mdprep.structure.pdb import read_pdb


def test_parse_propka_file_reads_supported_residue_lines():
    records = parse_propka_file("tests/data/propka_sample.pka")

    assert [record.resname for record in records] == ["ASP", "GLU", "HIS", "LYS", "CYS", "ARG"]
    assert records[0].resid == 25
    assert records[0].chain_id == "A"
    assert records[0].pka == 3.80


def test_parse_propka_line_ignores_headers_and_malformed_lines():
    assert parse_propka_line("The Determinants section") is None
    assert parse_propka_line("ASP missing fields") is None


def test_parse_propka_line_supports_blank_chain_marker():
    record = parse_propka_line("ASP  25 -   3.80")

    assert record is not None
    assert record.chain_id == ""


def test_parse_propka_file_prefers_summary_section_when_present(tmp_path):
    pka = tmp_path / "realistic.pka"
    pka.write_text(
        "\n".join(
            [
                "HIS   2 A   6.43     0 %   -0.07    8",
                "SUMMARY OF THIS PREDICTION",
                "       Group      pKa  model-pKa   ligand atom-type",
                "   HIS   2 A     6.43       6.50",
                "--------------------------------------------------------------------------------------------------------",
                "",
            ]
        ),
        encoding="utf-8",
    )

    records = parse_propka_file(pka)

    assert len(records) == 1
    assert records[0].resname == "HIS"
    assert records[0].pka == 6.43


def test_duplicate_or_insertion_code_mapping_fails_clearly(tmp_path):
    pdb = tmp_path / "icode_ambiguous.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ASP A  25A      0.000   0.000   0.000  1.00 20.00           N",
                "ATOM      2  CA  ASP A  25A      1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  N   ASP A  25B      0.000   1.000   0.000  1.00 20.00           N",
                "ATOM      4  CA  ASP A  25B      1.000   1.000   0.000  1.00 20.00           C",
                "END",
                "",
            ]
        ),
        encoding="utf-8",
    )
    structure = read_pdb(pdb)

    with pytest.raises(PropkaParseError) as excinfo:
        map_propka_records(structure, [PropkaRecord("ASP", 25, "A", 4.0, "ASP 25 A 4.0")])

    assert "ambiguously" in str(excinfo.value)
