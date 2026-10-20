"""Finding the closest X-ray structure to a model, and the CLI around it.

No network: the RCSB is replaced by canned answers in the shape its search
and data APIs return. test_external_water_transplant.py covers PyMOL.
"""

import io
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mdprep.cli import app
from mdprep.structure import closest_xray
from mdprep.structure.closest_xray import (
    ClosestXrayError,
    chain_sequences,
    recommended,
    search_closest_xray,
    search_query,
)
from mdprep.structure.pdb import read_pdb


SEQUENCE = "MSTEKLAAGHFRDDVWLPQKEESYNIRCLTGHAWPEDKMN"
runner = CliRunner()


def _match(identity, begin=1, end=40):
    return {
        "service_type": "sequence",
        "nodes": [
            {
                "match_context": [
                    {
                        "sequence_identity": identity,
                        "evalue": 1e-30,
                        "query_beg": begin,
                        "query_end": end,
                        "query_length": 40,
                    }
                ]
            }
        ],
    }


SEARCH = {
    "result_set": [
        {"identifier": "1AAA_1", "services": [_match(0.62)]},
        {"identifier": "2BBB_1", "services": [_match(0.62)]},
        {"identifier": "3CCC_1", "services": [_match(0.95, begin=1, end=20)]},
    ]
}

DATA = {
    "data": {
        "entries": [
            {
                "rcsb_id": "1AAA",
                "struct": {"title": "apo, higher resolution"},
                "rcsb_entry_info": {"resolution_combined": [1.5], "deposited_solvent_atom_count": 300},
                "nonpolymer_entities": [{"pdbx_entity_nonpoly": {"comp_id": "GOL"}}],
            },
            {
                "rcsb_id": "2BBB",
                "struct": {"title": "with iron and alpha-ketoglutarate"},
                "rcsb_entry_info": {"resolution_combined": [2.1], "deposited_solvent_atom_count": 250},
                "nonpolymer_entities": [
                    {"pdbx_entity_nonpoly": {"comp_id": "FE2"}},
                    {"pdbx_entity_nonpoly": {"comp_id": "AKG"}},
                ],
            },
            {
                "rcsb_id": "3CCC",
                "struct": {"title": "a domain only, no waters"},
                "rcsb_entry_info": {"resolution_combined": [2.8], "deposited_solvent_atom_count": 0},
                "nonpolymer_entities": [],
            },
        ],
        "polymer_entities": [
            {"rcsb_id": "1AAA_1", "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A"]}},
            {"rcsb_id": "2BBB_1", "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A", "B"]}},
        ],
    }
}


class _Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_rcsb(search=SEARCH, data=DATA):
    sent: list[dict] = []

    def opener(request, timeout):
        payload = json.loads(request.data)
        sent.append({"url": request.full_url, "payload": payload})
        if request.full_url == closest_xray.SEARCH_URL:
            return _Response(json.dumps(search).encode() if search else b"")
        return _Response(json.dumps(data).encode())

    opener.sent = sent  # type: ignore[attr-defined]
    return opener


def test_the_search_asks_for_x_ray_structures_with_scores_attached():
    query = search_query(SEQUENCE, identity_cutoff=0.3, rows=5)

    nodes = query["query"]["nodes"]
    assert nodes[0]["service"] == "sequence"
    assert nodes[0]["parameters"]["identity_cutoff"] == 0.3
    assert nodes[1]["parameters"]["value"] == "X-RAY DIFFRACTION"
    # Top-level, the RCSB rejects it with a 400; it goes in request_options.
    assert "results_verbosity" not in query
    assert query["request_options"]["results_verbosity"] == "verbose"


def test_hits_carry_what_the_choice_depends_on():
    hits = search_closest_xray(SEQUENCE, opener=fake_rcsb())

    by_entry = {hit.entry: hit for hit in hits}
    iron = by_entry["2BBB"]
    assert iron.chains == ("A", "B")
    assert iron.identity == pytest.approx(0.62)
    assert iron.coverage == pytest.approx(1.0)
    assert iron.resolution == 2.1
    assert iron.waters == 250
    assert iron.ligands == ("AKG", "FE2")
    assert iron.metals == ("FE2",)
    assert by_entry["3CCC"].coverage == pytest.approx(0.5)


def test_closer_sequences_come_first_then_the_ones_with_the_metal():
    hits = search_closest_xray(SEQUENCE, metal_elements={"FE"}, opener=fake_rcsb())

    assert [hit.entry for hit in hits] == ["3CCC", "2BBB", "1AAA"]


def test_the_recommendation_needs_waters_and_prefers_the_metal():
    hits = search_closest_xray(SEQUENCE, metal_elements={"FE"}, opener=fake_rcsb())

    assert recommended(hits, {"FE"}).entry == "2BBB"
    assert recommended(hits).entry == "2BBB"  # 3CCC has no waters to give
    assert recommended([]) is None


def test_no_match_is_an_empty_list_not_an_error():
    assert search_closest_xray(SEQUENCE, opener=fake_rcsb(search=None)) == []


def test_a_fragment_is_refused_before_anything_is_sent():
    opener = fake_rcsb()

    with pytest.raises(ClosestXrayError, match="at least 20 residues"):
        search_closest_xray("MSTEK", opener=opener)
    assert opener.sent == []  # type: ignore[attr-defined]


def test_an_unreachable_rcsb_is_reported_clearly():
    def offline(request, timeout):
        raise OSError("network is unreachable")

    with pytest.raises(ClosestXrayError, match="Could not reach the RCSB"):
        search_closest_xray(SEQUENCE, opener=offline)


def test_the_sequence_comes_from_the_protein_chain_only():
    sequences = chain_sequences(read_pdb("tests/data/protein_histidine_zinc.pdb"))

    assert sequences == {"A": "AHD"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_closest_xray_cli_names_the_structure_to_use(monkeypatch):
    seen = {}

    def search(sequence, **options):
        seen.update(options, sequence=sequence)
        return search_closest_xray(SEQUENCE, opener=fake_rcsb(), metal_elements=options["metal_elements"])

    monkeypatch.setattr(closest_xray, "search_closest_xray", search)

    result = runner.invoke(app, ["closest-xray", "tests/data/protein_histidine_zinc.pdb", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["chain"] == "A"
    assert payload["model_metals"] == ["ZN"]
    # No hit holds zinc, so the closest one with waters is put forward.
    assert payload["recommended"] == "1AAA"
    assert seen["sequence"] == "AHD"


def test_transplant_waters_cli_writes_the_model_and_a_report(tmp_path, monkeypatch):
    from mdprep.structure import waters

    reference = tmp_path / "ref.pdb"
    reference.write_text("END\n", encoding="utf-8")
    calls = {}

    def fake_transplant(model, reference_path, output, **options):
        calls.update(options, model=model, reference=reference_path)
        Path(output).write_text("END\n", encoding="utf-8")
        return waters.WaterTransplantResult(
            model_path=Path(model),
            reference_path=Path(reference_path),
            output_path=Path(output),
            fits=[],
            reference_waters=10,
            candidate_waters=4,
            selection=waters.WaterSelection(),
            moved_with_metal=[],
            metal_displacements=[],
            warnings=["check me"],
        )

    monkeypatch.setattr(waters, "transplant_waters", fake_transplant)
    output = tmp_path / "with_waters.pdb"

    result = runner.invoke(
        app,
        [
            "transplant-waters", "tests/data/protein_histidine_zinc.pdb",
            "--reference", str(reference), "-o", str(output),
            "--reference-chain", "B", "--around-metal", "8",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "check me" in result.output
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["reference_waters"] == 10
    assert calls["reference_chain"] == "B"
    assert calls["around_metal"] == 8.0


def test_transplant_waters_cli_rejects_a_reference_that_is_neither_file_nor_id(tmp_path):
    result = runner.invoke(
        app,
        ["transplant-waters", "tests/data/protein_histidine_zinc.pdb", "--reference", "nope.pdb"],
    )

    assert result.exit_code == 1
    assert "neither a file nor a PDB ID" in result.output
