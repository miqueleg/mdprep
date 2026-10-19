"""The X-ray structures closest to a model, by sequence.

A predicted model has no crystal waters, and the closest crystal structure
is where they come from (``mdprep transplant-waters``). This asks the RCSB:
its search API runs MMseqs2 against every deposited protein sequence, and its
data API says what each hit contains -- resolution, how many waters were
modelled, which ligands are bound. The ranking is by sequence identity and
how much of the model the hit covers; a structure without waters is listed
but cannot donate any.

Querying sends the model's sequence to the RCSB. Nothing else leaves the
machine.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from mdprep.structure.classify import COORDINATING_METAL_ELEMENTS, is_standard_protein_residue
from mdprep.structure.models import PdbStructure

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"

_ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "ASH": "D", "CYS": "C",
    "CYM": "C", "CYX": "C", "GLN": "Q", "GLU": "E", "GLH": "E", "GLY": "G",
    "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L",
    "LYS": "K", "LYN": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Chemical component ID -> element, for the metals mdprep models. Element
# symbols are components too (FE, ZN); the rest are the common
# oxidation-state codes.
_METAL_COMPONENTS = {element: element for element in COORDINATING_METAL_ELEMENTS} | {
    "FE2": "FE", "FE3": "FE", "CU1": "CU", "MN3": "MN", "CO3": "CO",
    "NI2": "NI", "ZN2": "ZN",
}


class ClosestXrayError(RuntimeError):
    """Raised when the RCSB cannot be searched or answers unexpectedly."""


@dataclass(frozen=True)
class XrayHit:
    entry: str
    entity: str
    chains: tuple[str, ...]
    identity: float
    coverage: float
    evalue: float
    resolution: float | None
    waters: int | None
    ligands: tuple[str, ...]
    title: str

    @property
    def metals(self) -> tuple[str, ...]:
        return tuple(ligand for ligand in self.ligands if ligand in _METAL_COMPONENTS)

    @property
    def metal_elements(self) -> frozenset[str]:
        return frozenset(_METAL_COMPONENTS[ligand] for ligand in self.metals)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "entity": self.entity,
            "chains": list(self.chains),
            "sequence_identity": round(self.identity, 3),
            "coverage": round(self.coverage, 3),
            "evalue": self.evalue,
            "resolution_angstrom": self.resolution,
            "waters": self.waters,
            "ligands": list(self.ligands),
            "metals": list(self.metals),
            "title": self.title,
        }


def chain_sequences(structure: PdbStructure) -> dict[str, str]:
    """One-letter sequence of each protein chain, in residue order."""

    sequences: dict[str, list[str]] = {}
    for residue in structure.residues:
        if is_standard_protein_residue(residue):
            sequences.setdefault(residue.id.chain_id, []).append(
                _ONE_LETTER[residue.id.resname]
            )
    return {chain: "".join(letters) for chain, letters in sequences.items()}


Opener = Callable[[urllib.request.Request, float], Any]


def _post(url: str, payload: dict[str, Any], *, opener: Opener, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with opener(request, timeout) as response:
            status = getattr(response, "status", 200)
            body = response.read()
    except OSError as exc:
        raise ClosestXrayError(f"Could not reach the RCSB ({url}): {exc}") from exc
    # The search API answers 204 with no body when nothing matches.
    if status == 204 or not body:
        return {}
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ClosestXrayError(f"The RCSB returned something that is not JSON from {url}") from exc


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed hosts


def search_query(sequence: str, *, identity_cutoff: float, rows: int) -> dict[str, Any]:
    """The RCSB search request: similar sequences, X-ray structures only."""

    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "evalue_cutoff": 0.1,
                        "identity_cutoff": identity_cutoff,
                        "sequence_type": "protein",
                        "value": sequence,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            "scoring_strategy": "sequence",
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "results_verbosity": "verbose",
        },
    }


_ENTRY_QUERY = """
query($entries: [String!]!, $entities: [String!]!) {
  entries(entry_ids: $entries) {
    rcsb_id
    struct { title }
    rcsb_entry_info {
      resolution_combined
      deposited_solvent_atom_count
    }
    nonpolymer_entities { pdbx_entity_nonpoly { comp_id } }
  }
  polymer_entities(entity_ids: $entities) {
    rcsb_id
    rcsb_polymer_entity_container_identifiers { auth_asym_ids }
  }
}
"""


def _sequence_match(result: dict[str, Any]) -> dict[str, Any] | None:
    for service in result.get("services", []):
        if service.get("service_type") != "sequence":
            continue
        for node in service.get("nodes", []):
            contexts = node.get("match_context") or []
            if contexts:
                return contexts[0]
    return None


def search_closest_xray(
    sequence: str,
    *,
    identity_cutoff: float = 0.3,
    rows: int = 10,
    metal_elements: frozenset[str] | set[str] = frozenset(),
    timeout: float = 30.0,
    opener: Opener | None = None,
) -> list[XrayHit]:
    """X-ray structures with a chain similar to ``sequence``, closest first.

    Among equally close hits, one that holds every metal in
    ``metal_elements`` (element symbols, e.g. ``{"FE"}``) comes first: a
    crystal without the metal cannot have the water bound to it.
    """

    sequence = "".join(sequence.split()).upper()
    if len(sequence) < 20:
        raise ClosestXrayError(
            "The RCSB sequence search needs at least 20 residues; "
            f"this sequence has {len(sequence)}"
        )
    open_url = opener or _default_opener
    found = _post(
        SEARCH_URL,
        search_query(sequence, identity_cutoff=identity_cutoff, rows=rows),
        opener=open_url,
        timeout=timeout,
    )
    matches: list[tuple[str, dict[str, Any]]] = []
    for result in found.get("result_set", []):
        context = _sequence_match(result)
        if context is not None:
            matches.append((result["identifier"], context))
    if not matches:
        return []

    entities = [identifier for identifier, _ in matches]
    entries = sorted({identifier.split("_")[0] for identifier in entities})
    details = _post(
        GRAPHQL_URL,
        {"query": _ENTRY_QUERY, "variables": {"entries": entries, "entities": entities}},
        opener=open_url,
        timeout=timeout,
    ).get("data") or {}
    by_entry = {entry["rcsb_id"]: entry for entry in details.get("entries") or []}
    by_entity = {entity["rcsb_id"]: entity for entity in details.get("polymer_entities") or []}

    hits: list[XrayHit] = []
    for identifier, context in matches:
        entry_id = identifier.split("_")[0]
        entry = by_entry.get(entry_id, {})
        info = entry.get("rcsb_entry_info") or {}
        resolution = info.get("resolution_combined") or [None]
        chains = (
            (by_entity.get(identifier) or {})
            .get("rcsb_polymer_entity_container_identifiers", {})
            .get("auth_asym_ids")
            or []
        )
        query_length = context.get("query_length") or len(sequence)
        covered = (context.get("query_end") or 0) - (context.get("query_beg") or 1) + 1
        hits.append(
            XrayHit(
                entry=entry_id,
                entity=identifier,
                chains=tuple(chains),
                identity=float(context.get("sequence_identity") or 0.0),
                coverage=max(0.0, min(1.0, covered / query_length)),
                evalue=float(context.get("evalue") or 0.0),
                resolution=resolution[0],
                waters=info.get("deposited_solvent_atom_count"),
                ligands=tuple(
                    sorted(
                        {
                            (entity.get("pdbx_entity_nonpoly") or {}).get("comp_id")
                            for entity in entry.get("nonpolymer_entities") or []
                        }
                        - {None}
                    )
                ),
                title=(entry.get("struct") or {}).get("title") or "",
            )
        )
    wanted = frozenset(element.upper() for element in metal_elements)
    hits.sort(
        key=lambda hit: (
            -round(hit.identity, 2),
            not wanted <= hit.metal_elements,
            -round(hit.coverage, 2),
            hit.resolution if hit.resolution is not None else 99.0,
        )
    )
    return hits


def recommended(hits: list[XrayHit], metal_elements: frozenset[str] | set[str] = frozenset()) -> XrayHit | None:
    """The first hit with waters that holds the model's metals, else with waters."""

    wanted = frozenset(element.upper() for element in metal_elements)
    with_waters = [hit for hit in hits if hit.waters]
    for hit in with_waters:
        if wanted <= hit.metal_elements:
            return hit
    return with_waters[0] if with_waters else None
