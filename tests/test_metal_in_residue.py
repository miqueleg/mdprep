"""A metal deposited inside a modified residue of a protein chain.

The PDB's chemical component dictionary marks such residues obsolete --
S-(methylmercury)cysteine (CMH), the lead-bound cysteine CSB -- but older
depositions and models made by hand still hold them. mdprep reads one the
way it reads a heme: the metal becomes a residue of its own, a metal site,
and what is left is the residue. In a chain the metal is written after the
chain, where it cannot end the chain early; and a residue whose metal is
bonded to more than the amino acid -- an organometallic group -- is left
whole and can only be replaced by its parent.

Built here from 1H4X's chain B: its Ser53 made a cysteine holding a zinc,
and one holding a methylmercury.
"""

from __future__ import annotations

from math import dist
from pathlib import Path

import pytest

from mdprep.config.decisions import plan_decisions
from mdprep.residues.capping import ResidueCappingError, capped_residue, chain_context
from mdprep.structure.pdb import read_pdb

FRAGMENT = Path("tests/data/heterogens/phosphoserine_1h4x.pdb")


def _hetatm(serial, name, resname, x, y, z, element):
    return (
        f"HETATM{serial:5d} {name:<4} {resname:>3} B  53    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00          {element:>2}"
    )


def _cysteine_holding(tmp_path: Path, resname: str, extra: list[tuple[str, str, float]]) -> Path:
    """Ser53 as a cysteine, deposited as ``resname`` with ``extra`` atoms on SG.

    Each extra atom is (name, element, distance), placed on from the atom
    before it along the CB->SG direction.
    """

    lines = []
    sulfur = None
    beta = None
    for line in FRAGMENT.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:20] == "SER" and line[22:26] == "  53":
            name = line[12:16].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            element = line[76:78].strip()
            if name == "OG":
                name, element = "SG", "S"
                sulfur = (x, y, z)
            if name == "CB":
                beta = (x, y, z)
            lines.append(_hetatm(int(line[6:11]), name, resname, x, y, z, element))
            if name == "SG":
                assert beta is not None
                direction = [s - b for s, b in zip(sulfur, beta)]
                length = dist(sulfur, beta)
                point = sulfur
                for offset, (extra_name, extra_element, distance) in enumerate(extra, start=1):
                    point = tuple(p + d / length * distance for p, d in zip(point, direction))
                    lines.append(_hetatm(1362 + offset * 100, extra_name, resname, *point, extra_element))
            continue
        lines.append(line)
    lines.insert(3, f"MODRES 1H4X {resname} B   53  CYS  MODIFIED RESIDUE")
    path = tmp_path / f"{resname.lower()}.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_metal_bound_only_to_its_residue_becomes_a_site_written_after_the_chain(tmp_path):
    structure = read_pdb(_cysteine_holding(tmp_path, "ZCY", [("ZN", "ZN", 2.3)]))

    names = [(residue.id.resname, residue.id.resid) for residue in structure.residues]
    assert names[-1] == ("ZN", 53) and ("ZCY", 53) in names
    (zcy,) = [residue for residue in structure.residues if residue.id.resname == "ZCY"]
    assert "ZN" not in zcy.atom_names()
    assert any("written after the chain" in warning for warning in structure.warnings)
    # Not between residues 53 and 54, where tleap would end the chain.
    order = [residue.id.resid for residue in structure.residues]
    assert order == [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 53]


def test_the_planner_asks_about_the_residue_and_the_metal_apart(tmp_path):
    plan = plan_decisions(_cysteine_holding(tmp_path, "ZCY", [("ZN", "ZN", 2.3)]))

    treatment = plan.by_id("residue.ZCY.treatment")
    assert [choice.value for choice in treatment.choices] == ["replace", "am1bcc", "qmmesp_pyscf"]
    donors = plan.by_id("metal.b_zn_53_site.confirm_donors")
    assert [choice.value for choice in donors.choices] == ["B:ZCY53"]
    # Replaced, it is a cysteine on a bonded zinc, and asked about as one.
    state = plan.by_id("metal.b_zn_53_site.donor_state.b_zcy_53")
    assert "replaced by CYS" in state.question
    assert [choice.value for choice in state.choices] == ["CYM", "CYS", "CYX"]
    assert ("residue.ZCY.treatment", ("replace",)) in [
        (condition.decision_id, condition.values) for condition in state.when
    ]


def test_an_organometallic_residue_is_left_whole_and_can_only_be_replaced(tmp_path):
    path = _cysteine_holding(tmp_path, "MHC", [("HG", "HG", 2.35), ("CM", "C", 2.1)])

    structure = read_pdb(path)
    (residue,) = [residue for residue in structure.residues if residue.id.resname == "MHC"]
    assert {"HG", "CM"} <= set(residue.atom_names())
    assert any("organometallic" in warning for warning in structure.warnings)

    treatment = plan_decisions(path).by_id("residue.MHC.treatment")
    assert [choice.value for choice in treatment.choices] == ["replace"]
    assert "Hg (HG)" in treatment.why and "organometallic" in treatment.why


def test_a_residue_holding_a_metal_is_not_parameterised_from_a_smiles(tmp_path):
    structure = read_pdb(_cysteine_holding(tmp_path, "MHC", [("HG", "HG", 2.35), ("CM", "C", 2.1)]))
    (residue,) = [residue for residue in structure.residues if residue.id.resname == "MHC"]

    with pytest.raises(ResidueCappingError, match="neither GAFF2 nor AM1-BCC describes"):
        capped_residue(
            chain_context(structure.residues, residue),
            "CC(=O)NC(CS[Hg]C)C(=O)NC",
            net_charge=0,
        )
