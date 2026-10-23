"""Modified residues replaced by their parents, through PDBFixer.

The fluorinase (PDB 1RQR) was solved with selenomethionine. Replacing it
keeps the backbone and the atoms the two share, drops the selenium, and
builds the sulfur in its place; every other atom comes back as it went in.

Chloroperoxidase (PDB 1CPO) begins with a pyroglutamate, whose side chain is
a lactam closed onto its own nitrogen. A glutamine shares its atom names but
not their places, so its side chain is built afresh beyond CB.

Skips without PDBFixer.
"""

from math import dist
from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.repair import StructureRepairError, repair_structure
from tests.test_config_validation import base_manifest

pytestmark = [pytest.mark.external, pytest.mark.openmm]

HETEROGENS = Path(__file__).parent / "data" / "heterogens"
SELENOMETHIONINE = HETEROGENS / "selenomethionine_1rqr.pdb"
PYROGLUTAMATE = HETEROGENS / "pyroglutamate_1cpo.pdb"


def _manifest(
    tmp_path, replacements, *, structure=SELENOMETHIONINE, add_missing_heavy_atoms=True
) -> ManifestConfig:
    data = base_manifest()
    data["project"]["input_structure"] = str(structure)
    data["project"]["output_dir"] = str(tmp_path / "out")
    data["structure"]["repair"] = {
        "backend": "pdbfixer",
        "add_missing_heavy_atoms": add_missing_heavy_atoms,
        "unresolved_residues": "leave_absent",
        "replace_modified_residues": replacements,
    }
    return ManifestConfig.model_validate(data)


MSE14 = {"selector": {"chain": "A", "resname": "MSE", "resid": 14}, "parent": "MET"}
PCA0 = {"selector": {"chain": "A", "resname": "PCA", "resid": 0}, "parent": "GLN"}


def test_the_selenomethionine_becomes_a_methionine(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")

    result = normalize_structure_stage(_manifest(tmp_path, [MSE14]), output_path=tmp_path / "n.pdb")

    residues = {residue.id.display(): residue for residue in result.normalized_structure.residues}
    assert "A:MSE14" not in residues
    methionine = residues["A:MET14"]
    assert "SD" in methionine.atom_names() and "SE" not in methionine.atom_names()
    # A residue of the protein now, built from the protein force field.
    assert methionine.record_names == {"ATOM"}
    report = result.to_report_dict()
    assert report["structure_repair"]["modified_residues_replaced"] == [
        {
            "chain": "A",
            "resname": "MSE",
            "resid": 14,
            "icode": None,
            "replacement": "MET",
            "side_chain_rebuilt": [],
        }
    ]


def test_every_atom_the_two_share_keeps_its_coordinates(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    before = {
        atom.name: (atom.x, atom.y, atom.z)
        for atom in read_pdb(SELENOMETHIONINE).atoms
        if atom.resid == 14
    }

    result = repair_structure(_manifest(tmp_path, [MSE14]), output_path=tmp_path / "r.pdb")

    after = {
        atom.name: (atom.x, atom.y, atom.z)
        for atom in result.structure.atoms
        if atom.resid == 14
    }
    for name in ("N", "CA", "C", "O", "CB", "CG", "CE"):
        assert after[name] == pytest.approx(before[name], abs=0.002)


def test_a_listed_residue_that_is_not_there_is_an_error(tmp_path):
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    missing = {"selector": {"chain": "A", "resname": "MSE", "resid": 99}, "parent": "MET"}

    with pytest.raises(StructureRepairError, match="not in the chains being built"):
        repair_structure(_manifest(tmp_path, [missing]), output_path=tmp_path / "r.pdb")


def test_replacing_a_residue_asks_for_nothing_else_to_be_built(tmp_path):
    """The planner asks to add missing atoms only when a deposition lacks some,
    so a replacement usually runs with add_missing_heavy_atoms false.

    What the parent has and the modified form lacked is built anyway, being
    the replacement. A chain's terminal OXT is not: it is left to tleap, as it
    is when nothing is repaired. The fragment ends at residue 17 without one.
    """
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")

    result = repair_structure(
        _manifest(tmp_path, [MSE14], add_missing_heavy_atoms=False),
        output_path=tmp_path / "r.pdb",
    )

    residues = {residue.id.display(): residue for residue in result.structure.residues}
    assert "SD" in residues["A:MET14"].atom_names()
    assert "OXT" not in residues["A:LEU17"].atom_names()


def test_a_pyroglutamate_opens_into_a_glutamine(tmp_path):
    """1CPO's PCA0: in the lactam CD is bonded to N, 1.47 A away. Kept there, a
    glutamine's CD would sit on its own backbone nitrogen."""
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")
    before = {
        atom.name: (atom.x, atom.y, atom.z)
        for atom in read_pdb(PYROGLUTAMATE).atoms
        if atom.resid == 0
    }
    assert dist(before["N"], before["CD"]) < 1.6

    result = repair_structure(
        _manifest(tmp_path, [PCA0], structure=PYROGLUTAMATE, add_missing_heavy_atoms=False),
        output_path=tmp_path / "r.pdb",
    )

    residues = {residue.id.display(): residue for residue in result.structure.residues}
    glutamine = {atom.name: (atom.x, atom.y, atom.z) for atom in residues["A:GLN0"].atoms}
    assert set(glutamine) == {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"}
    # Three bonds from N, as in any glutamine.
    assert dist(glutamine["N"], glutamine["CD"]) > 2.5
    # CB carries the chirality and stays where the crystal put it.
    for name in ("N", "CA", "C", "O", "CB"):
        assert glutamine[name] == pytest.approx(before[name], abs=0.002)
    (replaced,) = result.modified_residues_replaced
    assert replaced["side_chain_rebuilt"] == ["CD", "CG"]
    (warning,) = [text for text in result.warnings if "A:PCA0" in text]
    assert "rebuilt beyond CB" in warning and "minimize" in warning
