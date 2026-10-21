"""A selenomethionine replaced by its methionine, through PDBFixer.

The fluorinase (PDB 1RQR) was solved with selenomethionine. Replacing it
keeps the backbone and the atoms the two share, drops the selenium, and
builds the sulfur in its place; every other atom comes back as it went in.
Skips without PDBFixer.
"""

from pathlib import Path

import pytest

from mdprep.config.models import ManifestConfig
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.repair import StructureRepairError, repair_structure
from tests.test_config_validation import base_manifest

pytestmark = [pytest.mark.external, pytest.mark.openmm]

SELENOMETHIONINE = Path(__file__).parent / "data" / "heterogens" / "selenomethionine_1rqr.pdb"


def _manifest(tmp_path, replacements) -> ManifestConfig:
    data = base_manifest()
    data["project"]["input_structure"] = str(SELENOMETHIONINE)
    data["project"]["output_dir"] = str(tmp_path / "out")
    data["structure"]["repair"] = {
        "backend": "pdbfixer",
        "add_missing_heavy_atoms": True,
        "unresolved_residues": "leave_absent",
        "replace_modified_residues": replacements,
    }
    return ManifestConfig.model_validate(data)


MSE14 = {"selector": {"chain": "A", "resname": "MSE", "resid": 14}, "parent": "MET"}


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
        {"chain": "A", "resname": "MSE", "resid": 14, "icode": None, "replacement": "MET"}
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
