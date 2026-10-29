"""A phosphoserine kept in its chain, built through tleap.

SpoIIAA's SEP57 (PDB 1H4X, chain B residues 52-62) is prepared both ways a
kept residue can be: parameterised from its SMILES -- AM1-BCC, GAFF2 side
chain, the protein's backbone -- and from AmberTools' phosphoserine
templates. Each topology is read back: the residue is there, carries its
charge, and is bonded into the chain on both sides.

Skips without antechamber, parmchk2 or tleap, and the SMILES route without
RDKit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdprep.analysis.prmtop import read_prmtop
from mdprep.config.models import ManifestConfig
from mdprep.external.discovery import which_executable
from mdprep.residues.embedding import kept_residue_charges
from mdprep.residues.frcmod import is_gaff_type, junction_parameters
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.workflows.prepare import prepare_system
from tests.test_noncanonical_residues import _manifest, _smiles_block, _template_block

pytestmark = [pytest.mark.external, pytest.mark.ambertools, pytest.mark.tleap]

for _tool in ("antechamber", "parmchk2", "tleap"):
    if which_executable(_tool) is None:
        pytestmark.append(pytest.mark.skip(reason=f"{_tool} is required"))
        break


def _prepare(tmp_path: Path, blocks, **options) -> Path:
    data = _manifest(blocks, **options.pop("manifest", {}))
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["solvation"] = options.pop("solvation", {"enabled": False})
    path = tmp_path / "manifest.yaml"
    ManifestConfig.model_validate(data)
    import yaml

    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    prepare_system(path)
    return tmp_path / "prepared"


def _reports(out: Path, name: str) -> dict:
    return json.loads((out / "reports" / f"{name}.json").read_text(encoding="utf-8"))


def _sep_in(prmtop: Path, name: str = "SEP"):
    topology = read_prmtop(prmtop)
    (residue,) = [residue for residue in topology.residues if residue.name == name]
    return topology, residue


def test_a_phosphoserine_parameterised_from_its_smiles_builds_into_the_chain(tmp_path):
    pytest.importorskip("rdkit")

    out = _prepare(tmp_path, [_smiles_block()])

    validation = _reports(out, "validation_report")
    assert validation["errors"] == []
    (check,) = validation["noncanonical_residue_checks"]
    assert check == {
        "resname": "SEP",
        "built_as": "SEP",
        "expected_count": 1,
        "count": 1,
        "expected_charge": -2,
        "charges": [-2.0],
        "peptide_bonded": True,
        "ok": True,
    }
    topology, sep = _sep_in(out / "final" / "system.prmtop")
    atoms = {topology.atoms[index].name: topology.atoms[index] for index in sep.atom_indices}
    # The protein's types on the backbone and CB, GAFF2's beyond.
    assert {name: atoms[name].type for name in ("N", "H", "CA", "HA", "C", "O", "CB")} == {
        "N": "N", "H": "H", "CA": "CX", "HA": "H1", "C": "C", "O": "O", "CB": "CT",
    }
    assert all(is_gaff_type(atoms[name].type) for name in ("OG", "P", "O1P", "O2P", "O3P"))
    # Its backbone charges are those of ff14SB's anionic residues.
    assert (atoms["N"].charge, atoms["C"].charge) == pytest.approx((-0.5163, 0.5366), abs=1e-4)
    # The system: Asp56 -1, SEP57 -2, and the termini cancel.
    assert round(sum(atom.charge for atom in topology.atoms), 4) == -3.0

    # Nothing in the residue's frcmod is a term of the protein's own types:
    # every one it holds involves a GAFF2 type, and the phi torsion is not there.
    frcmod = (out / "residues" / "SEP" / "SEP.frcmod").read_text()
    kept, dropped = junction_parameters(frcmod)
    assert dropped == [] and "CT-os" in kept
    assert not any(line.startswith("C -N -CX-C") for line in frcmod.splitlines())
    residue = _reports(out, "residue_report")["noncanonical_residues"][0]
    assert residue["parameterisation"]["parameters_parmchk2_could_not_estimate"] == []
    assert residue["parameterisation"]["charges"]["backbone_set"] == "anionic"
    assert (out / "residues" / "SEP" / "tleap.log").is_file()


def test_ff19sb_gives_it_the_cmap_of_its_parent(tmp_path):
    pytest.importorskip("rdkit")

    out = _prepare(
        tmp_path,
        [_smiles_block(smiles="N[C@@H](COP(=O)([O-])[O-])C(=O)O")],
        manifest={"forcefield": "ff19SB"},
    )

    assert _reports(out, "validation_report")["errors"] == []
    frcmod = (out / "residues" / "SEP" / "SEP.frcmod").read_text()
    assert "%FLAG CMAP_RESLIST  1\nSEP\n" in frcmod
    topology, sep = _sep_in(out / "final" / "system.prmtop")
    alpha = next(topology.atoms[index] for index in sep.atom_indices if topology.atoms[index].name == "CA")
    assert (alpha.type, alpha.atomic_number) == ("XC", 6)
    parmed = pytest.importorskip("parmed")
    structure = parmed.load_file(str(out / "final" / "system.prmtop"))
    serine = next(cmap for cmap in structure.cmaps if cmap.atom3.residue.name == "SER")
    phospho = next(cmap for cmap in structure.cmaps if cmap.atom3.residue.name == "SEP")
    assert phospho.type.grid == serine.type.grid


@pytest.mark.parametrize(("template", "charge"), [("SEP", -2), ("S1P", -1)])
def test_amber_tools_template_builds_it_under_its_name(tmp_path, template, charge):
    out = _prepare(tmp_path, [_template_block(template)], solvation={"enabled": True, "buffer_angstrom": 6.0})

    validation = _reports(out, "validation_report")
    assert validation["errors"] == []
    (check,) = validation["noncanonical_residue_checks"]
    assert (check["built_as"], check["charges"], check["peptide_bonded"]) == (template, [float(charge)], True)
    assert "leaprc.phosaa14SB" in _reports(out, "tleap_report")["force_fields_sourced"]
    # Neutralised with the phosphate's charge counted.
    assert round(validation["parmed"]["total_charge"], 4) == 0.0
    renamed = _reports(out, "structure_report")["noncanonical_residues_kept"][0]
    assert renamed["final_resname"] == template and renamed["renamed"] == (template != "SEP")


def test_a_templates_charges_reach_the_histidine_embedding(tmp_path):
    data = _manifest([_template_block()])
    data["project"]["output_dir"] = str(tmp_path / "out")
    manifest = ManifestConfig.model_validate(data)
    structure = normalize_structure_stage(manifest).normalized_structure

    charges, missing = kept_residue_charges(structure, manifest)

    assert missing == []
    (values,) = charges.values()
    sep = next(residue for residue in structure.residues if residue.id.resname == "SEP")
    assert len(values) == len(sep.atoms) == 10
    assert sum(values) == pytest.approx(-2.0, abs=1e-5)
