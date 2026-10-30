"""A phosphoserine kept in its chain, built through tleap.

SpoIIAA's SEP57 (PDB 1H4X, chain B residues 52-62) is prepared both ways a
kept residue can be: parameterised from its SMILES -- AM1-BCC, GAFF2 side
chain, the protein's backbone -- and from AmberTools' phosphoserine
templates. Each topology is read back: the residue is there, carries its
charge, and is bonded into the chain on both sides.

Skips without antechamber, parmchk2 or tleap, and the SMILES route without
RDKit. The QMMESP fit also needs PySCF, ParmEd, respgen and resp.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdprep.ambertools.resp import amber_resp_available
from mdprep.analysis.prmtop import read_prmtop
from mdprep.config.models import ManifestConfig
from mdprep.external.discovery import which_executable
from mdprep.residues.charges import BACKBONE_CHARGES
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
        "refitted_by_mcpb": {},
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


def test_its_charges_can_be_fitted_to_the_esp_the_protein_polarises(tmp_path):
    for module in ("rdkit", "pyscf", "parmed"):
        pytest.importorskip(module)
    if not amber_resp_available():
        pytest.skip("respgen and resp are required")

    out = _prepare(
        tmp_path,
        [
            _smiles_block(
                parameters="qmmesp_pyscf",
                qmmesp={"method": "HF", "basis": "6-31G*", "num_threads": 2},
            )
        ],
    )

    validation = _reports(out, "validation_report")
    assert validation["errors"] == []
    (check,) = validation["noncanonical_residue_checks"]
    assert (check["charges"], check["peptide_bonded"]) == ([-2.0], True)
    residue = _reports(out, "residue_report")["noncanonical_residues"][0]
    fit = residue["qmmesp"]
    # Embedded in everything but the residue and the two its caps stand for.
    embedding = fit["embedding"]
    assert embedding["excluded_residues"] == ["B:ASP56", "B:SER58"]
    assert embedding["point_charge_count_after_cutoff"] > 50
    # And the peptide bonds joining those two to the chain beyond.
    assert embedding["boundary_atoms_excluded"] == ["MET@C", "MET@O", "GLY@N", "GLY@H"]
    assert embedding["min_distance"] > 1.0
    assert fit["pyscf_result"]["converged"] is True
    # RESP held the backbone and the caps; the residue keeps only its own.
    assert set(fit["fit"]["fixed_charges"]) == {"N", "H", "C", "O"}
    assert [len(group["atoms"]) for group in fit["fit"]["charge_groups"]] == [6, 6]
    fitted = fit["charges"]["charges"]
    assert set(fitted) == set(residue["parameterisation"]["atom_types"])
    assert fitted != residue["parameterisation"]["charges"]["charges"]
    # And those are the charges the final topology carries.
    topology, sep = _sep_in(out / "final" / "system.prmtop")
    built = {topology.atoms[index].name: topology.atoms[index].charge for index in sep.atom_indices}
    assert built == pytest.approx(fitted, abs=1e-4)
    assert {name: built[name] for name in ("N", "H", "C", "O")} == pytest.approx(
        BACKBONE_CHARGES["anionic"], abs=1e-4
    )
    # The provisional AM1-BCC library was only ever the provisional system's.
    tleap_input = (out / "leap" / "dry" / "tleap.in").read_text()
    assert "provisional_am1bcc" not in tleap_input
    assert "provisional_am1bcc" in (out / "qmmesp" / "provisional_leap" / "tleap.in").read_text()


def test_gla_is_fitted_in_the_field_of_the_calcium_it_binds(tmp_path):
    """Osteocalcin's Gla residues kept, with QMMESP charges, and its Ca2+ nonbonded.

    The fit's environment holds the three calcium ions at their formal +2,
    which is what polarising a metal-binding residue in place is for; the
    topology is the residues bonded into the chain beside the ions.
    """

    for module in ("rdkit", "pyscf", "parmed"):
        pytest.importorskip(module)
    if not amber_resp_available():
        pytest.skip("respgen and resp are required")
    import yaml

    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "examples" / "23_osteocalcin_gla_calcium.yaml").read_text())
    data["project"]["input_structure"] = str(
        root / "examples" / "tutorials" / "osteocalcin_1q8h" / "1Q8H.pdb"
    )
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    # Keep the test to AmberTools, PySCF and ParmEd: His35 stated rather than
    # decided by PropKa and xTB, and no water box.
    data["protonation"] = {
        "ph": 7.0,
        "method": "manual_only",
        "overrides": [
            {
                "selector": {"chain": "A", "resname": "HIS", "resid": 35},
                "state": "HIE",
                "reason": "stated for the test",
            }
        ],
    }
    data["solvation"] = {"enabled": False}
    (block,) = data["noncanonical_residues"]
    block["parameters"] = "qmmesp_pyscf"
    block["qmmesp"] = {"method": "HF", "basis": "6-31G*", "num_threads": 4}
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    prepare_system(path)

    out = tmp_path / "prepared"
    validation = _reports(out, "validation_report")
    assert validation["errors"] == []
    (check,) = validation["noncanonical_residue_checks"]
    assert (check["count"], check["charges"], check["peptide_bonded"]) == (3, [-2.0] * 3, True)
    fit = _reports(out, "residue_report")["noncanonical_residues"][0]["qmmesp"]
    assert fit["fitted_from"] == "A:CGU17"
    import csv

    with (out / "residues" / "CGU" / "qm" / "qmmesp_pyscf" / "mm_point_charges.csv").open() as handle:
        calcium = [row for row in csv.DictReader(handle) if row["residue_name"] == "CA"]
    assert [float(row["charge"]) for row in calcium] == [2.0, 2.0, 2.0]
    # The fit left the buried methine to trade charge with its hydrogen, and
    # the report says so rather than passing it on quietly.
    residue = _reports(out, "residue_report")["noncanonical_residues"][0]
    assert any("depart from its AM1-BCC ones" in warning for warning in residue["warnings"])
