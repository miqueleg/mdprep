"""A kept noncanonical residue on a bonded metal, through the real toolchain.

SyrB2's iron (PDB 2FCT, examples/17) with its His235 deposited as a modified
residue, HSK, and kept: parameterised from the SMILES of the tautomer that
binds the iron by NE2. The whole preparation runs -- protonation, the QM/MM
refinement, MCPB.py with a g-xTB Hessian and a GFN2 ESP, tleap -- and the
kept residue has to come out of it the way a histidine on the metal does:
renamed by MCPB.py to a template of its own, bonded to the iron and into the
chain, while the rest of the build keeps its library.

Skips cleanly without the tools; under a minute with them.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from mdprep.external.discovery import which_executable

pytestmark = [pytest.mark.external, pytest.mark.ambertools, pytest.mark.tleap]

DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
KEPT_HISTIDINE = "CC(=O)NC(Cc1cnc[nH]1)C(=O)NC"


def test_a_templates_unit_is_written_out_for_mcpb(tmp_path):
    if which_executable("tleap") is None:
        pytest.skip("tleap is required")
    from mdprep.ambertools.mol2 import read_mol2
    from mdprep.residues.templates import AMBER_RESIDUE_TEMPLATES
    from mdprep.residues.workflow import template_mol2

    path = template_mol2(
        AMBER_RESIDUE_TEMPLATES["SEP"], protein_forcefield="ff14SB", work_dir=tmp_path
    )

    atoms = read_mol2(path).atoms
    # The unit tleap builds the residue from, hydrogens and phosaa's types included.
    assert {atom.subst_name for atom in atoms} == {"SEP"}
    assert {atom.name: atom.atom_type for atom in atoms}["O1P"] == "OX"
    assert {"H", "HA", "HB2", "HB3"} <= {atom.name for atom in atoms}
    assert sum(atom.charge for atom in atoms) == pytest.approx(-2.0, abs=1e-3)


def _kept_histidine_deposition(tmp_path: Path) -> Path:
    lines = []
    for line in DEPOSITION.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM", "HETATM")) and line[17:26] == "HIS A 235":
            line = "HETATM" + line[6:17] + "HSK" + line[20:]
        lines.append(line)
    lines.insert(1, "MODRES 2FCT HSK A  235  HIS  MODIFIED RESIDUE")
    path = tmp_path / "2FCT_hsk.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def prepared(tmp_path_factory) -> Path:
    missing = [
        name
        for name in ("tleap", "MCPB.py", "antechamber", "parmchk2", "propka3", "xtb")
        if which_executable(name) is None
    ]
    if missing:
        pytest.skip(f"{', '.join(missing)} required for the bonded preparation")
    for module in ("pyscf", "openmm", "pdbfixer", "rdkit"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is required for the bonded preparation")
    gxtb = os.environ.get("MDPREP_GXTB_EXECUTABLE")
    if not gxtb or which_executable(gxtb) is None:
        pytest.skip("set MDPREP_GXTB_EXECUTABLE to an official g-xTB-enabled xtb binary")

    from mdprep.config.decisions import build_manifest, plan_decisions, proposed_answers
    from mdprep.config.models import ManifestConfig
    from mdprep.workflows.prepare import prepare_system

    work = tmp_path_factory.mktemp("kept_histidine")
    plan = plan_decisions(_kept_histidine_deposition(work))
    answers = {
        "structure.chains": "A",
        "structure.remove_unknown_heterogens": True,
        "structure.repair.add_missing_heavy_atoms": True,
        "ligand.a_dsu_896.include": False,
        "ligand.a_akg_897.net_charge": -2,
        "ligand.a_akg_897.smiles": "[O-]C(=O)CCC(=O)C(=O)[O-]",
        "residue.HSK.treatment": "am1bcc",
        "residue.HSK.net_charge": 0,
        "residue.HSK.smiles": KEPT_HISTIDINE,
        "metal.a_fe2_900_site.confirm_donors": True,
        "refinement.total_qm_multiplicity": 5,
        "project.name": "syrb2_kept_histidine",
        "project.output_dir": str(work / "prepared"),
        "solvation.enabled": False,
    }
    # Fe(II) +2, AKG -2, Cl -1, His116 and the kept His235 0, the water 0.
    assert proposed_answers(plan, answers)["metal.a_fe2_900_site.model_charge"] == "-1"
    answers["metal.a_fe2_900_site.model_charge"] = -1
    manifest = build_manifest(plan, answers)
    ManifestConfig.model_validate(manifest)
    path = work / "system.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    prepare_system(path, stop_after="tleap")
    return work / "prepared"


def _report(out: Path, name: str) -> dict:
    return json.loads((out / "reports" / f"{name}.json").read_text(encoding="utf-8"))


@pytest.mark.slow
def test_mcpb_models_the_kept_residue_as_an_amino_acid(prepared):
    site = _report(prepared, "metal_report")["bonded_mcpb_site"]
    work = Path(site["work_dir"])

    # It reached MCPB.py as a mol2, as a ligand does...
    assert "HSK.mol2" in (work / "mcpb.in").read_text(encoding="utf-8")
    # ...and MCPB.py capped it from its neighbours and renamed it, as it does
    # any amino acid on the metal.
    renames = {rename["original_resname"]: rename["final_resname"] for rename in site["residue_renames"]}
    assert renames["HSK"] == "HK1"
    large = (work / f"{site['site_id']}_large.pdb").read_text(encoding="utf-8")
    assert " ACE " in large and " NME " in large and " HSK " in large
    # The final build keeps the kept residue's own library everywhere else.
    assert not any(command.startswith("HSK = loadmol2") for command in site["leap_setup_commands"])
    assert any(command.startswith("HK1 = loadmol2") for command in site["leap_setup_commands"])


@pytest.mark.slow
def test_the_kept_residue_is_bonded_to_the_iron_and_into_the_chain(prepared):
    validation = _report(prepared, "validation_report")

    assert validation["errors"] == []
    (check,) = validation["noncanonical_residue_checks"]
    assert (check["count"], check["peptide_bonded"], check["ok"]) == (1, True, True)
    # Refitted with the site: reported, not held to an integer.
    (refitted,) = check["refitted_by_mcpb"].values()
    assert abs(refitted) < 1.0
    bonds = _report(prepared, "metal_report")["leap_bond_commands"]
    assert any(command.endswith(".FE") and ".NE2 " in command for command in bonds)
