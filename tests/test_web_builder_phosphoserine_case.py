"""The web builder on a phosphoserine in its chain.

SpoIIAA's SEP57 (PDB 1H4X, chain B residues 52-62). A student is asked what
to do with it -- replace it by serine, or keep it -- and nothing is proposed.
Kept from its SMILES, the SMILES and charge are offered rather than typed;
kept with AmberTools' template, the dianion's template is; kept with QMMESP
charges, the QM level is written for the student and the setup cell installs
PySCF. Every way, the manifest the page writes is one mdprep accepts.

Slow, like the other reference cases: a real deposition through Pyodide.
"""

import copy

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from mdprep.residues.templates import KNOWN_RESIDUE_CHEMISTRY
from tests.test_noncanonical_residues import PHOSPHOSERINE
from tests.test_web_builder_flavin_case import _drive, _steps

pytestmark = [pytest.mark.external, pytest.mark.slow]


@pytest.fixture(scope="module")
def parameterised(tmp_path_factory):
    return _drive(
        tmp_path_factory,
        PHOSPHOSERINE,
        {
            "structure.repair.add_missing_heavy_atoms": "false",
            "residue.SEP.treatment": "am1bcc",
        },
    )


@pytest.fixture(scope="module")
def templated(tmp_path_factory):
    return _drive(
        tmp_path_factory,
        PHOSPHOSERINE,
        {
            "structure.repair.add_missing_heavy_atoms": "false",
            "residue.SEP.treatment": "amber_library",
        },
    )


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    return _drive(
        tmp_path_factory,
        PHOSPHOSERINE,
        {
            "structure.repair.add_missing_heavy_atoms": "false",
            "residue.SEP.treatment": "qmmesp_pyscf",
        },
    )


def _accepted(report, tmp_path) -> dict:
    assert report["ok"], report.get("failure")
    manifest = yaml.safe_load(report["manifest"])
    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(PHOSPHOSERINE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")
    ManifestConfig.model_validate(candidate)
    return manifest


def test_keeping_or_replacing_is_the_students_to_say(parameterised):
    assert parameterised["errors"] == []
    assert parameterised["unanswered"] == []
    steps = _steps(parameterised)

    # A choice nothing proposes: no radio is selected until the student picks.
    assert steps["residue.SEP.treatment"]["selected"] is None
    assert steps["residue.SEP.treatment"]["answered"] == "am1bcc"
    assert not [key for key in steps if key.startswith("ligand.")]


def test_the_smiles_and_charge_are_offered_rather_than_typed(parameterised, tmp_path):
    steps = _steps(parameterised)
    chemistry = KNOWN_RESIDUE_CHEMISTRY["SEP"]

    assert steps["residue.SEP.smiles"]["prefilled"] == chemistry.smiles
    assert "second pKa" in steps["residue.SEP.smiles"]["note"]
    assert steps["residue.SEP.net_charge"]["prefilled"] == "-2"
    (block,) = _accepted(parameterised, tmp_path)["noncanonical_residues"]
    assert (block["parameters"], block["smiles"], block["net_charge"], block["parent"]) == (
        "am1bcc", chemistry.smiles, -2, "SER",
    )


def test_the_dianions_template_is_offered(templated, tmp_path):
    assert templated["errors"] == []
    assert templated["unanswered"] == []
    steps = _steps(templated)

    # A proposal for a choice is the radio the page selects.
    assert steps["residue.SEP.template"]["selected"] == "SEP"
    assert steps["residue.SEP.template"]["answered"] is None
    assert "choose S1P then" in steps["residue.SEP.template"]["note"]
    (block,) = _accepted(templated, tmp_path)["noncanonical_residues"]
    assert (block["parameters"], block["template"]) == ("amber_library", "SEP")


def test_qmmesp_charges_are_one_choice_away(fitted, tmp_path):
    assert fitted["errors"] == []
    assert fitted["unanswered"] == []
    steps = _steps(fitted)

    # The SMILES and charge are the same questions, proposed the same way.
    assert steps["residue.SEP.smiles"]["prefilled"] == KNOWN_RESIDUE_CHEMISTRY["SEP"].smiles
    (block,) = _accepted(fitted, tmp_path)["noncanonical_residues"]
    assert (block["parameters"], block["qmmesp"]) == (
        "qmmesp_pyscf", {"method": "HF", "basis": "6-31G*"},
    )
    # What the fit runs is what the Colab cell installs.
    assert "pyscf" in fitted["setup"] and "rdkit" in fitted["setup"]
