"""The web builder, driven over the reference structure a student will use.

test_web_builder_browser.py proves the page works, on a seven-atom fixture.
This asks the question that matters for the tutorial: does clicking through
the page on a real deposition -- four chains, a bonded iron, a halide in the
coordination sphere, a cofactor, histidines needing a tautomer, and five
residues the structure did not fully resolve -- produce a manifest mdprep
will accept, with the defaults the project intends?

It is slower still than the smoke test, because 6NIE is a real structure.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mdprep.config.models import ManifestConfig
from tests.test_web_builder_browser import (
    REPO,
    WEB,
    find_chrome,
    has_puppeteer,
    serve,
)

pytestmark = [pytest.mark.external, pytest.mark.slow]

DRIVER = REPO / "tests" / "browser" / "student_flow.mjs"
STRUCTURE = REPO / "examples" / "tutorials" / "besd_6nie" / "6NIE.pdb"

# Only the questions the engine refuses to default. Everything else is left
# alone, so what comes out is what a student gets by clicking Next.
ANSWERS = {
    "structure.chains": "A",
    "structure.repair.add_missing_heavy_atoms": "true",
    "ligand.a_lys_301.net_charge": "0",
    "ligand.a_lys_301.smiles": "C(CCN)CC(C(=O)O)N",
    "ligand.a_akg_304.net_charge": "-2",
    "metal.a_fe2_302_site.charge": "2",
    "metal.a_fe2_302_site.confirm_donors": "true",
    "metal.a_fe2_302_site.multiplicity": "5",
    "metal.a_fe2_302_site.model_charge": "-1",
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    chrome = find_chrome()
    if chrome is None:
        pytest.skip("a Chromium/Chrome binary is required for the browser test")
    if not has_puppeteer():
        pytest.skip("puppeteer-core is required: npm install puppeteer-core")
    if not STRUCTURE.is_file():
        pytest.skip(f"the vendored reference structure is missing: {STRUCTURE}")

    subprocess.run(
        [sys.executable, str(WEB / "build.py")],
        check=True,
        capture_output=True,
        cwd=REPO,
    )
    answers = tmp_path_factory.mktemp("answers") / "answers.json"
    answers.write_text(json.dumps(ANSWERS), encoding="utf-8")

    with serve(WEB) as url:
        result = subprocess.run(
            ["node", str(DRIVER), url, str(STRUCTURE), chrome, str(answers)],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=900,
        )
    if result.returncode != 0:
        pytest.fail(f"browser driver failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def manifest(report) -> dict:
    assert report["ok"], report.get("failure")
    return yaml.safe_load(report["manifest"])


def test_the_page_handles_a_real_deposition_without_errors(report):
    assert report["errors"] == []
    assert report["unanswered"] == []


def test_the_manifest_it_writes_is_one_mdprep_accepts(manifest, tmp_path):
    """Validated through the real schema, not just parsed as YAML."""

    # Copied, not mutated: the fixture is module scoped and other tests read
    # what the page actually produced.
    candidate = copy.deepcopy(manifest)
    candidate["project"]["input_structure"] = str(STRUCTURE)
    candidate["project"]["output_dir"] = str(tmp_path / "prepared")

    ManifestConfig.model_validate(candidate)


def test_clicking_through_gives_the_intended_defaults(manifest):
    assert manifest["protein"]["forcefield"] == "ff14SB"
    assert manifest["protein"]["water_model"] == "TIP3P"
    assert manifest["protonation"]["method"] == "propka_xtb_his"
    assert manifest["refinement"]["backend"] == "openmm"

    repair = manifest["structure"]["repair"]
    assert repair["backend"] == "pdbfixer"
    assert repair["unresolved_residues"] == "leave_absent"


def test_the_bonded_iron_is_set_up_for_gxtb_mcpb(manifest):
    site = next(item for item in manifest["metals"] if item["id"].endswith("fe2_302_site"))

    assert site["model"] == "bonded_mcpb"
    pyscf = site["mcpb"]["pyscf"]
    assert pyscf["hessian_backend"] == "xtb"
    assert pyscf["xtb"]["model"] == "gxtb"
    assert pyscf["esp_backend"] == "xtb"
    # Naming one would hand a g-xTB run the GFN-only conda binary.
    assert "executable" not in pyscf["xtb"]


def test_the_coordinating_halide_is_declared_as_an_ion(manifest):
    """It is neither a metal nor something antechamber can type, so a bonded
    site that lost it would be short a ligand and a charge."""

    site = next(item for item in manifest["metals"] if item["id"].endswith("fe2_302_site"))
    ions = site.get("coordinating_ions") or []

    assert [ion["element"] for ion in ions] == ["Cl"]
    assert ions[0]["charge"] == -1


def test_the_ligand_without_hydrogens_carries_its_smiles(manifest):
    """6NIE resolves the cofactor's CH2 groups but nothing on the free lysine,
    so only the lysine needs its protonation stated."""

    ligands = {item["id"]: item for item in manifest["ligands"]}

    assert ligands["a_lys_301"]["smiles"]
    assert "smiles" not in ligands["a_akg_304"]
    assert ligands["a_lys_301"]["charge_method"] == "am1bcc"
    assert ligands["a_lys_301"]["atom_types"] == "gaff2"


def test_a_proposed_answer_is_shown_filled_in_with_its_arithmetic(report):
    """A blocking question is shown with an empty box so its default cannot
    be accepted by reflex. A proposal is not a default -- it was computed
    from answers this user gave -- so it is filled in, and the sum behind it
    is on screen next to it.

    The capped model charge is the case: mdprep asks for the metal's
    oxidation state and each cofactor's net charge, then used to ask the user
    to add them up.
    """

    steps = {step["id"]: step for step in report["asked"] if step.get("id")}
    charge = next(
        step
        for key, step in steps.items()
        if key.endswith(".model_charge")
    )

    assert charge["prefilled"], "the proposal should be filled in"
    assert "Worked out from your earlier answers" in charge["note"]
    assert "=" in charge["note"], "the arithmetic should be shown, not just the total"
    assert "cannot be guessed" not in charge["note"]


def test_a_question_with_nothing_to_propose_is_still_shown_empty(report):
    """The iron's own oxidation state is not inferable from a deposition that
    calls the residue FE2 -- it is, and that one is proposed -- but the
    multiplicity is not, and must not be prefilled with anything."""

    steps = {step["id"]: step for step in report["asked"] if step.get("id")}
    multiplicity = next(
        step for key, step in steps.items() if key.endswith(".multiplicity")
    )

    assert multiplicity["prefilled"] == ""
    assert "cannot be guessed" in multiplicity["note"]


def test_the_page_never_asks_for_an_output_directory(report):
    """There is no filesystem behind the page to put anything in."""

    asked = {step["id"] for step in report["asked"] if step.get("id")}

    assert "project.output_dir" not in asked


def test_the_manifest_still_carries_an_output_directory(manifest):
    """Not asking is not the same as leaving it out: mdprep prepare needs
    somewhere to write, and the default is relative to wherever it runs."""

    output = manifest["project"]["output_dir"]

    assert output
    assert not output.startswith("/"), "a path from a browser must not be absolute"
