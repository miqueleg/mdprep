"""Heterogens that are not ligands, asked about as groups.

A deposition's heterogens are not all molecules to parameterise. The
references are real ones, from the enzymes a halogenase workshop meets:

* an unspecific peroxygenase (5OXU), with five N-glycans and the phosphate,
  glycerol and methanol it was crystallised from;
* the fluorinase (1RQR), solved with selenomethionine;
* chloroperoxidase (1CPO), which begins with a pyroglutamate;
* an L-2-haloacid dehalogenase (1ZRN), with its acetate bonded to the
  catalytic aspartate;
* a vanadium chloroperoxidase (1IDQ), whose vanadate is not a metal site.
"""

from pathlib import Path

import pytest
import yaml

from mdprep.config.decisions import (
    DecisionError,
    answer_conflicts,
    build_manifest,
    plan_decisions,
    proposed_answers,
)
from mdprep.config.models import ManifestConfig
from mdprep.structure.heterogens import classify_heterogens
from mdprep.structure.normalize import StructureNormalizationError, normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from tests.test_config_validation import base_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
HETEROGENS = DATA / "heterogens"
UPO = ROOT / "examples" / "tutorials" / "upo_5oxu" / "5OXU.pdb"
SELENOMETHIONINE = HETEROGENS / "selenomethionine_1rqr.pdb"
COVALENT = HETEROGENS / "covalent_intermediate_1zrn.pdb"
PYROGLUTAMATE = HETEROGENS / "pyroglutamate_1cpo.pdb"
VANADATE = HETEROGENS / "vanadate_1idq.pdb"


def walk(plan, given: dict) -> dict:
    """The answers a student gives by pressing Next past everything else."""

    answers: dict = {}
    for decision in plan.decisions:
        if not decision.is_applicable(answers):
            continue
        proposals = proposed_answers(plan, answers)
        if decision.id in given:
            answers[decision.id] = given[decision.id]
        elif decision.id in proposals:
            answers[decision.id] = proposals[decision.id]
        elif decision.default is not None:
            answers[decision.id] = decision.default
    return answers


def _atom(serial, name, resname, resid, element, x, y=0.0, z=0.0, chain="A", record="HETATM"):
    return (
        f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resid:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00          {element:>2}\n"
    )


def _write(tmp_path, text, name="input.pdb") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Glycans and crystallisation additives: the unspecific peroxygenase
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def upo():
    return plan_decisions(UPO)


UPO_ANSWERS = {
    "structure.remove_unknown_heterogens": "false",
    "metal.a_fe_401_site.charge": "3",
    "metal.a_fe_401_site.confirm_donors": "true",
    "metal.a_fe_401_site.donor_state.a_cys_36": "CYM",
    "refinement.total_qm_multiplicity": "2",
}


def test_the_glycans_and_the_additives_are_found(upo):
    assert upo.findings["glycans"] == [f"A:NAG{n}" for n in (404, 405, 406, 407, 408)]
    assert sorted(upo.findings["additives"]) == sorted(
        ["A:PO4409"]
        + [f"A:GOL{n}" for n in (410, 411, 412)]
        + [f"A:MOH{n}" for n in range(413, 423)]
    )
    # The heme is the only ligand: none of the above is offered as one
    # without being asked to keep it.
    assert upo.findings["ligands"][0] == "A:HEM401"


def test_each_group_is_one_question_naming_every_member(upo):
    glycans = upo.by_id("structure.remove_glycans")
    additives = upo.by_id("structure.remove_additives")

    assert glycans.default == "true" and additives.default == "true"
    assert len(glycans.targets) == 5
    assert len(additives.targets) == 14
    assert any("bonded A:ASN11 ND2" in line or "A:ASN11 ND2" in line for line in glycans.evidence)
    assert "A:GOL410 (glycerol)" in additives.evidence


def test_an_additive_is_a_ligand_question_only_once_the_additives_are_kept(upo):
    removed = walk(upo, UPO_ANSWERS)
    kept = walk(upo, {**UPO_ANSWERS, "structure.remove_additives": "false"})

    assert "ligand.a_gol_410.include" not in removed
    assert kept["ligand.a_gol_410.include"] == "true"


def test_the_manifest_removes_each_one_with_its_reason(upo, tmp_path):
    manifest = build_manifest(upo, walk(upo, UPO_ANSWERS))

    removed = {
        (item["selector"]["resname"], item["selector"]["resid"]): item["reason"]
        for item in manifest["structure"]["remove_residues"]
    }
    assert removed[("NAG", 404)] == "glycan bonded to the protein, which mdprep does not build"
    assert removed[("MOH", 413)] == "crystallisation additive (methanol)"
    assert len(removed) == 19
    manifest["project"]["output_dir"] = str(tmp_path / "out")
    ManifestConfig.model_validate(manifest)


def test_keeping_the_glycans_stops_the_build_at_the_question(upo):
    answers = walk(upo, {**UPO_ANSWERS, "structure.remove_glycans": "false"})

    with pytest.raises(DecisionError, match="does not build glycans") as caught:
        build_manifest(upo, answers)
    assert caught.value.decision_id == "structure.remove_glycans"
    assert answer_conflicts(upo, {"structure.remove_glycans": "false"}) is not None


def test_a_glycan_is_found_by_its_bond_length_when_the_file_declares_none(tmp_path):
    """A file another program rewrote has lost its LINK records."""

    text = "".join(line + "\n" for line in UPO.read_text().splitlines() if not line.startswith("LINK"))
    structure = read_pdb(_write(tmp_path, text))
    assert structure.links == ()

    groups = classify_heterogens(structure.residues)

    assert sorted(item.residue.id.display() for item in groups.glycans) == [
        f"A:NAG{n}" for n in (404, 405, 406, 407, 408)
    ]


def test_the_magnesium_is_proposed_nonbonded(upo):
    model = upo.by_id("metal.a_mg_402_site.model")

    assert model.default == "nonbonded"
    assert "closed-shell ion" in model.why


def test_a_heme_on_the_iron_and_the_magnesium_takes_its_charges_from_the_iron_fit(upo):
    """5OXU's heme holds its iron and, through a propionate, the magnesium.
    Its charges come from the iron's MCPB fit; the nonbonded magnesium has
    none, and is not the site it was looking for."""

    manifest = build_manifest(upo, walk(upo, UPO_ANSWERS))

    (heme,) = manifest["ligands"]
    assert heme["charge_method"] == "mcpb_resp_pyscf"
    assert [site["model"] for site in manifest["metals"]] == ["bonded_mcpb", "nonbonded"]


def test_an_additive_on_a_metal_is_not_offered_for_removal(tmp_path):
    """A heterogen that coordinates a metal is part of its site."""

    text = (DATA / "protein_histidine_zinc.pdb").read_text().replace("END\n", "")
    text += (
        _atom(30, "O", "ACT", 600, "O", 5.6, 5.4, 3.1)
        + _atom(31, "C", "ACT", 600, "C", 5.6, 5.4, 4.35)
        + _atom(32, "OXT", "ACT", 600, "O", 5.6, 6.5, 5.0)
        + _atom(33, "CH3", "ACT", 600, "C", 5.6, 4.2, 5.2)
        + _atom(34, "O1", "GOL", 700, "O", 30.0)
        + _atom(35, "C1", "GOL", 700, "C", 31.4)
        + "END\n"
    )

    plan = plan_decisions(_write(tmp_path, text))

    assert plan.findings["additives"] == ["A:GOL700"]
    assert "A:ACT600" in plan.findings["ligands"]


def test_with_only_additives_nothing_is_left_to_decline_once_they_go(tmp_path):
    text = (DATA / "protein_histidine_zinc.pdb").read_text().split("HETATM")[0]
    text += _atom(34, "O1", "GOL", 700, "O", 30.0) + _atom(35, "C1", "GOL", 700, "C", 31.4) + "END\n"
    plan = plan_decisions(_write(tmp_path, text))
    removing = {"structure.repair.add_missing_heavy_atoms": "false"}

    assert "structure.remove_unknown_heterogens" not in walk(plan, removing)
    kept = plan.applicable(walk(plan, {**removing, "structure.remove_additives": "false"}))
    assert "structure.remove_unknown_heterogens" in {decision.id for decision in kept}


# --------------------------------------------------------------------------
# Modified residues: the fluorinase's selenomethionine
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def selenomethionine():
    return plan_decisions(SELENOMETHIONINE)


def test_a_selenomethionine_is_a_residue_of_its_chain_not_a_ligand(selenomethionine):
    assert selenomethionine.findings["ligands"] == []
    assert selenomethionine.findings["modified_residues"] == ["A:MSE14 -> MET"]
    # Nor a gap in the chain where it sits.
    assert selenomethionine.findings["chain_breaks"] == []


def test_replacing_a_selenomethionine_is_proposed(selenomethionine):
    decision = selenomethionine.by_id("residue.MSE.treatment")

    assert decision.default == "replace"
    assert not decision.requires_user_input
    assert decision.evidence == ("A:MSE14 -> MET, from the file's MODRES records",)
    # Keeping it is offered too: AmberTools has a template for it.
    assert [choice.value for choice in decision.choices] == ["replace", "amber_library", "am1bcc"]


def test_the_replacement_reaches_the_repair_block(selenomethionine, tmp_path):
    manifest = build_manifest(selenomethionine, walk(selenomethionine, {}))

    assert manifest["structure"]["repair"] == {
        "backend": "pdbfixer",
        "add_missing_heavy_atoms": False,
        "unresolved_residues": "leave_absent",
        "replace_modified_residues": [
            {
                "selector": {"chain": "A", "resname": "MSE", "resid": 14, "icode": None},
                "parent": "MET",
            }
        ],
    }
    manifest["project"]["output_dir"] = str(tmp_path / "out")
    ManifestConfig.model_validate(manifest)


def test_keeping_a_selenomethionine_builds_it_from_its_template(selenomethionine, tmp_path):
    answers = walk(selenomethionine, {"residue.MSE.treatment": "amber_library"})

    manifest = build_manifest(selenomethionine, answers)

    assert manifest["noncanonical_residues"] == [
        {
            "resname": "MSE",
            "residues": [{"chain": "A", "resname": "MSE", "resid": 14, "icode": None}],
            "parameters": "amber_library",
            "template": "MSE",
        }
    ]
    assert "repair" not in manifest["structure"]
    manifest["project"]["output_dir"] = str(tmp_path / "out")
    ManifestConfig.model_validate(manifest)


def test_without_modres_the_peptide_bonds_say_it_is_in_the_chain(tmp_path):
    text = "".join(
        line + "\n"
        for line in SELENOMETHIONINE.read_text().splitlines()
        if not line.startswith("MODRES")
    )
    structure = read_pdb(_write(tmp_path, text))

    (item,) = classify_heterogens(structure.residues).modified_residues

    assert (item.residue.id.display(), item.parent, item.parent_source) == (
        "A:MSE14", "MET", "mdprep's table",
    )


def test_anything_else_modified_is_asked_without_a_proposal(tmp_path):
    """A phosphoserine's phosphate may be the chemistry being studied."""

    text = SELENOMETHIONINE.read_text().replace("MSE A  14", "SEP A  14").replace(
        "MSE A", "SEP A"
    ).replace(" MET  SELENOMETHIONINE", " SER  PHOSPHOSERINE    ")

    plan = plan_decisions(_write(tmp_path, text))
    decision = plan.by_id("residue.SEP.treatment")

    assert decision.default is None and decision.requires_user_input


def test_a_pyroglutamate_is_said_to_lose_its_lactam():
    """1CPO's N-terminal PCA0: a glutamine whose side chain closed onto its own
    nitrogen. Replacing it rebuilds the side chain and frees the amine, which
    is the student's to weigh, so it is said, and nothing is proposed."""

    decision = plan_decisions(PYROGLUTAMATE).by_id("residue.PCA.treatment")

    assert decision.evidence == ("A:PCA0 -> GLN, from the file's MODRES records",)
    assert decision.default is None and decision.requires_user_input
    assert "A:PCA0: its side chain is bonded to its own backbone" in decision.why
    assert "charged amine" in decision.why


def test_a_selenomethionine_keeps_its_side_chain(selenomethionine):
    decision = selenomethionine.by_id("residue.MSE.treatment")

    assert "bonded to its own backbone" not in decision.why


def test_only_proline_may_have_a_side_chain_on_its_backbone():
    """A hydroxyproline's CD-N bond is its parent's too."""
    from mdprep.structure.heterogens import side_chain_closes_onto_backbone

    lactam = [("N", (0.0, 0.0, 0.0)), ("CA", (1.46, 0.0, 0.0)), ("CD", (-0.5, 1.4, 0.0))]

    assert side_chain_closes_onto_backbone(lactam, "GLN")
    assert not side_chain_closes_onto_backbone(lactam, "PRO")


def test_a_modified_residue_with_no_known_parent_can_only_be_parameterised(tmp_path):
    """Nothing to replace it by, and no template: it is kept from its SMILES or not at all."""

    text = "".join(
        line.replace("MSE", "XYZ") + "\n"
        for line in SELENOMETHIONINE.read_text().splitlines()
        if not line.startswith("MODRES")
    )

    decision = plan_decisions(_write(tmp_path, text)).by_id("residue.XYZ.treatment")

    assert [choice.value for choice in decision.choices] == ["am1bcc"]
    assert decision.default is None and decision.requires_user_input


def test_a_modified_residue_that_ends_its_chain_with_no_parent_is_refused(tmp_path):
    """1RQR's first residue, renamed: no parent to replace it by, and a residue
    mdprep keeps must be bonded into its chain on both sides."""

    text = "".join(
        (line.replace("ATOM  ", "HETATM").replace("ILE A  11", "XYZ A  11") if " ILE A  11" in line else line)
        + "\n"
        for line in SELENOMETHIONINE.read_text().splitlines()
    )

    with pytest.raises(DecisionError, match="A:XYZ11 is a residue of a protein chain.*N-terminus"):
        plan_decisions(_write(tmp_path, text))


def test_a_free_amino_acid_is_still_a_ligand(tmp_path):
    """A substrate has a backbone too, and no peptide bond."""

    text = SELENOMETHIONINE.read_text().replace("END\n", "")
    text += (
        _atom(900, "N", "MSE", 900, "N", 60.0)
        + _atom(901, "CA", "MSE", 900, "C", 61.45)
        + _atom(902, "C", "MSE", 900, "C", 62.9)
        + "END\n"
    )

    plan = plan_decisions(_write(tmp_path, text))

    assert plan.findings["modified_residues"] == ["A:MSE14 -> MET"]
    assert "A:MSE900" in plan.findings["ligands"]


# --------------------------------------------------------------------------
# Heterogens bonded to the protein: the dehalogenase's intermediate
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def covalent():
    return plan_decisions(COVALENT)


def test_a_covalent_intermediate_is_asked_about_and_nothing_is_proposed(covalent):
    decision = covalent.by_id("structure.remove_covalent_heterogens")

    assert decision.default is None and decision.requires_user_input
    assert covalent.findings["ligands"] == []
    assert decision.evidence == (
        "A:ACY300: A:ASP10 OD2 -- A:ACY300 CH3 (declared in the file)",
    )


def test_removing_it_is_recorded(covalent):
    manifest = build_manifest(
        covalent, walk(covalent, {"structure.remove_covalent_heterogens": "true"})
    )

    assert manifest["structure"]["remove_residues"] == [
        {
            "selector": {"chain": "A", "resname": "ACY", "resid": 300, "icode": None},
            "reason": "bonded to the protein, which mdprep does not build",
        }
    ]


def test_keeping_it_stops_the_build(covalent):
    answers = walk(covalent, {"structure.remove_covalent_heterogens": "false"})

    with pytest.raises(DecisionError, match="heterogens bonded to the protein") as caught:
        build_manifest(covalent, answers)
    assert caught.value.decision_id == "structure.remove_covalent_heterogens"


def test_a_crosslink_between_two_protein_residues_is_refused(tmp_path):
    link = (
        "LINK         CD1 ILE A  11                 OG  SER A  15     1555   1555  1.43  \n"
    )

    with pytest.raises(DecisionError, match="neither a peptide bond nor a disulfide"):
        plan_decisions(_write(tmp_path, link + SELENOMETHIONINE.read_text()))


def test_a_bonded_cofactor_that_holds_a_metal_is_refused(tmp_path):
    """Myeloperoxidase's heme is held by three bonds to the protein."""

    text = (
        "LINK         OD1 ASP A  94                 CMD HEM A 601     1555   1555  1.45  \n"
        + _atom(1, "OD1", "ASP", 94, "O", 6.35, record="ATOM")
        + _atom(2, "FE", "HEM", 601, "FE", 0.0)
        + _atom(3, "NA", "HEM", 601, "N", 2.0)
        + _atom(4, "C1A", "HEM", 601, "C", 3.4)
        + _atom(5, "CMD", "HEM", 601, "C", 4.9)
        + "END\n"
    )

    with pytest.raises(DecisionError, match="covalently attached cofactor"):
        plan_decisions(_write(tmp_path, text))


# --------------------------------------------------------------------------
# A vanadate is not a metal in a cofactor
# --------------------------------------------------------------------------


def test_a_vanadate_is_left_whole_and_refused():
    structure = read_pdb(VANADATE)

    assert any(residue.id.resname == "VO4" for residue in structure.residues)
    assert any("an oxoanion such as vanadate" in warning for warning in structure.warnings)
    with pytest.raises(DecisionError, match="oxoanion such as vanadate"):
        plan_decisions(VANADATE)


# --------------------------------------------------------------------------
# A metal that is not part of the system
# --------------------------------------------------------------------------


def _metal_fixture(tmp_path, element: str) -> Path:
    text = (DATA / "protein_histidine_zinc.pdb").read_text()
    text = text.replace("ZN    ZN Z 500", f"{element:>2}    {element:>2} Z 500").replace(
        "          ZN\n", f"          {element:>2}\n"
    )
    return _write(tmp_path, text)


def test_a_free_metal_can_be_removed(tmp_path):
    plan = plan_decisions(_metal_fixture(tmp_path, "NI"))
    model = plan.by_id("metal.z_ni_500_site.model")
    assert "remove" in {choice.value for choice in model.choices}

    answers = walk(
        plan,
        {
            "metal.z_ni_500_site.model": "remove",
            "structure.repair.add_missing_heavy_atoms": "false",
        },
    )
    manifest = build_manifest(plan, answers)

    # Nothing about it is asked once it is going.
    assert "metal.z_ni_500_site.charge" not in answers
    assert manifest["metals"] == []
    assert manifest["structure"]["remove_residues"] == [
        {
            "selector": {"chain": "Z", "resname": "NI", "resid": 500, "icode": None},
            "reason": "metal left out as answered: not part of the system",
        }
    ]


def test_a_metal_read_out_of_its_cofactor_cannot_be_removed():
    plan = plan_decisions(ROOT / "examples" / "tutorials" / "cyp119_1io7" / "1IO7.pdb")

    model = plan.by_id("metal.a_fe_1401_site.model")

    assert "remove" not in {choice.value for choice in model.choices}


@pytest.mark.parametrize("element", ["MG", "CA"])
def test_magnesium_and_calcium_are_proposed_nonbonded(tmp_path, element):
    plan = plan_decisions(_metal_fixture(tmp_path, element))

    model = plan.by_id(f"metal.z_{element.lower()}_500_site.model")

    assert model.default == "nonbonded"


def test_other_metals_with_a_shell_are_still_proposed_bonded(tmp_path):
    plan = plan_decisions(_metal_fixture(tmp_path, "ZN"))

    assert plan.by_id("metal.z_zn_500_site.model").default == "bonded_mcpb"


# --------------------------------------------------------------------------
# Only what is in the chains being built
# --------------------------------------------------------------------------


def test_removals_follow_the_chains_kept(tmp_path):
    first = (DATA / "protein_histidine_zinc.pdb").read_text().split("HETATM")[0]
    second = "".join(
        line[:21] + "B" + line[22:] + "\n"
        for line in first.splitlines()
        if line.startswith("ATOM")
    )
    second = "".join(
        line[:30] + f"{float(line[30:38]) + 40.0:8.3f}" + line[38:] + "\n"
        for line in second.splitlines()
    )
    text = (
        first
        + "TER\n"
        + second
        + "TER\n"
        + _atom(100, "O1", "GOL", 700, "O", 0.0, 8.0)
        + _atom(101, "O1", "GOL", 701, "O", 40.0, 8.0, chain="B")
        + "END\n"
    )
    plan = plan_decisions(_write(tmp_path, text))

    manifest = build_manifest(
        plan,
        walk(
            plan,
            {
                "structure.chains": "A",
                "structure.remove_unknown_heterogens": "false",
                "structure.repair.add_missing_heavy_atoms": "false",
            },
        ),
    )

    assert [item["selector"]["chain"] for item in manifest["structure"]["remove_residues"]] == ["A"]


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def _manifest_with(tmp_path, **structure) -> dict:
    data = base_manifest()
    data["project"]["input_structure"] = str(COVALENT)
    data["project"]["output_dir"] = str(tmp_path / "out")
    data["structure"].update(structure)
    return data


ACETATE = {"chain": "A", "resname": "ACY", "resid": 300, "icode": None}


def test_a_residue_cannot_be_removed_twice(tmp_path):
    data = _manifest_with(
        tmp_path,
        remove_residues=[
            {"selector": ACETATE, "reason": "one"},
            {"selector": ACETATE, "reason": "two"},
        ],
    )

    with pytest.raises(ValueError, match="more than once"):
        ManifestConfig.model_validate(data)


def test_a_residue_cannot_be_removed_and_built(tmp_path):
    data = _manifest_with(tmp_path, remove_residues=[{"selector": ACETATE, "reason": "gone"}])
    data["ligands"] = [
        {
            "id": "acetate",
            "selector": ACETATE,
            "net_charge": -1,
            "multiplicity": 1,
            "atom_types": "gaff2",
            "charge_method": "am1bcc",
        }
    ]

    with pytest.raises(ValueError, match="leaves out residues the manifest also builds"):
        ManifestConfig.model_validate(data)


def test_a_replacement_must_be_a_standard_amino_acid(tmp_path):
    data = _manifest_with(
        tmp_path,
        repair={
            "backend": "pdbfixer",
            "replace_modified_residues": [
                {"selector": {"chain": "A", "resname": "MSE", "resid": 14}, "parent": "XYZ"}
            ],
        },
    )

    with pytest.raises(ValueError, match="twenty standard amino acids"):
        ManifestConfig.model_validate(data)


def test_a_replacement_alone_is_reason_enough_to_run_pdbfixer(tmp_path):
    data = _manifest_with(
        tmp_path,
        repair={
            "backend": "pdbfixer",
            "replace_modified_residues": [
                {"selector": {"chain": "A", "resname": "MSE", "resid": 14}, "parent": "met"}
            ],
        },
    )

    manifest = ManifestConfig.model_validate(data)

    assert manifest.structure.repair.replace_modified_residues[0].parent == "MET"


# --------------------------------------------------------------------------
# Normalisation leaves them out, and says so
# --------------------------------------------------------------------------


def test_normalisation_removes_what_the_manifest_names_and_reports_why(tmp_path):
    manifest = ManifestConfig.model_validate(
        _manifest_with(tmp_path, remove_residues=[{"selector": ACETATE, "reason": "bonded to Asp10"}])
    )

    result = normalize_structure_stage(manifest, output_path=tmp_path / "n.pdb")

    names = {residue.id.resname for residue in result.normalized_structure.residues}
    assert "ACY" not in names
    report = result.to_report_dict()
    assert report["residues_removed"][0]["reason"] == "bonded to Asp10"
    assert report["unknown_heterogens_causing_failure"] == []


def test_naming_a_residue_the_structure_does_not_have_is_an_error(tmp_path):
    manifest = ManifestConfig.model_validate(
        _manifest_with(
            tmp_path,
            remove_residues=[
                {"selector": {**ACETATE, "resid": 999}, "reason": "not there"}
            ],
        )
    )

    with pytest.raises(StructureNormalizationError, match="does not have"):
        normalize_structure_stage(manifest, output_path=tmp_path / "n.pdb")


def test_the_example_manifest_is_the_one_the_planner_writes(upo):
    """examples/20 is the builder's output for 5OXU with these answers."""

    example = yaml.safe_load(
        (ROOT / "examples" / "20_upo_5oxu_heme_glycans.yaml").read_text(encoding="utf-8")
    )
    built = build_manifest(upo, walk(upo, UPO_ANSWERS))

    assert example["structure"]["remove_residues"] == built["structure"]["remove_residues"]
    assert example["metals"] == built["metals"]
    assert example["ligands"] == built["ligands"]
