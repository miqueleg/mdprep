"""A metal deposited inside a cofactor: the iron of a heme.

CYP119 (PDB 1IO7) is the reference: a cytochrome P450 whose heme iron is
bonded to the porphyrin's four nitrogens, the proximal cysteine and a water.
The PDB writes heme b as one residue, HEM, with the iron among its 43 atoms.
mdprep reads the iron out into a residue of its own and models it as a metal
site; the porphyrin left behind is a ligand on it.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mdprep.config.decisions import (
    DecisionError,
    build_manifest,
    plan_decisions,
    proposal_explanations,
    proposed_answers,
)
from mdprep.config.models import ManifestConfig
from mdprep.ligands.cofactors import KNOWN_METAL_FREE_COFACTORS, known_metal_free_cofactor
from mdprep.structure.pdb import read_pdb
from mdprep.structure.pdb_fields import PdbFieldError
from mdprep.structure.writer import write_pdb

CYP119 = Path(__file__).resolve().parents[1] / "examples" / "tutorials" / "cyp119_1io7" / "1IO7.pdb"
HEME_SMILES = KNOWN_METAL_FREE_COFACTORS["HEM"].smiles

# Answers a student gives for chain A, accepting every proposal.
CHAIN_A = {
    "structure.chains": "A",
    "structure.remove_unknown_heterogens": "false",
    "structure.repair.add_missing_heavy_atoms": "true",
    "metal.a_fe_1401_site.charge": "3",
    "metal.a_fe_1401_site.confirm_donors": "true",
    "metal.a_fe_1401_site.donor_state.a_cys_317": "CYM",
    "metal.a_fe_1401_site.multiplicity": "2",
}


def _atom(serial, name, resname, resid, element, x, chain="A", record="HETATM"):
    return (
        f"{record:<6}{serial:>5} {name:<4} {resname:>3} {chain}{resid:>4}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 10.00          {element:>2}\n"
    )


def _pdb(tmp_path, lines) -> Path:
    path = tmp_path / "input.pdb"
    path.write_text("".join(lines) + "END\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def plan():
    return plan_decisions(CYP119)


def with_proposals(plan, answers: dict) -> dict:
    """The answers plus every proposal, as a student pressing Next gives."""

    answers = dict(answers)
    for _ in range(3):
        for key, value in proposed_answers(plan, answers).items():
            answers.setdefault(key, value)
    return answers


# --------------------------------------------------------------------------
# Reading: the iron comes out of its heme
# --------------------------------------------------------------------------


def test_a_heme_iron_is_read_out_into_a_residue_of_its_own():
    structure = read_pdb(CYP119)
    residues = {residue.id.display(): residue for residue in structure.residues}

    assert len(residues["A:HEM1401"].atoms) == 42
    assert "FE" not in residues["A:HEM1401"].atom_names()
    (iron,) = residues["A:FE1401"].atoms
    assert (iron.name, iron.element, iron.record_name) == ("FE", "Fe", "HETATM")
    assert "B:FE901" in residues
    # Right after its heme, which keeps the file order otherwise.
    order = [residue.id.display() for residue in structure.residues]
    assert order.index("A:FE1401") == order.index("A:HEM1401") + 1


def test_the_name_is_assigned_so_nothing_reads_an_oxidation_state_into_it():
    """FE is the PDB's code for Fe(III); a heme's iron is either."""

    structure = read_pdb(CYP119)

    assert "FE" in structure.assigned_resnames
    assert any("A:HEM1401 holds a metal atom, FE (Fe)" in warning for warning in structure.warnings)


def test_the_split_survives_writing_and_reading_back(tmp_path):
    written = tmp_path / "written.pdb"
    write_pdb(read_pdb(CYP119), written)

    again = read_pdb(written)

    assert [r.id.display() for r in again.residues if r.id.resname in {"HEM", "FE"}] == [
        "A:HEM1401", "A:FE1401", "B:HEM901", "B:FE901",
    ]
    assert "FE" in again.assigned_resnames


def test_every_other_atom_keeps_its_place(tmp_path):
    """Only the metal moves, to just after the cofactor's last atom -- even
    when a file interleaves the cofactor with something else."""

    path = _pdb(
        tmp_path,
        [
            _atom(1, "FE", "HEM", 401, "FE", 0.0),
            _atom(2, "NA", "HEM", 401, "N", 2.0),
            _atom(3, "O", "HOH", 501, "O", 9.0),
            _atom(4, "NB", "HEM", 401, "N", -2.0),
            _atom(5, "O", "HOH", 502, "O", -9.0),
        ],
    )

    atoms = read_pdb(path).atoms

    assert [(atom.resname, atom.name) for atom in atoms] == [
        ("HEM", "NA"), ("HOH", "O"), ("HEM", "NB"), ("FE", "FE"), ("HOH", "O"),
    ]


def test_a_lone_metal_and_a_cluster_are_left_as_they_are(tmp_path):
    path = _pdb(
        tmp_path,
        [
            _atom(1, "ZN", "ZN", 501, "ZN", 0.0),
            _atom(2, "FE1", "SF4", 502, "FE", 10.0),
            _atom(3, "FE2", "SF4", 502, "FE", 12.0),
            _atom(4, "S1", "SF4", 502, "S", 11.0),
        ],
    )

    structure = read_pdb(path)

    assert [(r.id.display(), len(r.atoms)) for r in structure.residues] == [
        ("A:ZN501", 1), ("A:SF4502", 3),
    ]
    assert structure.assigned_resnames == frozenset()


def test_a_metal_whose_new_residue_is_taken_is_refused(tmp_path):
    path = _pdb(
        tmp_path,
        [
            _atom(1, "FE", "HEM", 401, "FE", 0.0),
            _atom(2, "NA", "HEM", 401, "N", 2.0),
            _atom(3, "FE", "FE", 401, "FE", 20.0),
        ],
    )

    with pytest.raises(PdbFieldError, match="already has a residue by that name and number"):
        read_pdb(path)


# --------------------------------------------------------------------------
# Planning: an iron site with a porphyrin on it
# --------------------------------------------------------------------------


def test_the_iron_is_a_metal_site_and_the_porphyrin_a_ligand_on_it(plan):
    assert plan.findings["metal_ions"] == ["A:FE1401", "B:FE901"]
    assert plan.findings["ligands"] == ["A:HEM1401", "B:HEM901"]

    confirm = plan.by_id("metal.a_fe_1401_site.confirm_donors")
    donors = [(target.resname, target.resid, target.atom_name) for target in confirm.targets[1:]]
    assert sorted(donors) == sorted(
        [("HEM", 1401, name) for name in ("NA", "NB", "NC", "ND")]
        + [("CYS", 317, "SG"), ("HOH", 1402, "O")]
    )
    assert plan.by_id("metal.a_fe_1401_site.model").default == "bonded_mcpb"
    assert plan.by_id("ligand.a_hem_1401.charge_method").default == "mcpb_resp_pyscf"
    assert plan.by_id("metal.a_fe_1401_site.model").evidence[0].startswith(
        "A:FE1401 is the iron of A:HEM1401"
    )


def test_the_oxidation_state_is_asked_not_read_off_the_name(plan):
    charge = plan.by_id("metal.a_fe_1401_site.charge")

    assert charge.default is None
    assert charge.requires_user_input
    assert "came out of A:HEM1401" in charge.why
    assert "Fe(III) in a resting cytochrome P450" in charge.why
    assert "low-spin, 2" in plan.by_id("metal.a_fe_1401_site.multiplicity").why


def test_the_cysteine_state_is_asked_before_the_model_charge(plan):
    ids = [decision.id for decision in plan.decisions]

    assert (
        ids.index("metal.a_fe_1401_site.confirm_donors")
        < ids.index("metal.a_fe_1401_site.donor_state.a_cys_317")
        < ids.index("metal.a_fe_1401_site.multiplicity")
        < ids.index("metal.a_fe_1401_site.model_charge")
    )


def test_the_porphyrin_smiles_and_charge_are_proposed_with_their_reason(plan):
    proposed = proposed_answers(plan, {"structure.chains": "A"})
    why = proposal_explanations(plan, {"structure.chains": "A"})

    assert proposed["ligand.a_hem_1401.smiles"] == HEME_SMILES
    assert proposed["ligand.a_hem_1401.net_charge"] == "-4"
    assert "both propionates ionised" in why["ligand.a_hem_1401.smiles"]
    # A proposal is not a default: nothing reaches a manifest unaccepted.
    assert plan.by_id("ligand.a_hem_1401.smiles").default is None


@pytest.mark.parametrize(("state", "expected"), [("CYM", "-2"), ("CYS", "-1")])
def test_the_model_charge_counts_the_cysteine_as_it_was_stated(plan, state, expected):
    answers = with_proposals(
        plan, {**CHAIN_A, "metal.a_fe_1401_site.donor_state.a_cys_317": state}
    )

    assert answers["metal.a_fe_1401_site.model_charge"] == expected
    why = proposal_explanations(plan, answers)["metal.a_fe_1401_site.model_charge"]
    assert f"HEM1401 -4 {state}317" in why


def test_no_model_charge_is_proposed_before_the_cysteine_is_stated(plan):
    answers = with_proposals(plan, CHAIN_A)
    del answers["metal.a_fe_1401_site.donor_state.a_cys_317"]
    answers.pop("metal.a_fe_1401_site.model_charge", None)

    assert "metal.a_fe_1401_site.model_charge" not in proposed_answers(plan, answers)


def test_chain_a_builds_a_manifest_mdprep_accepts(plan, tmp_path):
    manifest = build_manifest(plan, with_proposals(plan, CHAIN_A))
    manifest["project"]["output_dir"] = str(tmp_path / "prepared")

    config = ManifestConfig.model_validate(manifest)

    (site,) = config.metals
    assert site.ions[0].selector.resname == "FE"
    assert (site.ions[0].charge, site.ions[0].multiplicity) == (3, 2)
    assert site.mcpb.small_model_charge == -2
    # Smeared, because a low-spin Fe(III) heme's SCF does not converge at
    # xTB's own 300 K.
    assert site.mcpb.pyscf.xtb.electronic_temperature_kelvin == 1000.0
    (ligand,) = config.ligands
    assert (ligand.selector.resname, ligand.net_charge, ligand.smiles) == ("HEM", -4, HEME_SMILES)
    assert [override.state for override in config.protonation.overrides] == ["CYM"]


def test_both_copies_share_one_fit(plan):
    answers = with_proposals(plan, {**CHAIN_A, "structure.chains": "all"})

    manifest = build_manifest(plan, answers)

    sites = {site["id"]: site for site in manifest["metals"]}
    assert sites["b_fe_901_site"]["parameters_from"] == "a_fe_1401_site"
    ligands = {ligand["id"]: ligand for ligand in manifest["ligands"]}
    assert ligands["b_hem_901"]["parameters_from"] == "a_hem_1401"
    ManifestConfig.model_validate(manifest)


def test_a_residue_named_hem_that_is_not_heme_gets_no_proposal(tmp_path):
    path = _pdb(
        tmp_path,
        [
            _atom(1, "FE", "HEM", 401, "FE", 0.0),
            _atom(2, "NA", "HEM", 401, "N", 2.0),
            _atom(3, "C1A", "HEM", 401, "C", 3.4),
        ],
    )

    (porphyrin,) = [r for r in read_pdb(path).residues if r.id.resname == "HEM"]

    assert known_metal_free_cofactor(porphyrin) is None


# --------------------------------------------------------------------------
# What cannot be prepared is refused before anything is asked
# --------------------------------------------------------------------------


def test_a_metal_cluster_is_refused_with_the_reason(tmp_path):
    path = _pdb(
        tmp_path,
        [
            _atom(1, "N", "ALA", 1, "N", -10.0, record="ATOM"),
            _atom(2, "CA", "ALA", 1, "C", -9.0, record="ATOM"),
            _atom(3, "FE1", "SF4", 502, "FE", 10.0),
            _atom(4, "FE2", "SF4", 502, "FE", 12.0),
            _atom(5, "S1", "SF4", 502, "S", 11.0),
        ],
    )

    with pytest.raises(DecisionError, match=r"A:SF4502 holds 2 metal atoms \(Fe\): a metal cluster"):
        plan_decisions(path)


def test_heme_c_is_refused_as_a_covalent_ligand(tmp_path):
    path = _pdb(
        tmp_path,
        [
            _atom(1, "N", "ALA", 1, "N", -10.0, record="ATOM"),
            _atom(2, "CA", "ALA", 1, "C", -9.0, record="ATOM"),
            _atom(3, "FE", "HEC", 201, "FE", 10.0),
            _atom(4, "NA", "HEC", 201, "N", 12.0),
            _atom(5, "C1A", "HEC", 201, "C", 13.4),
            _atom(6, "C2A", "HEC", 201, "C", 14.8),
        ],
    )

    with pytest.raises(DecisionError, match="A:HEC201 is heme c.*covalently attached"):
        plan_decisions(path)


# --------------------------------------------------------------------------
# The parts of the run a heme reached first
# --------------------------------------------------------------------------


def test_a_thiolate_is_passed_to_openmm_as_its_cysteine_variant():
    """OpenMM has CYS and CYX; Amber's CYM, a thiolate, lacks HG as CYX does."""

    from mdprep.protonation.temporary_hydrogenation import _requested_variants

    chain = SimpleNamespace(id="A")
    residues = [
        SimpleNamespace(chain=chain, id="317", insertionCode="", name="CYS"),
        SimpleNamespace(chain=chain, id="318", insertionCode="", name="GLU"),
    ]
    topology = SimpleNamespace(residues=lambda: iter(residues))

    variants, requested = _requested_variants(
        topology, {("A", 317, None): "CYM", ("A", 318, None): "GLH"}
    )

    assert variants == ["CYX", "GLH"]
    assert requested == (("A:CYS317", "CYM"), ("A:GLU318", "GLH"))


def test_the_xtb_hessian_takes_an_electronic_temperature():
    from mdprep.metals.gxtb_hessian import GxtbHessianError, build_gxtb_hessian_command

    command = build_gxtb_hessian_command(
        executable="xtb", xyz_name="m.xyz", charge=-2, spin=1, accuracy=0.001,
        electronic_temperature_kelvin=1000.0,
    )
    assert command[-2:] == ["--etemp", "1000"]
    assert "--etemp" not in build_gxtb_hessian_command(
        executable="xtb", xyz_name="m.xyz", charge=-2, spin=1, accuracy=0.001
    )
    with pytest.raises(GxtbHessianError, match="electronic temperature"):
        build_gxtb_hessian_command(
            executable="xtb", xyz_name="m.xyz", charge=-2, spin=1, accuracy=0.001,
            electronic_temperature_kelvin=0.0,
        )


def test_the_porphyrin_takes_its_hydrogens_from_the_proposed_smiles():
    pytest.importorskip("rdkit")
    from mdprep.ligands.hydrogenate import add_ligand_hydrogens

    (porphyrin,) = [
        residue for residue in read_pdb(CYP119).residues if residue.id.display() == "A:HEM1401"
    ]

    result = add_ligand_hydrogens(porphyrin, HEME_SMILES, ligand_id="a_hem_1401", net_charge=-4)

    # C34H30N4O4(4-): both pyrrole NH and both propionate OH gone.
    assert result.added_hydrogens == 30
    assert [atom.name for atom in result.residue.atoms[:42]] == porphyrin.atom_names()
