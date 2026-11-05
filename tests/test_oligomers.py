"""Dimers, trimers, tetramers: copies of one protein, parameterised once.

SyrB2 (PDB 2FCT) is two copies of the halogenase, each with its iron,
alpha-ketoglutarate, a chloride and a water on the iron -- and the copies
number their heterogens differently: chain A's cofactor is AKG897 and chain
B's AKG898, A's iron FE2900 and B's FE2899. A clone made by residue number
would give B's iron A's numbers and find nothing. These tests build every
variant from the deposition itself: renumbered, with a loop missing, laid
out the way a structure prediction writes ligands, and with one copy's
site different.
"""

from __future__ import annotations

from dataclasses import replace
from math import cos, dist, sin
from pathlib import Path

import pytest

from mdprep.config.decisions import (
    DecisionError,
    build_manifest,
    plan_decisions,
    proposed_answers,
)
from mdprep.config.models import ManifestConfig
from mdprep.structure.classify import is_standard_protein_residue, is_water_residue
from mdprep.structure.models import PdbStructure, build_residues
from mdprep.structure.oligomer import analyse_oligomer, kabsch
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import write_pdb


DEPOSITION = Path("examples/tutorials/syrb2_2fct/2FCT.pdb")
AKG_SMILES = "[O-]C(=O)CCC(=O)C(=O)[O-]"

# Chain A's site, as a user answers it in the builder.
SHARED = {
    "structure.chains": "all",
    "structure.remove_unknown_heterogens": True,
    "structure.repair.add_missing_heavy_atoms": True,
    "ligand.a_dsu_896.include": False,
    "ligand.a_akg_897.net_charge": -2,
    "ligand.a_akg_897.smiles": AKG_SMILES,
    "metal.a_fe2_900_site.confirm_donors": True,
    "metal.a_fe2_900_site.model_charge": -1,
    "refinement.total_qm_multiplicity": 5,
}

# Chain B's, for when it is answered on its own.
CHAIN_B = {
    "ligand.b_akg_898.net_charge": -2,
    "ligand.b_akg_898.smiles": AKG_SMILES,
    "metal.b_fe2_899_site.confirm_donors": True,
    "metal.b_fe2_899_site.model_charge": -1,
}


def written(structure: PdbStructure, path: Path) -> Path:
    write_pdb(structure, path)
    return path


def with_atoms(structure: PdbStructure, atoms) -> PdbStructure:
    atoms = list(atoms)
    return PdbStructure(
        path=structure.path,
        atoms=atoms,
        residues=build_residues(atoms),
        model_count=1,
    )


def renumbered(structure: PdbStructure, chain: str, offset: int) -> PdbStructure:
    return with_atoms(
        structure,
        (
            replace(atom, resid=atom.resid + offset) if atom.chain_id == chain else atom
            for atom in structure.atoms
        ),
    )


@pytest.fixture(scope="module")
def deposition() -> PdbStructure:
    return read_pdb(DEPOSITION)


@pytest.fixture(scope="module")
def plan():
    return plan_decisions(DEPOSITION)


# --------------------------------------------------------------------------
# Which residue is which
# --------------------------------------------------------------------------


def test_two_copies_are_recognised_and_paired_by_place_not_number(deposition):
    analysis = analyse_oligomer(deposition.residues)

    assert analysis.groups == (("A", "B"),)
    assert analysis.alignments["B"].identity == 1.0
    pairs = {
        copy.display(): reference.display()
        for copy, reference in analysis.heterogen_counterparts().items()
    }
    assert pairs == {
        "B:AKG898": "A:AKG897",
        "B:FE2899": "A:FE2900",
        "B:CL902": "A:CL901",
        "B:CL904": "A:CL903",
        "B:HOH1019": "A:HOH1072",
    }
    # The detergent is chain A's alone, and chain A asks its own questions.
    assert analysis.unmatched == {}


def test_a_copy_numbered_from_a_thousand_pairs_the_same(deposition):
    analysis = analyse_oligomer(renumbered(deposition, "B", 1000).residues)

    assert analysis.alignments["B"].renumbered
    pairs = {
        copy.display(): reference.display()
        for copy, reference in analysis.heterogen_counterparts().items()
    }
    assert pairs["B:AKG1898"] == "A:AKG897"
    assert pairs["B:FE21899"] == "A:FE2900"
    # Histidine 116 of chain B is now 1116, and is still A's 116.
    his = next(
        copy for copy, reference in analysis.counterparts.items()
        if copy.chain_id == "B" and copy.resid == 1116
    )
    assert analysis.counterparts[his].resid == 116


def test_a_copy_missing_a_loop_still_pairs_on_both_sides(deposition):
    gap = {150, 151, 152, 153, 154, 155}
    structure = with_atoms(
        deposition,
        (
            atom
            for atom in deposition.atoms
            if not (atom.chain_id == "B" and atom.resid in gap and atom.record_name == "ATOM")
        ),
    )

    analysis = analyse_oligomer(structure.residues)

    alignment = analysis.alignments["B"]
    assert alignment.identity == 1.0
    paired = {copy.resid for _, copy in alignment.pairs}
    assert {149, 156} <= paired and not paired & gap
    assert {c.display() for c in analysis.heterogen_counterparts()} >= {"B:AKG898", "B:FE2899"}


def test_different_proteins_are_not_copies(deposition):
    """A heterodimer: chain B is something else entirely."""

    swap = {"ALA": "TRP", "LEU": "GLY", "GLU": "PRO", "VAL": "CYS", "SER": "MET"}
    structure = with_atoms(
        deposition,
        (
            replace(atom, resname=swap.get(atom.resname, atom.resname))
            if atom.chain_id == "B" and atom.record_name == "ATOM"
            else atom
            for atom in deposition.atoms
        ),
    )

    analysis = analyse_oligomer(structure.residues)

    assert analysis.groups == (("A",), ("B",))
    assert not analysis.homomeric
    assert analysis.counterparts == {}


def prediction_layout(structure: PdbStructure) -> PdbStructure:
    """2FCT as a co-folding model writes it: no waters, every ligand a chain.

    Boltz and AlphaFold3 give each ligand a chain of its own, so chain
    letters say nothing about which copy a cofactor belongs to.
    """

    letters = iter("CDEFGHIJKLMNOP")
    new_chain: dict[tuple[str, int], str] = {}
    atoms = []
    for residue in structure.residues:
        if is_water_residue(residue) or residue.id.resname == "DSU":
            continue
        if is_standard_protein_residue(residue):
            atoms.extend(residue.atoms)
            continue
        key = (residue.id.chain_id, residue.id.resid)
        chain = new_chain.setdefault(key, next(letters))
        atoms.extend(replace(atom, chain_id=chain, resid=1) for atom in residue.atoms)
    return with_atoms(structure, atoms)


def test_a_predicted_ligand_belongs_to_the_chain_it_sits_against(deposition):
    analysis = analyse_oligomer(prediction_layout(deposition).residues)

    owners = analysis.chain_of_nonprotein_chain
    # In file order: A's iron, two chlorides and cofactor, then B's.
    assert [owners[chain] for chain in "CDEF"] == ["A"] * 4
    assert [owners[chain] for chain in "GHIJ"] == ["B"] * 4
    assert {copy.display(): reference.display() for copy, reference in
            analysis.heterogen_counterparts().items()} == {
        "G:FE21": "C:FE21",
        "H:CL1": "D:CL1",
        "I:CL1": "E:CL1",
        "J:AKG1": "F:AKG1",
    }


def test_keeping_one_predicted_copy_keeps_its_own_cofactors_only(tmp_path, deposition):
    """Every heterogen chain used to be kept whichever protein chain was
    chosen, which left the other copy's iron and cofactor in the box."""

    plan = plan_decisions(written(prediction_layout(deposition), tmp_path / "model.pdb"))
    answers = {
        "structure.chains": "A",
        "structure.repair.add_missing_heavy_atoms": True,
        "ligand.f_akg_1.net_charge": -2,
        "ligand.f_akg_1.smiles": AKG_SMILES,
        "ligand.f_akg_1.charge_method": "am1bcc",
        "metal.c_fe2_1_site.model": "nonbonded",
        "structure.remove_unknown_heterogens": False,
    }
    asked = {decision.id for decision in plan.applicable(answers)}

    assert {"ligand.f_akg_1.include", "metal.c_fe2_1_site.model"} <= asked
    assert not any(key.startswith(("ligand.j_", "metal.g_")) for key in asked)
    manifest = build_manifest(plan, answers)
    assert manifest["structure"]["keep_chains"] == ["A", "C", "D", "E", "F"]


def test_a_copy_whose_iron_lost_its_water_is_not_a_copy_of_the_site(tmp_path, deposition):
    structure = with_atoms(
        deposition,
        (atom for atom in deposition.atoms if (atom.chain_id, atom.resid) != ("B", 1019)),
    )

    analysis = analyse_oligomer(structure.residues)

    reason = analysis.unmatched[next(r for r in analysis.unmatched if r.resname == "FE2")]
    assert "A:HOH1072 O" in reason
    # The cofactor on that iron goes with it: its charges may come from the fit.
    assert any(residue.resname == "AKG" for residue in analysis.unmatched)


def test_kabsch_recovers_a_rigid_motion():
    points = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 2.0, 0.5), (1.0, 1.0, 3.0), (-2.0, 0.5, 1.0)]
    angle = 0.7
    moved = [
        (
            cos(angle) * x - sin(angle) * y + 4.0,
            sin(angle) * x + cos(angle) * y - 3.0,
            z + 2.5,
        )
        for x, y, z in points
    ]

    frame = kabsch(moved, points)

    for original, shifted in zip(points, moved):
        assert dist(frame.apply(shifted), original) < 1e-9


# --------------------------------------------------------------------------
# The question
# --------------------------------------------------------------------------


def test_the_copies_question_comes_with_all_chains_and_defaults_to_sharing(plan):
    decision = plan.by_id("structure.same_parametrization")

    assert decision.default == "true"
    assert [c.to_dict() for c in decision.when] == [
        {"decision_id": "structure.chains", "values": ["all"]}
    ]
    assert "B:AKG898 would share A:AKG897's parameters" in decision.evidence
    assert "Chain B: 100% identical to chain A over 299 residues" in decision.evidence
    ids = [d.id for d in plan.decisions]
    assert ids.index("structure.same_parametrization") == ids.index("structure.chains") + 1


def test_sharing_asks_about_chain_a_only(plan):
    asked = {d.id for d in plan.applicable({**SHARED, "structure.same_parametrization": True})}

    assert {"ligand.a_akg_897.include", "metal.a_fe2_900_site.model"} <= asked
    assert not any(key.startswith(("ligand.b_", "metal.b_")) for key in asked)


def test_not_sharing_asks_about_every_copy(plan):
    asked = {d.id for d in plan.applicable({**SHARED, "structure.same_parametrization": False})}

    assert {"ligand.b_akg_898.include", "metal.b_fe2_899_site.model"} <= asked


def test_keeping_chain_b_alone_asks_about_chain_b(plan):
    asked = {d.id for d in plan.applicable({"structure.chains": "B"})}

    assert "ligand.b_akg_898.include" in asked
    assert "structure.same_parametrization" not in asked
    assert not any(key.startswith(("ligand.a_", "metal.a_")) for key in asked)


def test_a_copys_question_names_the_one_it_stands_for(plan):
    decision = plan.by_id("metal.b_fe2_899_site.confirm_donors")

    assert decision.copy_of == "metal.a_fe2_900_site.confirm_donors"
    rendered = decision.to_dict()
    assert rendered["copy_of"] == "metal.a_fe2_900_site.confirm_donors"
    assert {"decision_id": "structure.same_parametrization", "values": ["false"]} in rendered[
        "when_any"
    ]


# --------------------------------------------------------------------------
# The manifest
# --------------------------------------------------------------------------


def test_shared_answers_build_every_copy_with_its_own_residues(plan):
    data = build_manifest(plan, SHARED)
    manifest = ManifestConfig.model_validate(data)

    ligands = {ligand.id: ligand for ligand in manifest.ligands}
    assert ligands["b_akg_898"].parameters_from == "a_akg_897"
    assert ligands["b_akg_898"].selector.resid == 898
    assert ligands["b_akg_898"].smiles == AKG_SMILES

    sites = {site.id: site for site in manifest.metals}
    copy = sites["b_fe2_899_site"]
    assert copy.parameters_from == "a_fe2_900_site"
    assert copy.ions[0].selector.resid == 899
    # In chain A's order, each bond the counterpart of A's in the same place.
    assert [
        (bond.coordinator.resname, bond.coordinator.resid, bond.coordinator.atom_name)
        for bond in copy.mcpb.bonds
    ] == [
        ("AKG", 898, "O2"),
        ("HIS", 116, "NE2"),
        ("AKG", 898, "O5"),
        ("HIS", 235, "NE2"),
        ("HOH", 1019, "O"),
        ("CL", 902, "CL"),
    ]
    assert [ion.selector.resid for ion in copy.coordinating_ions] == [902]


def test_only_the_fitted_copy_is_refined(plan):
    data = build_manifest(plan, SHARED)

    assert data["refinement"]["qm_components"] == {
        "ligands": ["a_akg_897"],
        "metal_sites": ["a_fe2_900_site"],
        "metal_coordinating_residues": {},
    }


def test_a_renumbered_copy_is_built_with_its_own_numbers(tmp_path, deposition):
    structure = written(renumbered(deposition, "B", 1000), tmp_path / "renumbered.pdb")
    plan = plan_decisions(structure)

    data = build_manifest(plan, SHARED)

    copy = next(site for site in data["metals"] if site["id"] == "b_fe2_1899_site")
    assert copy["parameters_from"] == "a_fe2_900_site"
    assert {bond["coordinator"]["resid"] for bond in copy["mcpb"]["bonds"]} == {
        1898, 1116, 1235, 2019, 1902
    }
    ManifestConfig.model_validate(data)


def test_two_fits_are_refused_with_the_way_out(plan):
    answers = {**SHARED, "structure.same_parametrization": False, **CHAIN_B}

    with pytest.raises(DecisionError, match="Answer yes to giving every copy the same parameters"):
        build_manifest(plan, answers)


def test_a_copy_answered_alike_shares_the_template_anyway(plan):
    """tleap builds every AKG from one template, so two AKGs answered alike
    are parameterised once -- and the manifest says so."""

    answers = {
        **SHARED,
        "structure.same_parametrization": False,
        "metal.a_fe2_900_site.model": "nonbonded",
        "metal.b_fe2_899_site.model": "nonbonded",
        "metal.b_fe2_899_site.charge": 2,
        "ligand.a_akg_897.charge_method": "am1bcc",
        "ligand.b_akg_898.charge_method": "am1bcc",
        "ligand.b_akg_898.net_charge": -2,
        "ligand.b_akg_898.smiles": AKG_SMILES,
    }

    data = build_manifest(plan, answers)

    b_akg = next(block for block in data["ligands"] if block["id"] == "b_akg_898")
    assert b_akg["parameters_from"] == "a_akg_897"
    assert data["refinement"] == {"enabled": False}
    ManifestConfig.model_validate(data)


def test_one_name_two_charges_is_refused(plan):
    answers = {
        **SHARED,
        "structure.same_parametrization": False,
        "metal.a_fe2_900_site.model": "nonbonded",
        "metal.b_fe2_899_site.model": "nonbonded",
        "metal.b_fe2_899_site.charge": 2,
        "ligand.a_akg_897.charge_method": "am1bcc",
        "ligand.b_akg_898.charge_method": "am1bcc",
        "ligand.b_akg_898.net_charge": -1,
        "ligand.b_akg_898.smiles": "OC(=O)CCC(=O)C(=O)[O-]",
    }

    with pytest.raises(DecisionError, match="both called AKG but differ in net_charge, smiles"):
        build_manifest(plan, answers)


def test_one_metal_name_two_oxidation_states_is_refused(plan):
    answers = {
        **SHARED,
        "structure.same_parametrization": False,
        "metal.a_fe2_900_site.model": "nonbonded",
        "metal.b_fe2_899_site.model": "nonbonded",
        "metal.b_fe2_899_site.charge": 3,
        "ligand.a_akg_897.charge_method": "am1bcc",
        "ligand.b_akg_898.charge_method": "am1bcc",
        "ligand.b_akg_898.net_charge": -2,
        "ligand.b_akg_898.smiles": AKG_SMILES,
    }

    with pytest.raises(DecisionError, match="both called FE2 but differ in charge"):
        build_manifest(plan, answers)


def test_the_region_spin_is_not_asked_when_no_site_is_bonded(plan):
    answers = {
        **SHARED,
        "metal.a_fe2_900_site.model": "nonbonded",
        "ligand.a_akg_897.charge_method": "am1bcc",
    }

    asked = {decision.id for decision in plan.applicable(answers)}

    assert "refinement.total_qm_multiplicity" not in asked


def test_the_proposals_still_add_up_for_the_reference_site(plan):
    answers = {key: value for key, value in SHARED.items() if "model_charge" not in key}

    assert proposed_answers(plan, answers)["metal.a_fe2_900_site.model_charge"] == "-1"


def test_the_wizard_names_the_shared_blocks(plan):
    from mdprep.workflows.wizard import render_manifest_yaml

    text = render_manifest_yaml(plan, SHARED)

    assert "#   b_akg_898 <- a_akg_897" in text
    assert "#   b_fe2_899_site <- a_fe2_900_site" in text


# --------------------------------------------------------------------------
# What a hand-written manifest may say
# --------------------------------------------------------------------------


def shared_manifest(plan) -> dict:
    import copy

    return copy.deepcopy(build_manifest(plan, SHARED))


def block(data: dict, section: str, block_id: str) -> dict:
    return next(item for item in data[section] if item["id"] == block_id)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda data: block(data, "ligands", "b_akg_898").update(parameters_from="nothing"),
            "which is not a configured ligand",
        ),
        (
            lambda data: block(data, "ligands", "b_akg_898").update(net_charge=-1),
            "differs from it in net_charge",
        ),
        (
            lambda data: block(data, "metals", "b_fe2_899_site").update(
                parameters_from="b_fe2_899_site"
            ),
            "must name a site that is parameterised itself",
        ),
        (
            lambda data: block(data, "metals", "b_fe2_899_site")["mcpb"]["bonds"].reverse(),
            "list the bonds in the reference's order",
        ),
        (
            lambda data: block(data, "metals", "b_fe2_899_site")["ions"][0].update(charge=3),
            "differs from FE2900 in element, charge or multiplicity",
        ),
        (
            lambda data: block(data, "metals", "b_fe2_899_site")["mcpb"].update(
                small_model_charge=0
            ),
            "mcpb settings differ in small_model_charge",
        ),
        (
            lambda data: block(data, "metals", "b_fe2_899_site").pop("parameters_from"),
            "would each need an MCPB fit of their own",
        ),
    ],
)
def test_a_copy_that_is_not_one_is_refused(plan, edit, message):
    data = shared_manifest(plan)
    edit(data)

    with pytest.raises(ValueError, match=message):
        ManifestConfig.model_validate(data)


def test_a_fitted_cofactor_can_only_be_shared_with_a_copy_of_its_site(plan):
    data = shared_manifest(plan)
    site = block(data, "metals", "b_fe2_899_site")
    site.update(model="nonbonded", nonbonded={"parameter_set": "12_6"})
    site.pop("mcpb")
    site.pop("parameters_from")

    with pytest.raises(ValueError, match="has to coordinate a copy of that ligand's site"):
        ManifestConfig.model_validate(data)


def test_a_nonbonded_copy_repeats_its_set(plan):
    answers = {
        **SHARED,
        "metal.a_fe2_900_site.model": "nonbonded",
        "metal.a_fe2_900_site.parameter_set": "hfe",
        "ligand.a_akg_897.charge_method": "am1bcc",
    }
    data = build_manifest(plan, answers)
    copy = block(data, "metals", "b_fe2_899_site")
    assert copy["parameters_from"] == "a_fe2_900_site"
    assert copy["nonbonded"] == {"parameter_set": "hfe"}
    ManifestConfig.model_validate(data)

    copy["nonbonded"] = {"parameter_set": "12_6"}
    with pytest.raises(ValueError, match="another ion parameter set"):
        ManifestConfig.model_validate(data)


# --------------------------------------------------------------------------
# Building a copy of a fitted site
# --------------------------------------------------------------------------


def _atom(serial, name, resname, chain, resid, x, y, z, element, record="ATOM"):
    from mdprep.structure.models import AtomRecord

    return AtomRecord(
        serial=serial, name=name, altloc=None, resname=resname, chain_id=chain,
        resid=resid, icode=None, x=x, y=y, z=z, occupancy=1.0, bfactor=0.0,
        element=element, record_name=record, original_line="",
    )


def two_zinc_copies(*, b_middle="HIS", b_first="ALA", b_proton="HD1") -> PdbStructure:
    """Two copies of ALA-HIS-ALA with a zinc on the histidine's NE2.

    Chain B sits 30 A away, and its zinc is numbered 501 to A's 500 --
    copies are often numbered apart."""

    atoms = []
    for chain, shift, first, middle, proton in (
        ("A", 0.0, "ALA", "HIS", "HD1"),
        ("B", 30.0, b_first, b_middle, b_proton),
    ):
        for resid, resname in ((1, first), (2, middle), (3, "ALA")):
            x0 = shift + 3.8 * (resid - 1)
            names = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
            if resid == 2:
                names += [("NE2", "N"), (proton, "H")]
            for offset, (name, element) in enumerate(names):
                atoms.append(
                    _atom(len(atoms) + 1, name, resname, chain, resid,
                          x0 + 0.3 * offset, 0.5 * offset, 0.0, element)
                )
    for chain, resid, shift in (("A", 500, 0.0), ("B", 501, 30.0)):
        ne2 = next(a for a in atoms if a.chain_id == chain and a.name == "NE2")
        atoms.append(
            _atom(len(atoms) + 1, "ZN", "ZN", chain, resid, ne2.x, ne2.y + 2.1, 0.0, "Zn", "HETATM")
        )
    return with_atoms(PdbStructure(path=Path("two.pdb"), atoms=[], residues=[], model_count=1), atoms)


def zinc_manifest() -> ManifestConfig:
    def site(chain, resid, **extra):
        zinc = {"chain": chain, "resname": "ZN", "resid": resid, "atom_name": "ZN"}
        return {
            "id": f"{chain.lower()}_zn_site",
            "model": "bonded_mcpb",
            "ions": [{"selector": zinc, "element": "Zn", "charge": 2, "multiplicity": 1}],
            "mcpb": {
                "workflow": "prepare_inputs",
                "provisional_nonbonded_parameter_set": "12_6",
                "bonds": [
                    {
                        "ion": zinc,
                        "coordinator": {
                            "chain": chain, "resname": "HIS", "resid": 2, "atom_name": "NE2"
                        },
                    }
                ],
                "small_model_charge": 2,
                "small_model_spin": 1,
                "large_model_charge": 2,
                "large_model_spin": 1,
            },
            **extra,
        }

    return ManifestConfig.model_validate(
        {
            "project": {"name": "t", "input_structure": "two.pdb", "output_dir": "out"},
            "structure": {},
            "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
            "protonation": {"method": "manual_only"},
            "disulfides": {},
            "metals": [site("A", 500), site("B", 501, parameters_from="a_zn_site")],
            "solvation": {"enabled": False},
            "validation": {},
        }
    )


def fitted_zinc(tmp_path, structure):
    """What a fit of chain A's site leaves behind, without running MCPB.py."""

    from mdprep.metals.coordination import resolve_metal_sites
    from mdprep.metals.mcpb import McpbSiteResult, write_mcpb_numbered_pdb

    manifest = zinc_manifest()
    sites = resolve_metal_sites(structure, manifest, validate_mcpb_cutoff=False)
    numbering = write_mcpb_numbered_pdb(structure, tmp_path / "numbered.pdb")
    restored = with_atoms(
        structure,
        (
            replace(atom, resname={"HIS": "HD1", "ZN": "ZN1"}[atom.resname])
            if atom.chain_id == "A" and atom.resname in {"HIS", "ZN"}
            else atom
            for atom in structure.atoms
        ),
    )
    write_pdb(restored, tmp_path / "restored.pdb")
    result = McpbSiteResult(
        site_id="a_zn_site",
        workflow="pyscf",
        complete=True,
        work_dir=tmp_path,
        input_path=tmp_path / "mcpb.in",
        mcpb_input_pdb_path=tmp_path / "numbered.pdb",
        restored_final_pdb_path=tmp_path / "restored.pdb",
        final_frcmod_path=tmp_path / "site.frcmod",
        generated_tleap_path=tmp_path / "tleap.in",
        leap_setup_commands=("HD1 = loadmol2 /x/HD1.mol2", "ZN1 = loadmol2 /x/ZN1.mol2"),
        leap_bond_commands=(
            "bond system.2.NE2 system.7.ZN",
            "bond system.1.C system.2.N",
            "bond system.2.C system.3.N",
        ),
        residue_renames=(
            {"chain": "A", "resid": 2, "icode": None, "original_resname": "HIS",
             "final_resname": "HD1", "source": "MCPB.py"},
            {"chain": "A", "resid": 500, "icode": None, "original_resname": "ZN",
             "final_resname": "ZN1", "source": "MCPB.py"},
        ),
        runs=(),
        expected_qm_artifacts=(),
        generated_files=(),
        amberhome=tmp_path,
    )
    return result, sites, manifest, numbering


def copy_site(tmp_path, structure):
    from mdprep.metals.mcpb import apply_mcpb_site_copies

    result, sites, manifest, numbering = fitted_zinc(tmp_path, structure)
    return apply_mcpb_site_copies(
        result,
        site=sites[0],
        copies=[sites[1]],
        structure=structure,
        manifest=manifest,
        numbering=numbering,
        output_path=tmp_path / "with_copies.pdb",
    )


def test_a_copy_takes_the_fits_templates_and_bonds(tmp_path):
    built = copy_site(tmp_path, two_zinc_copies())

    [copy] = built.copies
    assert [(r["chain"], r["resid"], r["original_resname"], r["final_resname"])
            for r in copy.residue_renames] == [("B", 501, "ZN", "ZN1"), ("B", 2, "HIS", "HD1")]
    assert all("a_zn_site" in r["source"] for r in copy.residue_renames)
    # B's histidine is residue 5 and its zinc 8: the same bonds, B's numbers.
    assert copy.leap_bond_commands == (
        "bond system.5.NE2 system.8.ZN",
        "bond system.4.C system.5.N",
        "bond system.5.C system.6.N",
    )
    assert built.all_leap_bond_commands[:3] == built.leap_bond_commands
    names = {(r.id.chain_id, r.id.resid): r.id.resname
             for r in read_pdb(built.final_pdb_path).residues}
    assert names[("B", 2)] == "HD1" and names[("B", 501)] == "ZN1"
    assert names[("A", 2)] == "HD1"


def test_a_copy_protonated_differently_cannot_take_the_template(tmp_path):
    from mdprep.metals.mcpb import McpbWorkflowError

    with pytest.raises(McpbWorkflowError, match="protonated the same way"):
        copy_site(tmp_path, two_zinc_copies(b_proton="HE2"))


def test_a_copy_with_another_residue_beside_its_donor_is_refused(tmp_path):
    from mdprep.metals.mcpb import McpbWorkflowError

    with pytest.raises(McpbWorkflowError, match="needs the same chain around it"):
        copy_site(tmp_path, two_zinc_copies(b_first="GLY"))


def test_a_bonded_site_between_two_chains_is_refused(tmp_path):
    from mdprep.metals.coordination import ResolvedMetalBond
    from mdprep.metals.mcpb import McpbWorkflowError, _validate_site_in_one_chain

    structure = two_zinc_copies()
    _, sites, _, _ = fitted_zinc(tmp_path, structure)
    site = sites[0]
    b_his = next(r for r in structure.residues if r.id.chain_id == "B" and r.id.resid == 2)
    across = replace(
        site,
        bonds=site.bonds
        + (
            ResolvedMetalBond(
                ion=site.ions[0],
                coordinator=b_his.atoms[4],
                coordinator_residue=b_his,
                distance_angstrom=2.1,
                discovered_by_cutoff=False,
            ),
        ),
    )

    with pytest.raises(McpbWorkflowError, match="bound by residues of chains A, B"):
        _validate_site_in_one_chain(across)


# --------------------------------------------------------------------------
# Ligand copies through the build
# --------------------------------------------------------------------------


def test_a_ligand_copy_reuses_the_reference_files(tmp_path):
    from mdprep.ligands.extract import ExtractedLigand
    from mdprep.ligands.workflow import LigandWorkflowItem, _each_once

    deposition = read_pdb(DEPOSITION)
    plan = plan_decisions(DEPOSITION)
    manifest = ManifestConfig.model_validate(build_manifest(plan, SHARED))
    residues = {(r.id.chain_id, r.id.resid): r for r in deposition.residues}
    extracted = [
        ExtractedLigand(
            config=ligand,
            residue=residues[(ligand.selector.chain, ligand.selector.resid)],
            pdb_path=tmp_path / f"{ligand.id}.pdb",
            identity_path=tmp_path / f"{ligand.id}.json",
            warnings=[],
        )
        for ligand in manifest.ligands
    ]
    calls = []

    def parameterise(item):
        calls.append(item.config.id)
        return LigandWorkflowItem(
            ligand_id=item.config.id, selector={}, residue_identity={}, atom_count=10,
            charge_method="am1bcc", atom_types="gaff2", net_charge=-2, multiplicity=1,
            extracted_pdb_path=item.pdb_path, identity_path=item.identity_path,
            final_mol2_path=tmp_path / "AKG.mol2", final_frcmod_path=tmp_path / "AKG.frcmod",
        )

    items = _each_once(extracted, parameterise)

    assert calls == ["a_akg_897"]
    assert [item.ligand_id for item in items] == ["a_akg_897", "b_akg_898"]
    assert items[1].final_mol2_path == items[0].final_mol2_path
    assert items[1].extracted_pdb_path == tmp_path / "b_akg_898.pdb"
    assert "parameterised once" in items[1].warnings[-1]


def test_a_ligand_copy_with_other_atom_names_is_refused(tmp_path):
    from mdprep.ligands.extract import ExtractedLigand
    from mdprep.ligands.workflow import LigandWorkflowError, _share_parameters

    deposition = read_pdb(DEPOSITION)
    plan = plan_decisions(DEPOSITION)
    manifest = ManifestConfig.model_validate(build_manifest(plan, SHARED))
    a, b = manifest.ligands
    residues = {(r.id.chain_id, r.id.resid): r for r in deposition.residues}
    renamed = residues[("B", 898)]
    renamed = type(renamed)(
        id=renamed.id,
        atoms=[replace(atom, name="X" + atom.name[:3]) if atom.name == "O1" else atom
               for atom in renamed.atoms],
        record_names=renamed.record_names,
        original_index=renamed.original_index,
    )
    reference = ExtractedLigand(a, residues[("A", 897)], tmp_path / "a.pdb", tmp_path / "a.json", [])
    copy = ExtractedLigand(b, renamed, tmp_path / "b.pdb", tmp_path / "b.json", [])

    with pytest.raises(LigandWorkflowError, match="same atoms, named alike"):
        _share_parameters(copy, object(), reference)


# --------------------------------------------------------------------------
# Refusals a front end can place
# --------------------------------------------------------------------------


def test_a_second_fit_is_refused_at_the_question_that_asks_for_it(plan):
    from mdprep.config.decisions import answer_conflicts

    answers = {**SHARED, "structure.same_parametrization": False, **CHAIN_B}
    for key in [k for k in answers if "model_charge" in k or "multiplicity" in k]:
        answers.pop(key)

    conflict = answer_conflicts(plan, answers)

    assert conflict.decision_id == "metal.b_fe2_899_site.model"
    assert "A:FE2900 and B:FE2899" in str(conflict)
    with pytest.raises(DecisionError) as refused:
        build_manifest(plan, {**SHARED, "structure.same_parametrization": False, **CHAIN_B})
    assert refused.value.decision_id == "metal.b_fe2_899_site.model"


def test_sharing_raises_no_conflict(plan):
    from mdprep.config.decisions import answer_conflicts

    assert answer_conflicts(plan, SHARED) is None


def test_a_fitted_cofactor_on_a_nonbonded_metal_is_refused_where_it_happens(plan):
    """The cofactor's charge method defaults to the site's fit, and is asked
    before the site: a site made nonbonded afterwards left it pointing at a
    fit that never runs."""

    from mdprep.config.decisions import answer_conflicts

    answers = {**SHARED, "structure.chains": "A", "metal.a_fe2_900_site.model": "nonbonded",
               "ligand.a_akg_897.charge_method": "mcpb_resp_pyscf"}

    conflict = answer_conflicts(plan, answers)

    assert conflict.decision_id == "metal.a_fe2_900_site.model"
    assert "A:AKG897's charges were to come from the MCPB fit" in str(conflict)
    with pytest.raises(DecisionError, match="such as AM1-BCC") as refused:
        build_manifest(plan, answers)
    assert refused.value.decision_id == "metal.a_fe2_900_site.model"


def test_a_name_conflict_points_at_the_answer_that_differs(plan):
    answers = {
        **SHARED,
        "structure.same_parametrization": False,
        "metal.a_fe2_900_site.model": "nonbonded",
        "metal.b_fe2_899_site.model": "nonbonded",
        "metal.b_fe2_899_site.charge": 2,
        "ligand.a_akg_897.charge_method": "am1bcc",
        "ligand.b_akg_898.charge_method": "gas_resp_pyscf",
        "ligand.b_akg_898.net_charge": -2,
        "ligand.b_akg_898.smiles": AKG_SMILES,
    }

    with pytest.raises(DecisionError) as refused:
        build_manifest(plan, answers)

    assert refused.value.decision_id == "ligand.b_akg_898.charge_method"


def test_the_wizard_refuses_a_second_fit_at_its_question_and_asks_again(tmp_path):
    from mdprep.workflows.wizard import run_wizard

    replies = {
        **{key: [str(value)] for key, value in SHARED.items()},
        "structure.same_parametrization": ["false"],
        "ligand.b_akg_898.net_charge": ["-2"],
        "ligand.b_akg_898.smiles": [AKG_SMILES],
        "ligand.b_akg_898.charge_method": ["am1bcc"],
        # Bonded first, as proposed: refused there and then; nonbonded next.
        "metal.b_fe2_899_site.model": ["bonded_mcpb", "nonbonded"],
    }
    asked: list[str] = []
    told: list[str] = []

    def prompt(decision):
        asked.append(decision.id)
        queue = replies.get(decision.id)
        return queue.pop(0) if queue else ""

    with pytest.raises(DecisionError, match="both called AKG but differ in charge_method"):
        run_wizard(DEPOSITION, output_path=tmp_path / "system.yaml", prompt=prompt, notify=told.append)

    assert asked.count("metal.b_fe2_899_site.model") == 2
    assert told and "A:FE2900 and B:FE2899 would each need a bonded MCPB fit" in told[0]


# --------------------------------------------------------------------------
# Copies of one site, drawn apart
#
# A predicted dimer seldom puts the same donor at the same distance from both
# irons. With the shells compared at exactly 2.6 A, a chloride 2.47 A from
# one iron and 2.70 A from the other made two sites of one: asked about
# separately, each needing a fit, and refused for that -- after the student
# had answered that the copies share their parameters.
# --------------------------------------------------------------------------

# Chain A's answers for 2FCT laid out as a prediction (prediction_layout).
PREDICTED = {
    "structure.chains": "all",
    "structure.remove_unknown_heterogens": False,
    "structure.repair.add_missing_heavy_atoms": True,
    "ligand.f_akg_1.net_charge": -2,
    "ligand.f_akg_1.smiles": AKG_SMILES,
    "metal.c_fe2_1_site.confirm_donors": True,
    "metal.c_fe2_1_site.model_charge": -1,
    "refinement.total_qm_multiplicity": 5,
}


def slid(structure: PdbStructure, metal: tuple[str, int], atom: tuple[str, int, str], to: float):
    """``structure`` with one atom moved along its line to a metal, ``to`` A from it."""

    centre = next(a for a in structure.atoms if (a.chain_id, a.resid) == metal)

    def moved(a):
        if (a.chain_id, a.resid, a.name) != atom:
            return a
        scale = to / dist((a.x, a.y, a.z), (centre.x, centre.y, centre.z))
        return replace(
            a,
            x=centre.x + (a.x - centre.x) * scale,
            y=centre.y + (a.y - centre.y) * scale,
            z=centre.z + (a.z - centre.z) * scale,
        )

    return with_atoms(structure, (moved(a) for a in structure.atoms))


def with_a_shared_additive(structure: PdbStructure) -> PdbStructure:
    """2FCT with chain A's DSU placed in chain B too, as a copy would hold it.

    Something besides the iron site that both copies share -- the second
    ligand a prediction puts in each monomer -- so the sharing question is
    asked whatever becomes of the site.
    """

    ca = {(a.chain_id, a.resid): (a.x, a.y, a.z) for a in structure.atoms if a.name == "CA"}
    paired = [key for key in ca if key[0] == "A" and ("B", key[1]) in ca]
    frame = kabsch([ca[key] for key in paired], [ca[("B", key[1])] for key in paired])
    serial = max(a.serial for a in structure.atoms)
    additive = []
    for offset, a in enumerate(a for a in structure.atoms if a.resname == "DSU"):
        x, y, z = frame.apply((a.x, a.y, a.z))
        additive.append(replace(a, chain_id="B", resid=1896, serial=serial + offset + 1, x=x, y=y, z=z))
    return with_atoms(structure, [*structure.atoms, *additive])


@pytest.fixture(scope="module")
def predicted(deposition) -> PdbStructure:
    return prediction_layout(deposition)


def test_a_donor_either_side_of_the_cutoff_is_one_site(predicted):
    structure = slid(predicted, ("G", 1), ("H", 1, "CL"), 2.7)

    analysis = analyse_oligomer(structure.residues)

    assert analysis.unmatched == {}
    assert {copy.display(): reference.display() for copy, reference in
            analysis.heterogen_counterparts().items()}["G:FE21"] == "C:FE21"
    # The copy's chloride is in its shell all the same, through the first copy's.
    [copied] = analysis.copied_donors[next(m for m in analysis.copied_donors)]
    assert (copied.residue.display(), copied.atom_name, copied.bound_to.display()) == (
        "H:CL1", "CL", "C:FE21"
    )
    assert copied.distance_there == pytest.approx(2.466, abs=0.01)


def test_copies_drawn_apart_share_one_fit(tmp_path, predicted):
    from mdprep.config.decisions import answer_conflicts

    plan = plan_decisions(written(slid(predicted, ("G", 1), ("H", 1, "CL"), 2.7), tmp_path / "m.pdb"))
    asked = {decision.id for decision in plan.applicable(PREDICTED)}

    assert "structure.same_parametrization" in asked
    assert not any(key.startswith(("ligand.j_", "metal.g_")) for key in asked)
    assert answer_conflicts(plan, PREDICTED) is None
    data = build_manifest(plan, PREDICTED)
    ManifestConfig.model_validate(data)
    copy = next(site for site in data["metals"] if site["id"] == "g_fe2_1_site")
    assert copy["parameters_from"] == "c_fe2_1_site"
    # Bonded to its chloride 2.70 A away, as the first copy is to its own.
    assert [(b["coordinator"]["chain"], b["coordinator"]["atom_name"]) for b in copy["mcpb"]["bonds"]] == [
        ("J", "O2"), ("B", "NE2"), ("J", "O5"), ("B", "NE2"), ("H", "CL")
    ]


def test_a_donor_further_out_in_the_first_copy_is_offered_there_too(tmp_path, predicted):
    """The first copy is the one asked about, so it has to offer what any copy binds."""

    plan = plan_decisions(written(slid(predicted, ("C", 1), ("F", 1, "O5"), 2.75), tmp_path / "m.pdb"))
    confirm = plan.by_id("metal.c_fe2_1_site.confirm_donors")

    assert "F:AKG1 O5 at 2.75 A; its counterpart is 2.28 A from G:FE21" in " ".join(confirm.evidence)
    assert "another copy of the protein" in confirm.why
    data = build_manifest(plan, PREDICTED)
    ManifestConfig.model_validate(data)
    fitted = next(site for site in data["metals"] if site["id"] == "c_fe2_1_site")
    assert ("F", "O5") in [
        (b["coordinator"]["chain"], b["coordinator"]["atom_name"]) for b in fitted["mcpb"]["bonds"]
    ]


def test_a_donor_that_has_left_still_makes_another_site_and_says_how_far(predicted):
    structure = slid(predicted, ("G", 1), ("J", 1, "O5"), 3.3)

    analysis = analyse_oligomer(structure.residues)

    reason = analysis.unmatched[next(r for r in analysis.unmatched if r.resname == "FE2")]
    assert "C:FE21 binds F:AKG1 O5 at 2.19 A, whose counterpart here is 3.30 A away" in reason
    assert analysis.copied_donors == {}


def test_the_refusal_of_two_fits_says_why_a_copy_is_not_one(tmp_path, deposition):
    """'These do not match it' was all a student got about two irons they
    knew to be alike. What differs, and by how much, is what they can act on."""

    from mdprep.config.decisions import answer_conflicts

    structure = slid(with_a_shared_additive(deposition), ("B", 899), ("B", 902, "CL"), 3.3)
    plan = plan_decisions(written(structure, tmp_path / "m.pdb"))
    answers = {**SHARED, "structure.same_parametrization": True,
               "metal.b_fe2_899_site.model": "bonded_mcpb"}

    conflict = answer_conflicts(plan, answers)

    assert conflict.decision_id == "metal.b_fe2_899_site.model"
    assert (
        "B:FE2899 does not: its first shell differs from A:FE2900's (A:FE2900 binds "
        "A:CL901 CL at 2.47 A, whose counterpart here is 3.30 A away)"
    ) in str(conflict)


def test_every_copy_of_a_fit_takes_the_lowest_cutoff(plan):
    """MCPB.py's cutoff drops below a donor left unticked, and a copy repeats
    the fit's settings. Each copy computed its own, from its own water, and
    the manifest failed config-check on the hundredths between them."""

    confirm = plan.by_id("metal.a_fe2_900_site.confirm_donors")
    answers = {
        **SHARED,
        "metal.a_fe2_900_site.confirm_donors": [
            choice.value for choice in confirm.choices if choice.value != "A:HOH1072"
        ],
    }

    data = build_manifest(plan, answers)

    cutoffs = {site["id"]: site["mcpb"]["cutoff_angstrom"] for site in data["metals"]}
    # A's water is 2.24 A from its iron and B's 2.28 A from its own.
    assert set(cutoffs.values()) == {2.19}
    ManifestConfig.model_validate(data)


def test_a_side_chain_near_one_copy_of_a_site_is_asked_about_in_every_copy(tmp_path, deposition):
    """Each titratable residue near a metal is asked its protonation. One near
    one copy's iron only gave the copies different questions, which they
    cannot share -- the same refusal, one radius further out."""

    from mdprep.config.decisions import answer_conflicts

    structure = slid(deposition, ("B", 899), ("B", 254, "NH1"), 2.95)
    plan = plan_decisions(written(structure, tmp_path / "m.pdb"))

    copy = plan.by_id("metal.b_fe2_899_site.donor_state.b_arg_254")
    assert copy.copy_of == "metal.a_fe2_900_site.donor_state.a_arg_254"
    answers = {**SHARED, "metal.a_fe2_900_site.donor_state.a_arg_254": "ARG"}
    assert answer_conflicts(plan, answers) is None
    data = build_manifest(plan, answers)
    assert next(site for site in data["metals"] if site["id"] == "b_fe2_899_site")[
        "parameters_from"
    ] == "a_fe2_900_site"
