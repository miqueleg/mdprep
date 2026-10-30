"""Noncanonical residues kept in their chain, without AmberTools.

The reference is SpoIIAA phosphorylated on Ser57 (PDB 1H4X): the fixture is
chain B, residues 52-62, with SEP57 between Asp56 and Ser58, declared by
MODRES and bonded by LINKs, one of them a short 1.23 A. Everything here runs
without AmberTools; what needs RDKit is skipped without it. The builds
themselves are in test_external_noncanonical_residues.py.
"""

from __future__ import annotations

from math import dist
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mdprep.config.decisions import (
    DecisionError,
    DecisionPlan,
    Decision,
    Choice,
    answer_conflicts,
    build_manifest,
    plan_decisions,
    proposal_explanations,
    proposed_answers,
)
from mdprep.config.loader import load_manifest
from mdprep.config.models import ManifestConfig
from mdprep.leap.builder import TLeapOutputs, build_tleap_script
from mdprep.leap.forcefields import ResidueLibraryFiles, forcefield_sources
from mdprep.residues.capping import ResidueCappingError, chain_context, residue_hydrogen_names
from mdprep.residues.charges import BACKBONE_CHARGES, constrain_to_backbone
from mdprep.residues.embedding import _aligned
from mdprep.residues.frcmod import cmap_section, junction_parameters, parent_cmap
from mdprep.residues.parameterise import library_script
from mdprep.residues.templates import (
    AMBER_RESIDUE_TEMPLATES,
    KNOWN_RESIDUE_CHEMISTRY,
    proposed_template,
    templates_for,
)
from mdprep.residues.workflow import ResidueLeapInputs, ResidueStageError, residue_leap_inputs
from mdprep.structure.hydrogens import is_hydrogen
from mdprep.structure.normalize import StructureNormalizationError, normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.writer import render_pdb
from tests.test_config_validation import base_manifest
from tests.test_heterogen_groups import unknown_methionine_like, walk

ROOT = Path(__file__).resolve().parents[1]
HETEROGENS = ROOT / "tests" / "data" / "heterogens"
PHOSPHOSERINE = HETEROGENS / "phosphoserine_1h4x.pdb"
SELENOMETHIONINE = HETEROGENS / "selenomethionine_1rqr.pdb"
SEP_CAPPED = "CC(=O)NC(COP(=O)([O-])[O-])C(=O)NC"
SEP_FREE = "N[C@@H](COP(=O)([O-])[O-])C(=O)O"
SEP57 = {"chain": "B", "resname": "SEP", "resid": 57}


def _manifest(blocks, *, forcefield="ff14SB", structure=PHOSPHOSERINE, **extra) -> dict:
    data = base_manifest()
    data["project"]["input_structure"] = str(structure)
    data["protein"]["forcefield"] = forcefield
    data["protonation"] = {"ph": 7.0, "method": "manual_only", "overrides": []}
    data["noncanonical_residues"] = blocks
    data["structure"].update(extra)
    return data


def _smiles_block(**changes) -> dict:
    return {
        "resname": "SEP",
        "residues": [SEP57],
        "parameters": "am1bcc",
        "smiles": SEP_CAPPED,
        "net_charge": -2,
        "parent": "SER",
        **changes,
    }


def _template_block(template="SEP") -> dict:
    return {"resname": "SEP", "residues": [SEP57], "parameters": "amber_library", "template": template}


# --------------------------------------------------------------------------
# The manifest block
# --------------------------------------------------------------------------


def test_either_block_validates():
    ManifestConfig.model_validate(_manifest([_smiles_block()]))
    ManifestConfig.model_validate(_manifest([_template_block("S1P")]))


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (_smiles_block(residues=[{**SEP57, "resname": "TPO"}]), "every residue of a block is a SEP"),
        (_smiles_block(residues=[SEP57, SEP57]), "more than once"),
        (_smiles_block(net_charge=None), "requires smiles and net_charge"),
        (_smiles_block(template="SEP"), "template belongs to parameters: amber_library"),
        (_smiles_block(resname="SER", residues=[{**SEP57, "resname": "SER"}]), "already define"),
        ({**_template_block(), "template": "TPO"}, "builds TPO, not SEP"),
        ({**_template_block(), "template": "XYZ"}, "not one mdprep knows"),
        ({**_template_block(), "net_charge": -2}, "a template brings its own chemistry"),
        (_smiles_block(parent="SEP"), "not one of the twenty"),
    ],
)
def test_a_block_that_says_two_things_is_refused(block, message):
    with pytest.raises(ValidationError, match=message):
        ManifestConfig.model_validate(_manifest([block]))


def test_ff19sb_needs_the_parent_whose_cmap_a_parameterised_residue_takes():
    with pytest.raises(ValidationError, match="needs a parent with ff19SB"):
        ManifestConfig.model_validate(_manifest([_smiles_block(parent=None)], forcefield="ff19SB"))
    ManifestConfig.model_validate(_manifest([_smiles_block()], forcefield="ff19SB"))
    # A template carries the CMAP AmberTools gave it.
    ManifestConfig.model_validate(_manifest([_template_block()], forcefield="ff19SB"))


def test_one_block_per_residue_name():
    with pytest.raises(ValidationError, match="more than one block for SEP"):
        ManifestConfig.model_validate(_manifest([_smiles_block(), _template_block()]))


@pytest.mark.parametrize(
    "claim",
    [
        {"remove_residues": [{"selector": SEP57, "reason": "dropped"}]},
        {
            "repair": {
                "backend": "pdbfixer",
                "replace_modified_residues": [{"selector": SEP57, "parent": "SER"}],
            }
        },
    ],
)
def test_a_kept_residue_is_not_also_removed_or_replaced(claim):
    with pytest.raises(ValidationError, match="also acts on"):
        ManifestConfig.model_validate(_manifest([_template_block()], **claim))


def test_a_kept_residue_takes_no_protonation_override():
    data = _manifest([_template_block()])
    data["protonation"]["overrides"] = [{"selector": SEP57, "state": "ASP", "reason": "no"}]
    with pytest.raises(ValidationError, match="protonation.overrides"):
        ManifestConfig.model_validate(data)


def _with_kept_residue(example: str) -> dict:
    data = load_manifest(ROOT / "examples" / example).model_dump(mode="json")
    data["noncanonical_residues"] = [_template_block()]
    return data


def test_a_bonded_metal_site_and_a_kept_residue_are_built_together():
    ManifestConfig.model_validate(_with_kept_residue("17_syrb2_bound_water_mcpb.yaml"))
    ManifestConfig.model_validate(_with_kept_residue("10_qmmm_refinement_ash.yaml"))


def test_qmmesp_charges_for_a_residue_on_a_bonded_metal_are_refused():
    """MCPB.py refits the charges of every residue on the metal."""

    data = load_manifest(ROOT / "examples" / "17_syrb2_bound_water_mcpb.yaml").model_dump(mode="json")
    data["noncanonical_residues"] = [
        {
            "resname": "KCX",
            # Where the site's His116 is: a residue the fit refits.
            "residues": [{"chain": "A", "resname": "KCX", "resid": 116}],
            "parameters": "qmmesp_pyscf",
            "smiles": KNOWN_RESIDUE_CHEMISTRY["KCX"].smiles,
            "net_charge": -1,
            "parent": "LYS",
            "qmmesp": {"method": "HF", "basis": "6-31G*"},
        }
    ]

    with pytest.raises(ValidationError, match="refits the charges of every residue on the metal"):
        ManifestConfig.model_validate(data)


# --------------------------------------------------------------------------
# Templates, and the chemistry mdprep proposes
# --------------------------------------------------------------------------


def test_the_phosphoserine_template_has_the_deposited_residues_atoms():
    sep = next(r for r in read_pdb(PHOSPHOSERINE).residues if r.id.resname == "SEP")
    heavy = {atom.name.strip() for atom in sep.atoms if not is_hydrogen(atom)}

    assert [template.name for template in templates_for("SEP")] == ["SEP", "S1P"]
    for template in templates_for("SEP"):
        assert template.heavy_atoms == heavy


def test_a_template_name_is_a_protonation_state():
    assert {name: AMBER_RESIDUE_TEMPLATES[name].net_charge for name in ("SEP", "S1P", "TPO", "T1P", "PTR")} == {
        "SEP": -2, "S1P": -1, "TPO": -2, "T1P": -1, "PTR": -2,
    }
    assert AMBER_RESIDUE_TEMPLATES["SEP"].leaprc("ff19SB") == "leaprc.phosaa19SB"
    assert AMBER_RESIDUE_TEMPLATES["HYP"].leaprc("ff14SB") is None


def test_every_proposed_smiles_has_the_atoms_and_charge_it_claims():
    Chem = pytest.importorskip("rdkit.Chem")
    caps = {"C": 3, "N": 1, "O": 1}
    for entry in KNOWN_RESIDUE_CHEMISTRY.values():
        molecule = Chem.MolFromSmiles(entry.smiles)
        assert molecule is not None, entry.resname
        counts: dict[str, int] = {}
        for atom in molecule.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        for element, count in caps.items():
            counts[element] -= count
        assert {k: v for k, v in counts.items() if v} == dict(entry.heavy_atoms), entry.resname
        assert Chem.GetFormalCharge(molecule) == entry.net_charge, entry.resname


def test_the_dianion_is_proposed_for_a_phosphate_and_the_monoanion_named():
    name, why = proposed_template("SEP", ["SEP", "S1P"])

    assert name == "SEP"
    assert "choose S1P then" in why
    assert proposed_template("ALY", ["ALY"]) is None


# --------------------------------------------------------------------------
# Charges on the protein's backbone
# --------------------------------------------------------------------------


def _fitted(residue: dict[str, float], caps=(("ACE", "CY", 0.61), ("NME", "NT", -0.55))):
    return [(role, name, charge) for role, name, charge in caps] + [
        ("RES", name, charge) for name, charge in residue.items()
    ]


@pytest.mark.parametrize(("net", "which"), [(-2, "anionic"), (0, "neutral"), (1, "cationic")])
def test_the_backbone_takes_the_charges_of_a_residue_of_the_same_sign(net, which):
    residue = {"N": -0.5, "H": 0.3, "CA": 0.05, "HA": 0.1, "C": 0.6, "O": -0.6, "CB": 0.2, "OG": -0.5}

    charges = constrain_to_backbone(_fitted(residue), net)

    assert charges.backbone_set == which
    for name, value in BACKBONE_CHARGES[which].items():
        assert charges.charges[name] == value
    assert round(sum(charges.charges.values()), 6) == net
    # What is missing is spread evenly over the rest, which keep their differences.
    shift = charges.charges["CB"] - residue["CB"]
    assert charges.charges["OG"] - residue["OG"] == pytest.approx(shift, abs=1e-6)
    assert charges.cap_sums == {"ACE": 0.61, "NME": -0.55}


def test_the_charges_sum_to_the_integer_at_six_decimals():
    residue = {"N": -0.5, "H": 0.3, "CA": 1 / 3, "HA": 1 / 7, "C": 0.6, "O": -0.6, "CB": 2 / 9}

    charges = constrain_to_backbone(_fitted(residue), -1)

    micro = sum(round(value * 1_000_000) for value in charges.charges.values())
    assert micro == -1_000_000


# --------------------------------------------------------------------------
# The parameters joining a GAFF2 side chain to the protein's backbone
# --------------------------------------------------------------------------

PARMCHK2 = """Remark line goes here
MASS
CX 12.010        0.878               same as c3
os 16.000        0.465               same as os

BOND
C -N   317.46   1.397       same as  c- n, penalty score=  0.0
CT-os  282.27   1.427       same as c3-os, penalty score=  0.0

ANGLE
C -N -CX   77.240     118.830   same as c -n -c3, penalty score=  0.0
CX-CT-os   77.440     107.620   same as c3-c3-os, penalty score=  0.0

DIHE
C -CX-N -C    1    0.390       180.000          -2.000      same as c -c3-n -c
os-CT-CX-N    9    1.400         0.000           3.000      same as X -c3-c3-X , penalty score=  0.0

IMPROPER
CX-N -C -O          1.1          180.0         2.0          Using the default value

NONBON
  N           1.7852  0.1636             same as n
  os          1.6837  0.0726             same as os
"""


def test_only_the_terms_joining_gaff2_to_the_backbone_are_kept():
    """parmchk2 also writes GAFF2 guesses for the protein's own terms, the phi
    torsion C-N-CX-C among them, which would rewrite every residue's."""

    kept, dropped = junction_parameters(PARMCHK2)

    assert "CT-os" in kept and "CX-CT-os" in kept and "os-CT-CX-N" in kept
    for term in ("C -N  ", "C -N -CX", "C -CX-N -C", "CX-N -C -O", "CX 12.010", "  N    "):
        assert term not in kept
    assert sorted(dropped) == sorted(
        ["MASS CX", "BOND C-N", "ANGLE C-N-CX", "DIHE C-CX-N-C", "IMPROPER CX-N-C-O", "NONBON N"]
    )


def test_ff19sbs_alpha_carbon_type_is_given_back():
    kept, _ = junction_parameters(PARMCHK2, rename={"CX": "XC"})

    assert "XC-CT-os" in kept and "os-CT-XC-N" in kept and "CX" not in kept


CMAPS = """CMAP
%FLAG CMAP_COUNT   1
%FLAG CMAP_TITLE
Gly CMAP
%FLAG CMAP_RESLIST  1
GLY
%FLAG CMAP_RESOLUTION 2
%FLAG CMAP_PARAMETER
  1.00000   2.00000
  3.00000   4.00000
%FLAG CMAP_COUNT   2
%FLAG CMAP_TITLE
Ser CMAP
%FLAG CMAP_RESLIST  2
SER THR
%FLAG CMAP_RESOLUTION 2
%FLAG CMAP_PARAMETER
 -0.50000   0.25000
  0.75000  -1.00000
"""


def test_a_parameterised_residue_takes_its_parents_cmap_under_its_own_name():
    block = parent_cmap(CMAPS, "SER")
    section = cmap_section("SEP", "SER", block, "ff19SB")

    assert block.title == "Ser CMAP" and block.residues == ("SER", "THR")
    assert "%FLAG CMAP_RESLIST  1\nSEP\n" in section
    assert "-0.50000   0.25000\n  0.75000  -1.00000" in section
    with pytest.raises(ValueError, match="no CMAP for PHE"):
        parent_cmap(CMAPS, "PHE")


# --------------------------------------------------------------------------
# Cutting the residue out between its caps
# --------------------------------------------------------------------------


def _sep(structure=None):
    structure = structure or read_pdb(PHOSPHOSERINE)
    return structure, next(r for r in structure.residues if r.id.resname == "SEP")


def test_the_neighbours_are_found_by_their_peptide_bonds():
    structure, sep = _sep()
    context = chain_context(structure.residues, sep)

    assert (context.previous.id.display(), context.following.id.display()) == ("B:ASP56", "B:SER58")


def test_a_residue_that_ends_its_chain_is_not_capped():
    structure = read_pdb(PHOSPHOSERINE)
    first = structure.residues[0]

    with pytest.raises(ResidueCappingError, match="N-terminus of its chain"):
        chain_context(structure.residues, first)


@pytest.mark.parametrize(
    ("parent", "count", "names"),
    [
        ("N", 1, ["H"]),
        ("CA", 1, ["HA"]),
        ("CB", 2, ["HB2", "HB3"]),
        ("NZ", 3, ["HZ1", "HZ2", "HZ3"]),
        ("CG2", 3, ["HG21", "HG22", "HG23"]),
        ("O3P", 1, ["H3P"]),
    ],
)
def test_hydrogens_are_named_as_the_pdb_names_an_amino_acids(parent, count, names):
    assert residue_hydrogen_names(parent, count, set()) == names


def test_the_capped_and_the_free_amino_acid_are_one_molecule():
    pytest.importorskip("rdkit")
    from mdprep.residues.capping import capped_residue

    structure, sep = _sep()
    context = chain_context(structure.residues, sep)

    capped = capped_residue(context, SEP_CAPPED, net_charge=-2)
    free = capped_residue(context, SEP_FREE, net_charge=-2)
    zwitterion = capped_residue(context, "[NH3+]C(COP(=O)([O-])[O-])C([O-])=O", net_charge=-2)

    assert (capped.smiles_form, free.smiles_form) == ("capped", "free amino acid")
    placed = [(a.name, a.x, a.y, a.z) for a in capped.atoms if a.role == "RES" and a.element == "H"]
    assert [name for name, *_ in placed] == ["H", "HA", "HB2", "HB3"]
    for other in (free, zwitterion):
        assert [(a.name, a.x, a.y, a.z) for a in other.atoms if a.role == "RES" and a.element == "H"] == placed
    # The amide hydrogen sits on N, bonded to it.
    nitrogen = next(a for a in capped.atoms if a.role == "RES" and a.name == "N")
    hydrogen = next(a for a in capped.atoms if a.role == "RES" and a.name == "H")
    assert dist((nitrogen.x, nitrogen.y, nitrogen.z), (hydrogen.x, hydrogen.y, hydrogen.z)) < 1.1


@pytest.mark.parametrize(
    ("smiles", "net", "message"),
    [
        (SEP_CAPPED, -1, "formal charge of -2 but net_charge is -1"),
        ("CC(=O)NC(CO)C(=O)NC", 0, "alpha-amino-acid backbone"),
        ("CCCC", 0, "has 4 heavy atoms, and the residue has 10"),
        ("not a smiles", 0, "not a valid SMILES"),
    ],
)
def test_a_smiles_that_is_not_the_residue_is_refused(smiles, net, message):
    pytest.importorskip("rdkit")
    from mdprep.residues.capping import capped_residue

    structure, sep = _sep()
    with pytest.raises(ResidueCappingError, match=message):
        capped_residue(chain_context(structure.residues, sep), smiles, net_charge=net)


# --------------------------------------------------------------------------
# The structure the build sees
# --------------------------------------------------------------------------


def test_a_modified_residue_nothing_accounts_for_stops_the_build(tmp_path):
    """remove_unknown_heterogens used to cut it out of its chain."""

    data = _manifest([], remove_unknown_heterogens=True)
    data["project"]["output_dir"] = str(tmp_path / "out")

    with pytest.raises(StructureNormalizationError, match="B:SEP57 is a residue of a protein chain"):
        normalize_structure_stage(ManifestConfig.model_validate(data))


def test_a_kept_residue_is_written_as_a_residue_of_its_chain(tmp_path):
    pytest.importorskip("rdkit")
    data = _manifest([_smiles_block()])
    data["project"]["output_dir"] = str(tmp_path / "out")

    result = normalize_structure_stage(ManifestConfig.model_validate(data))

    _, sep = _sep(result.normalized_structure)
    assert sep.record_names == {"ATOM"}
    assert [atom.name for atom in sep.atoms if is_hydrogen(atom)] == ["H", "HA", "HB2", "HB3"]
    # As a HETATM after Asp56 it would have opened a new chain in tleap.
    assert "TER" not in render_pdb(result.normalized_structure).split("END")[0]
    (kept,) = result.to_report_dict()["noncanonical_residues_kept"]
    assert (kept["parameters"], kept["hydrogens_placed"], kept["renamed"]) == ("am1bcc", 4, False)


def test_a_template_renames_the_residue_and_says_so(tmp_path):
    data = _manifest([_template_block("S1P")])
    data["project"]["output_dir"] = str(tmp_path / "out")

    result = normalize_structure_stage(ManifestConfig.model_validate(data))

    names = {residue.id.display() for residue in result.normalized_structure.residues}
    assert "B:S1P57" in names and "B:SEP57" not in names
    (kept,) = result.noncanonical_residues_kept
    assert (kept["final_resname"], kept["renamed"], kept["hydrogens_placed"]) == ("S1P", True, 0)
    assert any("renamed S1P" in warning for warning in result.warnings)


def test_a_template_without_the_residues_atoms_is_refused(tmp_path):
    text = "".join(
        line + "\n" for line in PHOSPHOSERINE.read_text().splitlines() if " O3P SEP " not in line
    )
    structure = tmp_path / "no_o3p.pdb"
    structure.write_text(text)
    data = _manifest([_template_block()], structure=structure)
    data["project"]["output_dir"] = str(tmp_path / "out")

    with pytest.raises(StructureNormalizationError, match="it lacks O3P"):
        normalize_structure_stage(ManifestConfig.model_validate(data))


def test_protonation_keeps_the_hydrogens_of_the_smiles(tmp_path):
    pytest.importorskip("rdkit")
    from mdprep.protonation.apply import apply_protonation_stage

    data = _manifest([_smiles_block()])
    data["project"]["output_dir"] = str(tmp_path / "out")
    manifest = ManifestConfig.model_validate(data)
    normalized = normalize_structure_stage(manifest).normalized_structure

    result = apply_protonation_stage(
        normalized,
        manifest,
        input_normalized_pdb_path=tmp_path / "n.pdb",
        output_protonation_pdb_path=tmp_path / "p.pdb",
    )

    _, sep = _sep(result.structure)
    assert [atom.name for atom in sep.atoms if is_hydrogen(atom)] == ["H", "HA", "HB2", "HB3"]


# --------------------------------------------------------------------------
# What tleap is given
# --------------------------------------------------------------------------


def test_a_build_loads_the_residue_libraries_after_every_source(tmp_path):
    files = ResidueLibraryFiles(
        resname="SEP", library_path=tmp_path / "SEP.lib", frcmod_path=tmp_path / "SEP.frcmod"
    )
    sources = forcefield_sources(
        protein_forcefield="ff14SB",
        water_model="TIP3P",
        ligands=[],
        residues=ResidueLeapInputs(
            sources=("leaprc.phosaa14SB",), files=(files,), atom_types=("gaff2",)
        ),
    )
    script = build_tleap_script(
        sources=sources,
        ligands=[],
        input_pdb=tmp_path / "in.pdb",
        disulfide_bonds=[],
        outputs=TLeapOutputs(prmtop=tmp_path / "s.prmtop", inpcrd=tmp_path / "s.inpcrd", pdb=tmp_path / "s.pdb"),
    ).splitlines()

    assert script[:6] == [
        "source leaprc.protein.ff14SB",
        "source leaprc.phosaa14SB",
        "source leaprc.water.tip3p",
        "source leaprc.gaff2",
        f"loadoff {tmp_path / 'SEP.lib'}",
        f"loadamberparams {tmp_path / 'SEP.frcmod'}",
    ]
    assert script[6].startswith("system = loadpdb")


def test_the_library_is_bonded_head_to_tail_with_its_elements_stated():
    script = library_script(
        resname="SEP",
        protein_forcefield="ff19SB",
        atom_types="gaff2",
        mol2_name="SEP.mol2",
        frcmod_name="SEP.frcmod",
        library_name="SEP.lib",
        elements={"CA": "C", "N": "N"},
    ).splitlines()

    for line in (
        "source leaprc.protein.ff19SB",
        "set SEP.1.CA element C",
        "set SEP head SEP.1.N",
        "set SEP tail SEP.1.C",
        "set SEP.1 connect0 SEP.1.N",
        "set SEP.1 connect1 SEP.1.C",
        "set SEP.1 restype protein",
        "saveoff SEP SEP.lib",
    ):
        assert line in script


def test_a_build_without_the_residue_stage_is_refused(tmp_path):
    data = _manifest([_smiles_block()])
    data["project"]["output_dir"] = str(tmp_path / "out")
    manifest = ManifestConfig.model_validate(data)

    with pytest.raises(ResidueStageError, match="has not been parameterised"):
        residue_leap_inputs(manifest, tmp_path / "out")
    inputs = residue_leap_inputs(
        ManifestConfig.model_validate(_manifest([_template_block()])), tmp_path / "out"
    )
    assert inputs.sources == ("leaprc.phosaa14SB",) and inputs.files == ()


def test_a_templates_hydrogens_charge_goes_to_its_heavy_atom():
    """The embedding sees the residue before tleap has built its hydrogens."""

    from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord

    def atom(name):
        return AtomRecord(
            serial=None, name=name, altloc=None, resname="SEP", chain_id="B", resid=57,
            icode=None, x=0.0, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0, element=name[0],
            record_name="ATOM", original_line="",
        )

    residue = ResidueRecord(
        id=ResidueId(chain_id="B", resname="SEP", resid=57, icode=None),
        atoms=[atom("N"), atom("CA"), atom("C")],
        record_names={"ATOM"},
        original_index=0,
    )
    charges = {"N": -0.5, "H": 0.3, "CA": -0.2, "HA": 0.1, "C": 0.3}
    neighbours = {"N": {"H", "CA"}, "H": {"N"}, "CA": {"N", "HA", "C"}, "HA": {"CA"}, "C": {"CA"}}

    aligned = _aligned(residue, charges, neighbours)

    assert aligned == pytest.approx([-0.2, -0.1, 0.3])
    assert sum(aligned) == pytest.approx(sum(charges.values()))


# --------------------------------------------------------------------------
# The planner's question, and the manifest it writes
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phosphoserine():
    return plan_decisions(PHOSPHOSERINE)


def test_a_phosphoserine_is_one_question_with_four_answers(phosphoserine):
    decision = phosphoserine.by_id("residue.SEP.treatment")

    assert [choice.value for choice in decision.choices] == [
        "replace",
        "amber_library",
        "am1bcc",
        "qmmesp_pyscf",
    ]
    assert decision.default is None and decision.requires_user_input
    assert [target.resid for target in decision.targets] == [57]
    assert phosphoserine.findings["ligands"] == []
    assert phosphoserine.findings["noncanonical_residues"]["SEP"]["templates"] == ["SEP", "S1P"]


def _answers(plan, treatment, **given):
    return walk(
        plan,
        {"residue.SEP.treatment": treatment, "structure.repair.add_missing_heavy_atoms": "false", **given},
    )


@pytest.mark.parametrize(
    ("treatment", "expected"),
    [
        ("amber_library", {"resname": "SEP", "parameters": "amber_library", "template": "SEP"}),
        (
            "am1bcc",
            {
                "resname": "SEP",
                "parameters": "am1bcc",
                "smiles": KNOWN_RESIDUE_CHEMISTRY["SEP"].smiles,
                "net_charge": -2,
                "parent": "SER",
            },
        ),
        (
            "qmmesp_pyscf",
            {
                "resname": "SEP",
                "parameters": "qmmesp_pyscf",
                "smiles": KNOWN_RESIDUE_CHEMISTRY["SEP"].smiles,
                "net_charge": -2,
                "parent": "SER",
                "qmmesp": {"method": "HF", "basis": "6-31G*"},
            },
        ),
    ],
)
def test_keeping_it_writes_a_block_mdprep_accepts(phosphoserine, tmp_path, treatment, expected):
    manifest = build_manifest(phosphoserine, _answers(phosphoserine, treatment))

    (block,) = manifest["noncanonical_residues"]
    assert {key: value for key, value in block.items() if key != "residues"} == expected
    assert block["residues"] == [{**SEP57, "icode": None}]
    assert "repair" not in manifest["structure"]
    manifest["project"]["output_dir"] = str(tmp_path / "out")
    ManifestConfig.model_validate(manifest)


def test_replacing_it_writes_the_repair(phosphoserine):
    manifest = build_manifest(phosphoserine, _answers(phosphoserine, "replace"))

    assert "noncanonical_residues" not in manifest
    assert manifest["structure"]["repair"]["replace_modified_residues"] == [
        {"selector": {**SEP57, "icode": None}, "parent": "SER"}
    ]


def test_the_template_and_the_chemistry_are_proposed_and_explained(phosphoserine):
    library = proposed_answers(phosphoserine, {"residue.SEP.treatment": "amber_library"})
    smiles = proposal_explanations(phosphoserine, {"residue.SEP.treatment": "am1bcc"})

    assert library["residue.SEP.template"] == "SEP"
    assert proposed_answers(phosphoserine, {})["residue.SEP.net_charge"] == "-2"
    assert "second pKa" in smiles["residue.SEP.smiles"]


def test_a_residue_with_no_parent_cannot_take_ff19sbs_cmap(tmp_path):
    text = "".join(
        unknown_methionine_like(line) + "\n"
        for line in SELENOMETHIONINE.read_text().splitlines()
        if not line.startswith("MODRES")
    )
    path = tmp_path / "xyz.pdb"
    path.write_text(text)
    plan = plan_decisions(path)
    answers = walk(
        plan,
        {
            "residue.XYZ.treatment": "am1bcc",
            "residue.XYZ.net_charge": "0",
            "residue.XYZ.smiles": "CC(=O)NC(CCSC)C(=O)NC",
            "protein.forcefield": "ff19SB",
            "protein.water_model": "OPC",
            "structure.repair.add_missing_heavy_atoms": "false",
        },
    )

    with pytest.raises(DecisionError, match="CMAP") as raised:
        build_manifest(plan, answers)
    assert raised.value.decision_id == "protein.forcefield"


def test_the_question_follows_the_chains_kept(tmp_path):
    """Two copies: one question for both, asked while either is kept."""

    lines = PHOSPHOSERINE.read_text().splitlines()
    copy_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38]) + 40.0
            copy_lines.append(line[:21] + "C" + line[22:30] + f"{x:8.3f}" + line[38:])
        elif line.startswith("MODRES"):
            copy_lines.append(line[:16] + "C" + line[17:])
    text = "\n".join([*lines[:-1], *copy_lines, "END"]) + "\n"
    path = tmp_path / "two_copies.pdb"
    path.write_text(text)
    plan = plan_decisions(path)
    decision = plan.by_id("residue.SEP.treatment")

    assert [target.chain for target in decision.targets] == ["B", "C"]
    assert decision.when[0].values == ("all", "B", "C")
    manifest = build_manifest(
        plan, _answers(plan, "amber_library", **{"structure.chains": "C"})
    )
    assert manifest["noncanonical_residues"][0]["residues"] == [{**SEP57, "chain": "C", "icode": None}]


def test_a_kept_residue_away_from_a_bonded_metal_is_no_conflict():
    from mdprep.config.decisions import _require_buildable_residues

    plan = DecisionPlan(
        structure_path=PHOSPHOSERINE,
        decisions=(
            Decision(
                id="metal.a_zn_1.model",
                section="metals",
                question="Model?",
                kind="choice",
                why="",
                choices=(Choice("bonded_mcpb", "Bonded"),),
            ),
        ),
        findings={"noncanonical_residues": {"SEP": {"parent": "SER"}}},
    )

    for treatment in ("amber_library", "am1bcc", "qmmesp_pyscf"):
        _require_buildable_residues(
            plan, {"residue.SEP.treatment": treatment, "metal.a_zn_1.model": "bonded_mcpb"}
        )


def test_the_example_is_what_the_planner_writes():
    example = yaml.safe_load((ROOT / "examples" / "22_spoiiaa_phosphoserine.yaml").read_text())
    plan = plan_decisions(ROOT / "examples" / "tutorials" / "spoiiaa_1h4x" / "1H4X.pdb")
    manifest = build_manifest(
        plan,
        walk(
            plan,
            {
                "structure.chains": "B",
                "residue.SEP.treatment": "amber_library",
                "project.name": "spoiiaa_1h4x",
                "project.output_dir": "prepared/spoiiaa_1h4x",
            },
        ),
    )
    manifest["project"]["input_structure"] = example["project"]["input_structure"]

    assert manifest == example


def test_the_calcium_example_is_what_the_planner_writes():
    example = yaml.safe_load((ROOT / "examples" / "23_osteocalcin_gla_calcium.yaml").read_text())
    plan = plan_decisions(ROOT / "examples" / "tutorials" / "osteocalcin_1q8h" / "1Q8H.pdb")
    manifest = build_manifest(
        plan,
        walk(
            plan,
            {
                "residue.CGU.treatment": "am1bcc",
                "project.name": "osteocalcin_1q8h",
                "project.output_dir": "prepared/osteocalcin_1q8h",
            },
        ),
    )
    manifest["project"]["input_structure"] = example["project"]["input_structure"]

    assert manifest == example
    # Gla24 is on all three calcium ions, and is asked about once, as CGU.
    assert [site["model"] for site in manifest["metals"]] == ["nonbonded"] * 3
    for site in ("a_ca_71_site", "a_ca_72_site", "a_ca_73_site"):
        donors = plan.by_id(f"metal.{site}.confirm_donors")
        assert "A:CGU24" in [choice.value for choice in donors.choices]


def _bonded_gla(treatment: str, **given) -> tuple:
    plan = plan_decisions(ROOT / "examples" / "tutorials" / "osteocalcin_1q8h" / "1Q8H.pdb")
    answers = walk(
        plan,
        {
            "residue.CGU.treatment": treatment,
            "metal.a_ca_71_site.model": "bonded_mcpb",
            "metal.a_ca_71_site.confirm_donors": ["A:CGU24", "A:HOH87"],
            "project.name": "osteocalcin_1q8h",
            "project.output_dir": "prepared/osteocalcin_1q8h",
            **given,
        },
    )
    return plan, answers


@pytest.mark.parametrize(
    ("treatment", "given", "charge", "terms"),
    [
        # Kept, Gla brings the charge its SMILES has.
        ("am1bcc", {}, 0, ["CA +2", "CGU24 -2", "HOH87 +0"]),
        # Replaced, it is a glutamate on the metal, asked about as one.
        (
            "replace",
            {"metal.a_ca_71_site.donor_state.a_cgu_24": "GLU"},
            1,
            ["CA +2", "GLU24 -1", "HOH87 +0"],
        ),
    ],
)
def test_gla_on_a_bonded_calcium_counts_in_the_model_charge(treatment, given, charge, terms):
    from mdprep.config.decisions import proposed_model_charge

    plan, answers = _bonded_gla(treatment, **given)

    assert answer_conflicts(plan, answers) is None
    assert proposed_model_charge(plan, answers, "a_ca_71_site") == (charge, terms)


def test_qmmesp_charges_for_gla_on_a_bonded_calcium_are_refused_at_the_residue():
    plan, answers = _bonded_gla("qmmesp_pyscf")

    conflict = answer_conflicts(plan, answers)

    assert conflict is not None and "would never reach the build" in str(conflict)
    assert conflict.decision_id == "residue.CGU.treatment"


def test_a_bonded_site_takes_one_gaff_family_for_ligands_and_kept_residues():
    data = load_manifest(ROOT / "examples" / "17_syrb2_bound_water_mcpb.yaml").model_dump(mode="json")
    data["noncanonical_residues"] = [_smiles_block()]
    for ligand in data["ligands"]:
        ligand["atom_types"] = "gaff"

    with pytest.raises(ValidationError, match="one GAFF family per run"):
        ManifestConfig.model_validate(data)
