"""Non-metal ions bound in a metal site.

A coordinating halide belongs nowhere else in the manifest. It is not a metal,
so ions[] would have MCPB treat it as a metal centre; and it cannot be a GAFF
ligand, because antechamber's semi-empirical step refuses a single atom. Amber
parameterises F-, Cl-, Br- and I- directly, which is what the force-field
relaxation uses.

A polyatomic coordinating anion such as azide needs none of this: antechamber
handles it, so it is a cofactor under ligands[] named as an mcpb coordinator,
exactly as alpha-ketoglutarate already is.
"""

import pytest

from mdprep.config.models import CoordinatingIonConfig, ManifestConfig
from mdprep.metals.nonbonded import NonbondedMetalError, ion_atom_type, ion_frcmod_name


# --------------------------------------------------------------------------
# Amber's atom-type spelling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("element", "charge", "expected"),
    [
        ("Na", 1, "Na+"),      # magnitude omitted when it is one
        ("K", 1, "K+"),
        ("Zn", 2, "Zn2+"),
        ("Fe", 3, "Fe3+"),
        ("Cl", -1, "Cl-"),     # and the sign follows the magnitude
        ("Br", -1, "Br-"),
        ("I", -1, "I-"),
        ("F", -1, "F-"),
    ],
)
def test_ion_atom_types_match_ambers_spelling(element, charge, expected):
    """Building it as element+charge+"+" gave "Na1+", which matches nothing."""
    assert ion_atom_type(element, charge) == expected


def test_a_zero_charge_ion_is_rejected():
    with pytest.raises(NonbondedMetalError, match="zero charge"):
        ion_atom_type("Cl", 0)


def test_monovalent_ions_of_either_sign_share_a_parameter_file():
    """frcmod.ions1lm_* carries Li+/Na+/K+ alongside F-/Cl-/Br-/I-."""
    cation = ion_frcmod_name(charge=1, water_model="TIP3P", parameter_set="12_6")
    anion = ion_frcmod_name(charge=-1, water_model="TIP3P", parameter_set="12_6")

    assert cation == anion == "frcmod.ions1lm_126_tip3p"


def test_a_charge_with_no_li_merz_parameters_fails_clearly():
    with pytest.raises(NonbondedMetalError, match="not among them"):
        ion_frcmod_name(charge=-2, water_model="TIP3P", parameter_set="12_6")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def halide(**overrides) -> dict:
    base = {
        "selector": {
            "chain": "A", "resname": "CL", "resid": 303,
            "icode": None, "atom_name": "CL",
        },
        "element": "Cl",
        "charge": -1,
    }
    base.update(overrides)
    return base


def test_a_coordinating_halide_is_accepted():
    ion = CoordinatingIonConfig.model_validate(halide())

    assert ion.element == "Cl"
    assert ion.charge == -1
    assert ion.parameter_set == "12_6"


def test_a_cation_is_rejected_because_it_belongs_in_ions():
    """A coordinating cation is a metal and MCPB must see it as one."""
    with pytest.raises(ValueError):
        CoordinatingIonConfig.model_validate(halide(element="Zn", charge=2))


def manifest_with_site(**site_overrides) -> dict:
    site = {
        "id": "fe",
        "model": "nonbonded",
        "ions": [
            {
                "selector": {
                    "chain": "A", "resname": "FE2", "resid": 302,
                    "icode": None, "atom_name": "FE",
                },
                "element": "Fe",
                "charge": 2,
                "multiplicity": 5,
            }
        ],
        "nonbonded": {"parameter_set": "12_6"},
    }
    site.update(site_overrides)
    return {
        "project": {"name": "t", "input_structure": "in.pdb", "output_dir": "out"},
        "structure": {},
        "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
        "protonation": {"method": "manual_only"},
        "disulfides": {},
        "ligands": [],
        "metals": [site],
        "solvation": {},
        "validation": {},
    }


def test_a_site_may_carry_both_a_metal_and_a_halide():
    manifest = ManifestConfig.model_validate(
        manifest_with_site(coordinating_ions=[halide()])
    )

    site = manifest.metals[0]
    assert site.ions[0].element == "Fe"
    assert site.coordinating_ions[0].element == "Cl"


def test_sites_carry_no_coordinating_ions_by_default():
    manifest = ManifestConfig.model_validate(manifest_with_site())

    assert manifest.metals[0].coordinating_ions == []


# --------------------------------------------------------------------------
# What the rest of the pipeline makes of it
# --------------------------------------------------------------------------


def test_the_besd_example_declares_its_chloride_as_an_ion():
    import yaml
    from pathlib import Path

    example = yaml.safe_load(
        Path("examples/16_besd_halogenase_gfn2_mcpb.yaml").read_text(encoding="utf-8")
    )

    site = example["metals"][0]
    assert site["coordinating_ions"][0]["element"] == "Cl"
    assert site["coordinating_ions"][0]["charge"] == -1
    # And it is not also a GAFF ligand, which antechamber could not handle.
    assert all(entry["id"] != "a_cl_303" for entry in example["ligands"])


def test_a_coordinating_ion_contributes_its_charge_to_an_xtb_cluster():
    """Otherwise the cluster is short an electron and xTB refuses it."""
    from mdprep.protonation.histidine_xtb import _metal_charge_by_residue
    from mdprep.structure.pdb import read_pdb

    structure = read_pdb("examples/tutorials/besd_6nie/6NIE.pdb")
    from mdprep.config.loader import load_manifest

    manifest = load_manifest(
        "examples/16_besd_halogenase_gfn2_mcpb.yaml", resolve_paths=True
    )
    charges = _metal_charge_by_residue(list(structure.residues), manifest)

    by_name = {}
    for residue in structure.residues:
        if id(residue) in charges:
            by_name[residue.id.resname] = charges[id(residue)]
    assert by_name.get("FE2") == 2
    assert by_name.get("CL") == -1


def test_a_coordinating_ion_is_not_an_unknown_heterogen():
    """Normalization must keep it rather than demand a ligands[] entry."""
    from mdprep.config.loader import load_manifest
    from mdprep.structure.normalize import _mcpb_model_residue_keys
    from mdprep.structure.pdb import read_pdb

    structure = read_pdb("examples/tutorials/besd_6nie/6NIE.pdb")
    manifest = load_manifest(
        "examples/16_besd_halogenase_gfn2_mcpb.yaml", resolve_paths=True
    )

    configured = _mcpb_model_residue_keys(structure, manifest)

    chlorides = [
        residue
        for residue in structure.residues
        if residue.id.resname == "CL" and residue.id.chain_id == "A"
    ]
    assert chlorides
    assert any(id(residue) in configured for residue in chlorides)
