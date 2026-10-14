"""Tests for deriving a manifest's external toolchain.

The web builder generates a Colab setup cell from this, so an omission makes a
run fail halfway through and an over-estimate makes people wait on a large
conda solve for a package they never call.
"""

import pytest

from mdprep.config.decisions import build_manifest, plan_decisions
from mdprep.workflows.runtime import (
    conda_packages_for,
    micromamba_setup_commands,
)


def manifest(**overrides) -> dict:
    base = {
        "protonation": {"method": "manual_only"},
        "ligands": [],
        "metals": [],
        "validation": {"run_openmm_energy_check": True},
        "molecular_dynamics": {"enabled": False},
    }
    base.update(overrides)
    return base


def test_ambertools_is_always_required():
    """No supported workflow avoids tleap."""
    assert "ambertools" in conda_packages_for(manifest())


def test_a_plain_job_does_not_pull_in_quantum_chemistry():
    packages = conda_packages_for(
        manifest(ligands=[{"charge_method": "am1bcc"}])
    )

    assert "pyscf" not in packages
    assert "xtb" not in packages
    assert "propka" not in packages


def test_propka_is_added_only_when_a_propka_method_is_selected():
    assert "propka" not in conda_packages_for(manifest())
    assert "propka" in conda_packages_for(
        manifest(protonation={"method": "propka"})
    )


def test_xtb_is_added_only_for_the_xtb_histidine_method():
    assert "xtb" not in conda_packages_for(manifest(protonation={"method": "propka"}))

    packages = conda_packages_for(manifest(protonation={"method": "propka_xtb_his"}))

    assert "xtb" in packages
    assert "propka" in packages


@pytest.mark.parametrize(
    "charge_method", ["gas_resp_pyscf", "qmmesp_pyscf", "mcpb_resp_pyscf"]
)
def test_qm_charge_methods_require_pyscf(charge_method):
    packages = conda_packages_for(manifest(ligands=[{"charge_method": charge_method}]))

    assert "pyscf" in packages


def test_a_bonded_mcpb_site_requires_pyscf():
    packages = conda_packages_for(manifest(metals=[{"model": "bonded_mcpb"}]))

    assert "pyscf" in packages


def test_a_nonbonded_metal_does_not_require_pyscf():
    assert "pyscf" not in conda_packages_for(manifest(metals=[{"model": "nonbonded"}]))


def test_openmm_follows_the_energy_check_and_the_md_stage():
    without = conda_packages_for(
        manifest(validation={"run_openmm_energy_check": False})
    )
    with_md = conda_packages_for(
        manifest(
            validation={"run_openmm_energy_check": False},
            molecular_dynamics={"enabled": True},
        )
    )

    assert "openmm" not in without
    assert "openmm" in with_md


def test_parmed_is_always_required_because_validation_always_runs():
    assert "parmed" in conda_packages_for(
        manifest(validation={"run_openmm_energy_check": False})
    )


def test_package_order_is_stable():
    first = conda_packages_for(manifest(protonation={"method": "propka_xtb_his"}))
    second = conda_packages_for(manifest(protonation={"method": "propka_xtb_his"}))

    assert first == second
    assert first[0] == "ambertools"


def test_no_package_is_listed_twice():
    packages = conda_packages_for(
        manifest(
            ligands=[{"charge_method": "qmmesp_pyscf"}, {"charge_method": "gas_resp_pyscf"}],
            metals=[{"model": "bonded_mcpb"}],
            molecular_dynamics={"enabled": True},
        )
    )

    assert len(packages) == len(set(packages))


def test_it_works_on_a_manifest_the_builder_actually_produces():
    plan = plan_decisions("tests/data/protein_two_ligands.pdb")
    built = build_manifest(
        plan,
        {
            "structure.remove_unknown_heterogens": False,
            "structure.repair.add_missing_heavy_atoms": False,
            "ligand.b_sub_501.net_charge": 0,
            "ligand.b_sub_501.smiles": "C",
            "ligand.c_cof_601.net_charge": 0,
            "ligand.c_cof_601.smiles": "C",
        },
    )

    packages = conda_packages_for(built)

    assert packages[0] == "ambertools"
    assert "pyscf" not in packages  # both ligands default to AM1-BCC


def test_setup_commands_install_micromamba_and_then_mdprep():
    commands = micromamba_setup_commands(manifest(), mdprep_spec="mdprep==9.9.9")
    joined = "\n".join(commands)

    assert "micromamba" in joined
    assert "-c conda-forge" in joined
    assert "ambertools" in joined
    assert "pip install -q mdprep==9.9.9" in joined
    # The environment is created before anything is installed into it.
    assert joined.index("micromamba create") < joined.index("pip install")


def test_setup_commands_do_not_disturb_the_notebook_python():
    """condacolab replaces the kernel's interpreter; micromamba must not."""
    joined = "\n".join(micromamba_setup_commands(manifest()))

    assert "condacolab" not in joined
    assert "MAMBA_ROOT_PREFIX" in joined


def test_structure_repair_requires_pdbfixer():
    packages = conda_packages_for(
        manifest(structure={"repair": {"backend": "pdbfixer", "add_missing_heavy_atoms": True}})
    )

    assert "pdbfixer" in packages
    assert "openmm" in packages


def test_no_repair_means_no_pdbfixer():
    assert "pdbfixer" not in conda_packages_for(
        manifest(structure={"repair": {"backend": "none"}})
    )


def test_a_gfn2_qmmm_refinement_requires_xtb_and_openmm():
    packages = conda_packages_for(
        manifest(
            refinement={
                "enabled": True,
                "backend": "ash",
                "qm_method": "gfn2_xtb",
            }
        )
    )

    assert "xtb" in packages
    assert "openmm" in packages


def test_the_force_field_refinement_needs_no_quantum_engine():
    """qm_method is only read by the ASH backend, which is no longer the
    default, so it must not pull xtb into an environment that never runs it."""

    packages = conda_packages_for(
        manifest(refinement={"enabled": True, "qm_method": "gfn2_xtb"})
    )

    assert "openmm" in packages
    assert "xtb" not in packages


def test_a_disabled_refinement_adds_nothing():
    packages = conda_packages_for(manifest(refinement={"enabled": False}))

    assert "xtb" not in packages


def test_a_ligand_with_a_smiles_requires_rdkit():
    packages = conda_packages_for(
        manifest(ligands=[{"charge_method": "am1bcc", "smiles": "CCO"}])
    )

    assert "rdkit" in packages


def test_a_ligand_without_a_smiles_does_not():
    assert "rdkit" not in conda_packages_for(
        manifest(ligands=[{"charge_method": "am1bcc"}])
    )
