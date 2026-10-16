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


# --------------------------------------------------------------------------
# Metal sites in Colab
#
# A bonded metal site defaults to an xTB Hessian, and to the g-xTB model in
# particular. conda-forge carries only the GFN-only build, so a setup cell
# derived from the conda packages alone produces an environment that gets all
# the way to the Hessian before failing.
# --------------------------------------------------------------------------


def mcpb_site(**pyscf_overrides) -> dict:
    pyscf = {"hessian_backend": "xtb", "esp_backend": "xtb", "xtb": {"model": "gxtb"}}
    pyscf.update(pyscf_overrides)
    return {"model": "bonded_mcpb", "mcpb": {"pyscf": pyscf}}


def test_a_metal_site_with_a_gfn2_hessian_installs_conda_xtb():
    """Without this the run reaches the Hessian and finds no binary."""

    site = mcpb_site(xtb={"model": "gfn2"})
    assert "xtb" in conda_packages_for(manifest(metals=[site]))


def test_a_gxtb_site_installs_one_xtb_rather_than_two():
    """The g-xTB release is an ordinary xtb 6.7.1 with the g-xTB method
    added, and gives GFN2 energies identical to the conda build, so it
    serves the histidine tautomers and the ESP as well."""

    from mdprep.workflows.runtime import gxtb_bin_directory

    one = manifest(metals=[mcpb_site()])

    assert "xtb" not in conda_packages_for(one)
    setup = "\n".join(micromamba_setup_commands(one))
    assert f"export PATH={gxtb_bin_directory()}:$PATH" in setup


def test_a_metal_site_with_a_pyscf_hessian_does_not_need_xtb():
    packages = conda_packages_for(
        manifest(metals=[mcpb_site(hessian_backend="pyscf", esp_backend="pyscf", xtb=None)])
    )

    assert "xtb" not in packages


def test_the_gxtb_build_is_installed_because_conda_does_not_carry_it():
    from mdprep.workflows.runtime import (
        GXTB_SHA256,
        gxtb_executable_path,
        manifest_wants_gxtb,
    )

    one = manifest(metals=[mcpb_site()])
    assert manifest_wants_gxtb(one)[0]

    commands = micromamba_setup_commands(one)
    joined = "\n".join(commands)
    assert "grimme-lab/g-xtb" in joined
    # Verified before it is unpacked: this binary computes the force
    # constants, so a truncated download must stop the run rather than
    # quietly change the parameters.
    assert GXTB_SHA256 in joined
    assert joined.index(GXTB_SHA256) < joined.index("tar -xJf")
    assert gxtb_executable_path() in joined


def test_a_gfn2_hessian_does_not_download_the_gxtb_build():
    """GFN2 is what conda installs, so nothing extra is needed."""

    from mdprep.workflows.runtime import manifest_wants_gxtb

    site = mcpb_site(xtb={"model": "gfn2"})
    assert not manifest_wants_gxtb(manifest(metals=[site]))[0]
    assert "grimme-lab" not in "\n".join(micromamba_setup_commands(manifest(metals=[site])))


def test_a_configured_gxtb_path_is_left_alone():
    """Someone who named their own build does not want ours downloaded."""

    from mdprep.workflows.runtime import manifest_wants_gxtb

    site = mcpb_site(xtb={"model": "gxtb", "executable": "/opt/gxtb/bin/xtb"})
    wants, configured = manifest_wants_gxtb(manifest(metals=[site]))

    assert wants and configured == "/opt/gxtb/bin/xtb"
    assert "grimme-lab" not in "\n".join(micromamba_setup_commands(manifest(metals=[site])))


def test_the_run_command_carries_gxtb_into_the_environment():
    """`micromamba run` starts a fresh environment, so anything the setup
    cell exported is gone by the time the run cell executes."""

    from mdprep.workflows.runtime import gxtb_executable_path, micromamba_run_command

    command = micromamba_run_command(manifest(metals=[mcpb_site()]), "system.yaml")

    assert f"MDPREP_GXTB_EXECUTABLE={gxtb_executable_path()}" in command
    assert command.index("MDPREP_GXTB_EXECUTABLE") < command.index("micromamba run")
    assert command.endswith("mdprep prepare system.yaml")


def test_the_run_command_stays_plain_when_no_metal_needs_gxtb():
    from mdprep.workflows.runtime import micromamba_run_command

    assert "GXTB" not in micromamba_run_command(manifest(), "system.yaml")


def test_the_setup_cell_can_actually_fail():
    """Without `set -e` the cell runs every line whatever the last one did.

    A 404 pipes "Not Found" into tar, a failed checksum is followed by the
    unpack anyway, and a dead conda solve still ends on a green tick. The
    checksum this cell advertises means nothing without this.
    """

    commands = micromamba_setup_commands(manifest(metals=[mcpb_site()]))

    assert commands[0] == "set -euo pipefail"
    # And curl has to treat an HTTP error as an error rather than writing the
    # error page into the file it was told to save.
    for line in commands:
        if line.startswith("curl "):
            flags = line.split("curl ", 1)[1].split(" ", 1)[0]
            assert flags.startswith("-") and "f" in flags, line


def test_the_manifest_the_builder_produces_installs_what_that_run_invokes():
    """The package list is derived from manifest settings, and three times
    now it has missed one: xtb for an MCPB Hessian, the g-xTB build, and
    PDBFixer for the hydrogenated environment the histidine comparison
    needs. Each omission surfaced as a run dying in Colab after the whole
    conda solve, naming a tool the setup cell was never told to install.

    This pins the defaults the web builder actually emits for a metal
    protein with histidines, which is the path students take.
    """

    from pathlib import Path

    from mdprep.config.decisions import (
        build_manifest,
        plan_decisions,
        proposed_model_charge,
    )

    structure = Path("tests/data/boltz_far_donor.cif")
    plan = plan_decisions(structure)
    answers = {
        "project.name": "coverage",
        "structure.remove_unknown_heterogens": False,
        "structure.repair.add_missing_heavy_atoms": True,
        "ligand.k_lg2_1.net_charge": 0,
        "ligand.k_lg2_1.smiles": "OCC=O",
        "metal.f_fe_1_site.charge": 2,
        "metal.f_fe_1_site.confirm_donors": True,
        "metal.f_fe_1_site.multiplicity": 5,
        "metal.f_fe_1_site.donor_state.a_asp_131": "ASP",
        "molecular_dynamics.enabled": False,
    }
    answers["metal.f_fe_1_site.model_charge"] = proposed_model_charge(
        plan, answers, "f_fe_1_site"
    )[0]
    packages = conda_packages_for(build_manifest(plan, answers))

    for needed, why in [
        ("ambertools", "tleap, antechamber, parmchk2, MCPB.py"),
        ("propka", "the propka_xtb_his protonation method"),
        ("rdkit", "placing ligand hydrogens from a SMILES"),
        ("pyscf", "the bonded MCPB workflow"),
        ("openmm", "MM point charges and the energy check"),
        ("pdbfixer", "the hydrogenated environment for the tautomer comparison"),
        ("parmed", "topology and charge validation"),
    ]:
        assert needed in packages, f"{needed} is missing, and this run needs it for {why}"
