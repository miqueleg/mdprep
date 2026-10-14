"""The requirement check behind `mdprep doctor`."""

import pytest

from mdprep.workflows.doctor import (
    REQUIREMENTS,
    Requirement,
    ash_note,
    check_requirement,
    required_packages,
    run_doctor,
)


def _manifest(**overrides) -> dict:
    data = {
        "project": {"name": "t", "input_structure": "t.pdb", "output_dir": "out"},
        "protein": {"forcefield": "ff14SB", "water_model": "TIP3P"},
        "protonation": {"method": "manual_only"},
        "ligands": [],
        "metals": [],
    }
    data.update(overrides)
    return data


def test_a_minimal_manifest_needs_only_ambertools_and_the_always_on_checks():
    packages = required_packages(_manifest())

    assert "ambertools" in packages
    # Nothing has asked for a quantum chemistry package.
    assert "pyscf" not in packages
    assert "rdkit" not in packages


def test_a_smiles_ligand_needs_rdkit():
    packages = required_packages(
        _manifest(ligands=[{"charge_method": "am1bcc", "smiles": "CCO"}])
    )

    assert "rdkit" in packages


def test_the_embedded_histidine_comparison_needs_openmm_and_xtb():
    """Its MM point charges are read out of OpenMM's force field files."""

    packages = required_packages(
        _manifest(
            protonation={
                "method": "propka_xtb_his",
                "histidine": {"xtb": {"embedding": "qmmm"}},
            }
        )
    )

    assert {"propka", "xtb", "openmm"} <= set(packages)


def test_the_cluster_histidine_comparison_does_not_need_openmm():
    """The OpenMM energy check is on by default and would mask this, so it is
    turned off here to isolate what the tautomer comparison itself needs."""

    without_check = {"run_openmm_energy_check": False}
    cluster = required_packages(
        _manifest(
            validation=without_check,
            protonation={
                "method": "propka_xtb_his",
                "histidine": {"xtb": {"embedding": "cluster"}},
            },
        )
    )
    embedded = required_packages(
        _manifest(
            validation=without_check,
            protonation={
                "method": "propka_xtb_his",
                "histidine": {"xtb": {"embedding": "qmmm"}},
            },
        )
    )

    assert "xtb" in cluster and "openmm" not in cluster
    assert "openmm" in embedded


def test_every_requirement_names_a_real_conda_package_and_a_purpose():
    for package, requirements in REQUIREMENTS.items():
        for requirement in requirements:
            assert requirement.conda_package == package
            assert requirement.purpose
            assert requirement.kind in {"python", "program"}


def test_a_missing_requirement_is_reported_with_an_install_command():
    missing = Requirement(
        name="nonexistent-tool",
        kind="program",
        conda_package="nonexistent",
        purpose="a thing that is not installed",
        probe="mdprep-no-such-program",
    )
    finding = check_requirement(missing)

    assert not finding.found
    assert finding.detail == "not on PATH"
    assert missing.install_hint == "conda install -c conda-forge nonexistent"


def test_a_present_python_package_reports_its_version():
    finding = check_requirement(
        Requirement("pydantic", "python", "pydantic", "manifest validation", "pydantic")
    )

    assert finding.found
    assert finding.detail[0].isdigit()


def test_the_report_is_ok_only_when_nothing_is_missing():
    report = run_doctor(_manifest())

    assert report.ok == (not report.missing)
    assert (report.install_command is None) == report.ok


def test_ash_is_called_out_because_no_command_here_can_install_it(monkeypatch):
    """`pip install ash` fetches unrelated software, so the note must say so."""

    import mdprep.workflows.doctor as doctor

    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: None)
    note = ash_note(
        _manifest(refinement={"enabled": True, "backend": "ash"})
    )

    assert note is not None
    assert "github.com/RagnarB83/ash" in note
    assert "pip install ash" in note


def test_the_default_refinement_backend_raises_no_ash_note():
    assert ash_note(_manifest(refinement={"enabled": True})) is None
    assert ash_note(_manifest(refinement={"enabled": False})) is None
    assert ash_note(None) is None


def _metal_manifest(**pyscf) -> dict:
    settings = {"geometry_source": "qmmm_refinement", "method": "B3LYP", "basis": "6-31G*"}
    settings.update(pyscf)
    return _manifest(
        metals=[{"id": "site", "model": "bonded_mcpb", "mcpb": {"pyscf": settings}}]
    )


def test_a_metal_site_without_a_gxtb_build_is_reported(monkeypatch):
    """A g-xTB Hessian is the default for a metal site, and the binary is a
    separate download, so this is the failure users will hit most."""

    import mdprep.workflows.doctor as doctor
    from mdprep.metals.gxtb_hessian import GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE

    monkeypatch.delenv(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)

    note = doctor.gxtb_note(_metal_manifest())

    assert note is not None
    assert "github.com/grimme-lab/g-xtb" in note
    assert "model: gfn2" in note


def test_an_xtb_without_gxtb_support_is_not_mistaken_for_one(monkeypatch):
    """conda's xtb is GFN-only. Accepting it on the strength of its name
    would make this check pass everywhere and help nobody."""

    import mdprep.workflows.doctor as doctor
    from mdprep.metals.gxtb_hessian import GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE

    monkeypatch.delenv(GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/conda/bin/xtb")
    monkeypatch.setattr(doctor, "supports_gxtb", lambda executable: False)

    note = doctor.gxtb_note(_metal_manifest())

    assert note is not None
    assert "/conda/bin/xtb" in note
    assert "no --gxtb support" in note


def test_a_real_gxtb_build_raises_no_note(monkeypatch):
    import mdprep.workflows.doctor as doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/opt/g-xtb/bin/xtb")
    monkeypatch.setattr(doctor, "supports_gxtb", lambda executable: True)

    assert doctor.gxtb_note(_metal_manifest()) is None


def test_a_site_that_asked_for_gfn2_or_dft_raises_no_gxtb_note(monkeypatch):
    import mdprep.workflows.doctor as doctor

    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)

    assert doctor.gxtb_note(_metal_manifest(xtb={"model": "gfn2"})) is None
    assert doctor.gxtb_note(_metal_manifest(hessian_backend="pyscf")) is None
    assert doctor.gxtb_note(_manifest()) is None
