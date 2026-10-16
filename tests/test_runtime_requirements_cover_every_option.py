"""Whatever a user chooses, the setup cell must install what that choice runs.

`conda_packages_for` turns manifest settings into a package list, and three
times it has missed one: xtb for an MCPB Hessian, the g-xTB build, and
PDBFixer for the environment the histidine comparison hydrogenates. Each
surfaced the same way -- a run dying in Colab, after the whole conda solve,
naming a tool the cell was never told to install -- and each was fixed for
the one combination that had been reported.

That is the wrong shape of fix, because the resolver is wrong per *option*,
not per structure. The gaps survived because the manifests anyone tested
with happened to pull the dependency in for another reason: PDBFixer always
arrived through structure repair, so no run with repair enabled could show
the hole.

So this walks the combinations instead. The expectations below are written
from the guards in the source -- the places that refuse when a tool is
missing -- and deliberately not from the resolver, because a test that asks
the resolver what the resolver thinks proves nothing. Two statements of the
same fact, derived apart, and the test is that they agree.

Each entry cites the guard it came from. When one moves, this fails, and the
honest fix is to re-read that guard rather than to relax the expectation.
"""

from itertools import product
from typing import Any, Callable

import pytest

from mdprep.workflows.runtime import conda_packages_for


# --------------------------------------------------------------------------
# The oracle: what each guard in the source demands, and when it fires.
# --------------------------------------------------------------------------


def _histidine_xtb(manifest: dict[str, Any]) -> dict[str, Any]:
    protonation = manifest.get("protonation") or {}
    return ((protonation.get("histidine") or {}).get("xtb")) or {}


def _mcpb_sites(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        site
        for site in (manifest.get("metals") or [])
        if site.get("model") == "bonded_mcpb"
    ]


def needs_ambertools(manifest: dict[str, Any]) -> bool:
    """ambertools/commands.py: every supported build goes through tleap."""

    return True


def needs_propka(manifest: dict[str, Any]) -> bool:
    """protonation/propka.py: the propka methods shell out to propka3."""

    return (manifest.get("protonation") or {}).get("method") in {
        "propka",
        "propka_xtb_his",
    }


def needs_xtb(manifest: dict[str, Any]) -> bool:
    """The tautomer comparison and any xTB-backed MCPB step call the binary.

    protonation/xtb_runner.py, metals/gxtb_hessian.py, qm/xtb_esp.py. A site
    asking for g-xTB is excluded: that build is downloaded separately and
    serves GFN2 as well, so conda's xtb would be a second copy.
    """

    from mdprep.workflows.runtime import manifest_wants_gxtb

    if manifest_wants_gxtb(manifest)[0]:
        return False
    if (manifest.get("protonation") or {}).get("method") == "propka_xtb_his":
        return True
    for site in _mcpb_sites(manifest):
        pyscf = (site.get("mcpb") or {}).get("pyscf") or {}
        if {
            pyscf.get("hessian_backend", "xtb"),
            pyscf.get("esp_backend", "pyscf"),
        } & {"xtb", "gxtb"}:
            return True
    refinement = manifest.get("refinement") or {}
    return bool(
        refinement.get("enabled")
        and refinement.get("backend", "openmm") == "ash"
        and refinement.get("qm_method") == "gfn2_xtb"
    )


def needs_pdbfixer(manifest: dict[str, Any]) -> bool:
    """Two guards demand it, and either is enough.

    structure/repair.py:96 for coordinate repair, and
    protonation/temporary_hydrogenation.py:97 for the hydrogenated
    environment the histidine comparison needs -- which defaults to on and
    is independent of the embedding.
    """

    repair = (manifest.get("structure") or {}).get("repair") or {}
    if repair.get("backend") not in (None, "none"):
        return True
    if (manifest.get("protonation") or {}).get("method") != "propka_xtb_his":
        return False
    return bool(_histidine_xtb(manifest).get("add_missing_protein_hydrogens", True))


def needs_openmm(manifest: dict[str, Any]) -> bool:
    """Everything PDBFixer needs, plus its own callers.

    refinement/openmm_minimizer.py:92, md/roe_brooks.py:280,
    validation/openmm_check.py, and the QM/MM embedding, which reads its MM
    point charges out of OpenMM's force-field files.
    """

    if needs_pdbfixer(manifest):
        return True
    if (manifest.get("protonation") or {}).get("method") == "propka_xtb_his":
        if _histidine_xtb(manifest).get("embedding", "qmmm") == "qmmm":
            return True
    if (manifest.get("refinement") or {}).get("enabled"):
        return True
    if (manifest.get("molecular_dynamics") or {}).get("enabled"):
        return True
    return bool((manifest.get("validation") or {}).get("run_openmm_energy_check", True))


def needs_rdkit(manifest: dict[str, Any]) -> bool:
    """ligands/hydrogenate.py places hydrogens from a SMILES through RDKit."""

    return any(ligand.get("smiles") for ligand in (manifest.get("ligands") or []))


def needs_pyscf(manifest: dict[str, Any]) -> bool:
    """qm/pyscf_runner.py:161 and :560, metals/pyscf_mcpb.py:544."""

    qm_methods = {"gas_resp_pyscf", "qmmesp_pyscf", "mcpb_resp_pyscf"}
    if any(
        ligand.get("charge_method") in qm_methods
        for ligand in (manifest.get("ligands") or [])
    ):
        return True
    return bool(_mcpb_sites(manifest))


def needs_parmed(manifest: dict[str, Any]) -> bool:
    """qm/point_charges.py:102, metals/c4.py:62, metals/pyscf_mcpb.py:1068,
    and validation/parmed_check.py, which always runs."""

    return True


ORACLE: dict[str, Callable[[dict[str, Any]], bool]] = {
    "ambertools": needs_ambertools,
    "propka": needs_propka,
    "xtb": needs_xtb,
    "pdbfixer": needs_pdbfixer,
    "openmm": needs_openmm,
    "rdkit": needs_rdkit,
    "pyscf": needs_pyscf,
    "parmed": needs_parmed,
}


# --------------------------------------------------------------------------
# The option space: every setting that gates one of those.
# --------------------------------------------------------------------------

PROTONATION_METHODS = ("manual_only", "propka", "propka_xtb_his")
HYDROGENATE = (True, False)
EMBEDDINGS = ("qmmm", "cluster")
REPAIR_BACKENDS = (None, "none", "pdbfixer")
CHARGE_METHODS = ("am1bcc", "gas_resp_pyscf", "qmmesp_pyscf", "mcpb_resp_pyscf")
SMILES = ("CCO", None)
METAL_MODELS = (None, "nonbonded", "bonded_mcpb")
HESSIAN_BACKENDS = ("xtb", "pyscf", "gxtb")
ESP_BACKENDS = ("xtb", "pyscf")
XTB_MODELS = ("gfn2", "gxtb")
TOGGLES = (True, False)


def build_manifest(
    *,
    method,
    hydrogenate,
    embedding,
    repair,
    charge_method,
    smiles,
    metal_model,
    hessian_backend,
    esp_backend,
    xtb_model,
    refinement,
    md,
    energy_check,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "protonation": {"method": method},
        "structure": {"repair": {"backend": repair}},
        "ligands": [{"id": "l", "charge_method": charge_method}],
        "metals": [],
        "validation": {"run_openmm_energy_check": energy_check},
        "molecular_dynamics": {"enabled": md},
        "refinement": {"enabled": refinement, "backend": "openmm"},
    }
    if smiles:
        manifest["ligands"][0]["smiles"] = smiles
    if method == "propka_xtb_his":
        manifest["protonation"]["histidine"] = {
            "neutral_tautomer_method": "xtb",
            "xtb": {
                "embedding": embedding,
                "add_missing_protein_hydrogens": hydrogenate,
            },
        }
    if metal_model:
        site: dict[str, Any] = {"id": "m", "model": metal_model}
        if metal_model == "bonded_mcpb":
            site["mcpb"] = {
                "pyscf": {
                    "hessian_backend": hessian_backend,
                    "esp_backend": esp_backend,
                    "xtb": {"model": xtb_model},
                }
            }
        manifest["metals"].append(site)
    return manifest


COMBINATIONS = [
    dict(
        method=method,
        hydrogenate=hydrogenate,
        embedding=embedding,
        repair=repair,
        charge_method=charge_method,
        smiles=smiles,
        metal_model=metal_model,
        hessian_backend=hessian_backend,
        esp_backend=esp_backend,
        xtb_model=xtb_model,
        refinement=refinement,
        md=md,
        energy_check=energy_check,
    )
    for (
        method,
        hydrogenate,
        embedding,
        repair,
        charge_method,
        smiles,
        metal_model,
        hessian_backend,
        esp_backend,
        xtb_model,
        refinement,
        md,
        energy_check,
    ) in product(
        PROTONATION_METHODS,
        HYDROGENATE,
        EMBEDDINGS,
        REPAIR_BACKENDS,
        CHARGE_METHODS,
        SMILES,
        METAL_MODELS,
        HESSIAN_BACKENDS,
        ESP_BACKENDS,
        XTB_MODELS,
        TOGGLES,
        TOGGLES,
        TOGGLES,
    )
]


def test_the_option_space_is_actually_being_covered():
    """A guard against this file quietly shrinking to nothing."""

    assert len(COMBINATIONS) > 10000


@pytest.mark.parametrize("package", sorted(ORACLE))
def test_every_option_combination_installs_what_it_runs(package):
    """For one package, across every combination of settings.

    Split by package so a failure names which dependency is wrong rather
    than only which combination.
    """

    expects = ORACLE[package]
    missing: list[dict[str, Any]] = []
    for options in COMBINATIONS:
        manifest = build_manifest(**options)
        if expects(manifest) and package not in conda_packages_for(manifest):
            missing.append(options)
    assert not missing, (
        f"{package} is needed but not installed for {len(missing)} "
        f"combination(s); first: {missing[0]}"
    )


@pytest.mark.parametrize("package", sorted(ORACLE))
def test_nothing_is_installed_that_the_run_will_not_touch(package):
    """The other direction. A student waits on every package in the list,
    and an unused one is minutes of a Colab session for nothing."""

    expects = ORACLE[package]
    spurious: list[dict[str, Any]] = []
    for options in COMBINATIONS:
        manifest = build_manifest(**options)
        if package in conda_packages_for(manifest) and not expects(manifest):
            spurious.append(options)
    assert not spurious, (
        f"{package} is installed but unused for {len(spurious)} "
        f"combination(s); first: {spurious[0]}"
    )
