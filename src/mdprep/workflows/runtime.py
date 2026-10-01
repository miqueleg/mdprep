"""Which external programs a particular manifest actually needs.

mdprep's external dependencies are large -- AmberTools alone is a substantial
conda environment, and PySCF is another -- but a given run uses only some of
them. A protein with an AM1-BCC ligand and manual protonation needs AmberTools
and nothing else; PropKa, xTB and PySCF are only pulled in by settings that
actually call them.

Working that out from the manifest keeps a generated setup cell honest in both
directions: it does not claim a job needs a quantum chemistry package it will
never invoke, and it does not omit one the run will fail without.
"""

from __future__ import annotations

from typing import Any, Mapping


# tleap, antechamber and parmchk2 all come from this one conda-forge package,
# and no supported workflow avoids tleap.
ALWAYS_REQUIRED = ("ambertools",)

QM_CHARGE_METHODS = {"gas_resp_pyscf", "qmmesp_pyscf", "mcpb_resp_pyscf"}


def conda_packages_for(manifest: Mapping[str, Any]) -> list[str]:
    """conda-forge packages needed to run `mdprep prepare` on this manifest.

    Returned in a stable order so that a generated command is reproducible.
    """

    packages: list[str] = list(ALWAYS_REQUIRED)

    def add(name: str) -> None:
        if name not in packages:
            packages.append(name)

    protonation = manifest.get("protonation") or {}
    method = protonation.get("method", "manual_only")
    if method in {"propka", "propka_xtb_his"}:
        add("propka")
    if method == "propka_xtb_his":
        # Only the xTB tautomer comparison shells out to the xtb binary.
        add("xtb")

    for ligand in manifest.get("ligands") or []:
        if ligand.get("charge_method") in QM_CHARGE_METHODS:
            add("pyscf")

    for site in manifest.get("metals") or []:
        if site.get("model") == "bonded_mcpb":
            # MCPB sites take their ESP/Hessian work through PySCF.
            add("pyscf")

    repair = (manifest.get("structure") or {}).get("repair") or {}
    if repair.get("backend") not in (None, "none"):
        # PDBFixer is an OpenMM application-layer package.
        add("pdbfixer")
        add("openmm")

    refinement = manifest.get("refinement") or {}
    if refinement.get("enabled"):
        # ASH drives the QM/MM refinement through OpenMM, and the GFN2 engine
        # is the xtb binary.
        add("openmm")
        if refinement.get("qm_method") == "gfn2_xtb":
            add("xtb")

    validation = manifest.get("validation") or {}
    if validation.get("run_openmm_energy_check", True):
        add("openmm")
    # ParmEd backs the topology and charge checks, which always run.
    add("parmed")

    if (manifest.get("molecular_dynamics") or {}).get("enabled"):
        add("openmm")

    return packages


def micromamba_setup_commands(
    manifest: Mapping[str, Any],
    *,
    environment_name: str = "mdprep-runtime",
    python_version: str = "3.11",
    mdprep_spec: str = "mdprep",
    prefix: str = "/content",
) -> list[str]:
    """Shell lines that build a runnable mdprep environment in a Colab VM.

    micromamba is used rather than condacolab because it installs into its own
    prefix instead of replacing the notebook's Python, so the kernel does not
    have to restart and the rest of the notebook keeps working.
    """

    packages = " ".join([f"python={python_version}", *conda_packages_for(manifest)])
    return [
        f"mkdir -p {prefix}/bin",
        (
            f"curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest "
            f"| tar -xj -C {prefix}/bin --strip-components=1 bin/micromamba"
        ),
        f"export MAMBA_ROOT_PREFIX={prefix}/mamba-root",
        (
            f"{prefix}/bin/micromamba create -y -n {environment_name} "
            f"-c conda-forge {packages}"
        ),
        (
            f"{prefix}/bin/micromamba run -n {environment_name} "
            f"pip install -q {mdprep_spec}"
        ),
    ]
