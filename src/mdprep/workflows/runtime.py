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

import shutil
from typing import Any, Mapping


# tleap, antechamber and parmchk2 all come from this one conda-forge package,
# and no supported workflow avoids tleap.
ALWAYS_REQUIRED = ("ambertools",)

QM_CHARGE_METHODS = {"gas_resp_pyscf", "qmmesp_pyscf", "mcpb_resp_pyscf"}

# g-xTB is not on conda-forge. It is a separate build of xtb published by the
# Grimme group, and the one conda installs is GFN-only, so a bonded metal site
# -- which defaults to a g-xTB Hessian -- cannot run on a conda environment
# alone. The release is pinned rather than tracked, with the checksum the
# release publishes, because a prepared system is supposed to be reproducible
# and "whatever was latest that day" is not.
GXTB_RELEASE_TAG = "v2.0.1"
GXTB_ARCHIVE = "xtb-6.7.1-gxtb-140526-linux-x86_64.tar.xz"
GXTB_SHA256 = "cdfb81164f8d3300fc451dbbf2414528b533e71f7a30ec824b564cd30d6d9e5c"
GXTB_ARCHIVE_ROOT = "xtb-6.7.1"
GXTB_RELEASE_URL = (
    f"https://github.com/grimme-lab/g-xtb/releases/download/{GXTB_RELEASE_TAG}"
)


def manifest_wants_gxtb(manifest: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Whether any metal site asks for a g-xTB Hessian, and where it was told
    to find the binary.

    The model is read rather than the backend name: `hessian_backend: xtb`
    with `xtb.model: gxtb` is the same request as `hessian_backend: gxtb`, and
    it is what the decision engine writes by default.
    """

    wants = False
    configured: str | None = None
    for site in manifest.get("metals") or []:
        if site.get("model") != "bonded_mcpb":
            continue
        pyscf = (site.get("mcpb") or {}).get("pyscf") or {}
        backend = pyscf.get("hessian_backend", "xtb")
        if backend == "gxtb":
            settings = pyscf.get("gxtb") or {}
            model = "gxtb"
        elif backend == "xtb":
            settings = pyscf.get("xtb") or {}
            model = settings.get("model", "gxtb")
        else:
            continue
        if model == "gxtb":
            wants = True
            configured = configured or settings.get("executable")
    return wants, configured


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
        histidine_xtb = (protonation.get("histidine") or {}).get("xtb") or {}
        if histidine_xtb.get("embedding", "qmmm") == "qmmm":
            # The embedded QM/MM tautomer comparison reads its MM point
            # charges out of OpenMM's ff14SB and TIP3P files.
            add("openmm")
        # The tautomer comparison needs a hydrogenated environment around the
        # histidine, and builds one with PDBFixer unless told the input
        # already has it. This defaults to on, and nothing here read it: a
        # run whose structure needed no repair installed no PDBFixer and
        # stopped at protonation, after the whole conda solve.
        if histidine_xtb.get("add_missing_protein_hydrogens", True):
            add("pdbfixer")
            add("openmm")

    for ligand in manifest.get("ligands") or []:
        if ligand.get("charge_method") in QM_CHARGE_METHODS:
            add("pyscf")
        if ligand.get("smiles"):
            # Placing hydrogens from a SMILES goes through RDKit.
            add("rdkit")

    for site in manifest.get("metals") or []:
        if site.get("model") != "bonded_mcpb":
            continue
        # The MCPB driver is PySCF-based whichever engine does the individual
        # steps, so it stays on the list.
        add("pyscf")
        # Which engine computes the Hessian and the ESP is a per-site choice,
        # though, and both default to xTB. Reading them is what makes a metal
        # site installable at all: without this the run reaches for a binary
        # the setup cell never installed.
        pyscf_config = (site.get("mcpb") or {}).get("pyscf") or {}
        backends = {
            pyscf_config.get("hessian_backend", "xtb"),
            pyscf_config.get("esp_backend", "pyscf"),
        }
        if backends & {"xtb", "gxtb"}:
            add("xtb")
    # The g-xTB release is an ordinary xtb 6.7.1 with the g-xTB method added,
    # and it gives GFN2 energies identical to the conda build. When it is
    # being downloaded anyway there is no reason to install a second xtb; the
    # setup cell puts it on PATH under the name every step already looks for.
    if manifest_wants_gxtb(manifest)[0] and "xtb" in packages:
        packages.remove("xtb")

    repair = (manifest.get("structure") or {}).get("repair") or {}
    if repair.get("backend") not in (None, "none"):
        # PDBFixer is an OpenMM application-layer package.
        add("pdbfixer")
        add("openmm")

    refinement = manifest.get("refinement") or {}
    if refinement.get("enabled"):
        # Both backends build the provisional system through OpenMM. Only the
        # ASH backend runs a quantum engine, and ASH itself is a manual
        # install that no package manager here can provide.
        add("openmm")
        if refinement.get("backend", "openmm") == "ash":
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


def gxtb_bin_directory(prefix: str = "/content") -> str:
    """The directory gxtb_setup_commands unpacks the binaries into."""

    return f"{prefix}/{GXTB_ARCHIVE_ROOT}/bin"


def gxtb_executable_path(prefix: str = "/content") -> str:
    """Where gxtb_setup_commands leaves the binary."""

    return f"{gxtb_bin_directory(prefix)}/xtb"


def gxtb_setup_commands(prefix: str = "/content") -> list[str]:
    """Shell lines that install the g-xTB build of xtb beside the environment.

    Separate from the conda packages because conda-forge does not carry it.
    The checksum is verified before anything is unpacked: this downloads a
    100 MB binary that then computes the force constants a student's model is
    built from, so a truncated or substituted file should stop the run rather
    than quietly change the parameters.
    """

    archive = f"{prefix}/{GXTB_ARCHIVE}"
    return [
        f"curl -fLs -o {archive} {GXTB_RELEASE_URL}/{GXTB_ARCHIVE}",
        f'echo "{GXTB_SHA256}  {archive}" | sha256sum -c -',
        f"tar -xJf {archive} -C {prefix}",
        f"export MDPREP_GXTB_EXECUTABLE={gxtb_executable_path(prefix)}",
        # The same binary answers to plain `xtb` for the GFN2 steps -- the
        # histidine tautomers, the ESP -- so no second xtb is installed.
        f"export PATH={gxtb_bin_directory(prefix)}:$PATH",
    ]


def micromamba_run_command(
    manifest: Mapping[str, Any],
    manifest_name: str,
    *,
    environment_name: str = "mdprep-runtime",
    prefix: str = "/content",
) -> str:
    """The one shell line that runs this manifest in the prepared environment.

    mdprep lives in the micromamba environment rather than the notebook's
    Python, so it has to be invoked through it, and `micromamba run` starts a
    fresh environment -- anything the setup cell exported is gone by the time
    the run cell executes, which is why g-xTB is named here as well.
    """

    wants_gxtb, configured = manifest_wants_gxtb(manifest)
    settings = [f"MAMBA_ROOT_PREFIX={prefix}/mamba-root"]
    if wants_gxtb and configured is None:
        settings.append(f"MDPREP_GXTB_EXECUTABLE={gxtb_executable_path(prefix)}")
        # The downloaded build is also the only xtb installed, so the GFN2
        # steps have to be able to find it under its plain name.
        settings.append(f"PATH={gxtb_bin_directory(prefix)}:$PATH")
    return (
        " ".join(settings)
        + f" {prefix}/bin/micromamba run -n {environment_name} "
        + f"mdprep prepare {manifest_name}"
    )


# OpenMM's CUDA platform compiles its kernels at run time with NVRTC, which
# comes with the conda CUDA runtime, and hands the result to the GPU driver.
# A driver older than that NVRTC refuses it: CUDA_ERROR_UNSUPPORTED_PTX_VERSION
# on the first kernel, while OpenCL -- compiled by the driver itself -- works.
# conda-forge resolves to the newest runtime the driver's major release allows,
# which can be newer than the driver: on Colab's T4 runtime the CUDA platform
# never started, and a CUDA 13.0 driver here was given NVRTC 13.4. Capping
# cuda-version at what the driver reports keeps the two in step. Without a GPU
# nothing is capped.
CUDA_PIN_VARIABLE = "MDPREP_CUDA_PIN"


def cuda_pin_commands() -> list[str]:
    """Shell lines that set MDPREP_CUDA_PIN to ``cuda-version<=X.Y``.

    X.Y is the CUDA version the installed driver supports, as nvidia-smi
    reports it; the variable is empty on a machine without an NVIDIA GPU.
    Safe under ``set -euo pipefail``.
    """

    return [
        f'{CUDA_PIN_VARIABLE}=""',
        "if command -v nvidia-smi >/dev/null 2>&1; then",
        (
            "  DRIVER_CUDA=\"$(nvidia-smi 2>/dev/null | sed -n "
            "'s/.*CUDA Version: *\\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p' || true)\""
        ),
        '  if [ -n "$DRIVER_CUDA" ]; then',
        f'    {CUDA_PIN_VARIABLE}="cuda-version<=$DRIVER_CUDA"',
        '    echo "The GPU driver supports CUDA $DRIVER_CUDA: installing a CUDA runtime no newer."',
        "  fi",
        "fi",
    ]


def with_cuda_pin(command: str) -> str:
    """``command`` with the CUDA cap, and without it if nothing fits the cap.

    A driver too old for any OpenMM build with a matching runtime still gets
    a working environment, whose MD then runs on OpenCL.
    """

    return (
        f'{command} ${{{CUDA_PIN_VARIABLE}:+"${CUDA_PIN_VARIABLE}"}} || '
        f'{{ echo "No OpenMM build matches this driver; installing without the cap, '
        f'so the MD will use OpenCL."; {command}; }}'
    )


def micromamba_install_commands(
    manifest: Mapping[str, Any],
    *,
    environment_name: str = "mdprep-runtime",
    prefix: str = "/content",
) -> list[str]:
    """Shell lines that add what a manifest needs to an existing environment.

    For a second manifest in the same Colab session -- and to repair an
    environment built before the CUDA cap, whose CUDA platform cannot start:
    installing with the cap brings its CUDA runtime down to the driver's.
    """

    packages = conda_packages_for(manifest)
    install = (
        f"{prefix}/bin/micromamba install -y -q -n {environment_name} "
        f"-c conda-forge {' '.join(packages)}"
    )
    commands = ["set -euo pipefail", f"export MAMBA_ROOT_PREFIX={prefix}/mamba-root"]
    if "openmm" in packages:
        commands.extend(cuda_pin_commands())
        commands.append(with_cuda_pin(install))
    else:
        commands.append(install)
    return commands


def driver_cuda_version() -> str | None:
    """The CUDA version this machine's NVIDIA driver supports, if there is one."""

    import re
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        output = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", output)
    return match.group(1) if match else None


def explain_cuda_failure(message: str) -> str:
    """What to do about an OpenMM CUDA platform that will not start."""

    driver = driver_cuda_version()
    cap = f'"cuda-version<={driver}"' if driver else '"cuda-version<=X.Y"'
    if "UNSUPPORTED_PTX_VERSION" in message or "PTX" in message:
        return (
            "The CUDA runtime installed with OpenMM is newer than this GPU's "
            f"driver{f' (CUDA {driver})' if driver else ''} can run. Install one no "
            f"newer: micromamba install -n mdprep-runtime -c conda-forge {cap} -- "
            "rerunning section C of the official notebook does this. OpenCL "
            "works meanwhile."
        )
    return (
        "OpenMM's CUDA platform did not start on this GPU. If the driver is "
        f"older than the CUDA runtime, install with {cap}; OpenCL may work "
        "meanwhile."
    )


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

    needed = conda_packages_for(manifest)
    packages = " ".join([f"python={python_version}", *needed])
    create = (
        f"{prefix}/bin/micromamba create -y -n {environment_name} "
        f"-c conda-forge {packages}"
    )
    commands = [
        # Without this the cell cannot fail. It is %%bash, so every line runs
        # whatever the one before it did: a 404 from the release URL pipes
        # "Not Found" into tar, a failed checksum is followed by the unpack
        # anyway, and a conda solve that died still ends on a green tick --
        # the student finds out in the next cell, or not at all. The checksum
        # the note promises only means something with this line present.
        "set -euo pipefail",
        f"mkdir -p {prefix}/bin",
        (
            f"curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest "
            f"| tar -xj -C {prefix}/bin --strip-components=1 bin/micromamba"
        ),
        f"export MAMBA_ROOT_PREFIX={prefix}/mamba-root",
        *(cuda_pin_commands() if "openmm" in needed else []),
        with_cuda_pin(create) if "openmm" in needed else create,
        (
            f"{prefix}/bin/micromamba run -n {environment_name} "
            f"pip install -q {mdprep_spec}"
        ),
    ]
    # A bonded metal site needs a build of xtb that conda-forge does not
    # carry, and without it the run fails only once it reaches the Hessian --
    # after the long part of the setup and most of the preparation.
    wants_gxtb, configured = manifest_wants_gxtb(manifest)
    if wants_gxtb and configured is None:
        commands.extend(gxtb_setup_commands(prefix))
    return commands
