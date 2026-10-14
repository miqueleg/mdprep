"""What this machine can actually run, and how to fix what it cannot.

mdprep drives several large external programs. When one is missing, the
failure surfaces deep inside a run -- an ImportError during refinement, a
KeyError from MCPB.py, an executable not found halfway through protonation --
and the message says what broke rather than what to install. This answers the
question up front, and for a given manifest rather than in the abstract, so
nobody installs PySCF for a job that never calls it.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping

from mdprep.workflows.runtime import conda_packages_for


@dataclass(frozen=True)
class Requirement:
    """One thing a run needs, and where it comes from."""

    name: str
    kind: str  # "python" or "program"
    conda_package: str
    purpose: str
    # Several programs arrive in one conda package, and a Python package's
    # import name is not always its install name.
    probe: str

    @property
    def install_hint(self) -> str:
        return f"conda install -c conda-forge {self.conda_package}"


# Keyed by the conda package name conda_packages_for returns.
REQUIREMENTS: dict[str, tuple[Requirement, ...]] = {
    "ambertools": (
        Requirement("tleap", "program", "ambertools", "builds the Amber system", "tleap"),
        Requirement("antechamber", "program", "ambertools", "ligand atom types and AM1-BCC charges", "antechamber"),
        Requirement("parmchk2", "program", "ambertools", "fills missing ligand parameters", "parmchk2"),
        Requirement("MCPB.py", "program", "ambertools", "bonded metal sites", "MCPB.py"),
    ),
    "propka": (
        Requirement("propka3", "program", "propka", "pH-dependent protonation states", "propka3"),
    ),
    "xtb": (
        Requirement("xtb", "program", "xtb", "histidine tautomers, GFN2 charges and Hessians", "xtb"),
    ),
    "openmm": (
        Requirement("openmm", "python", "openmm", "force-field refinement, MM point charges, validation", "openmm"),
    ),
    "pdbfixer": (
        Requirement("pdbfixer", "python", "pdbfixer", "structure repair and temporary hydrogenation", "pdbfixer"),
    ),
    "rdkit": (
        Requirement("rdkit", "python", "rdkit", "places ligand hydrogens from a SMILES", "rdkit"),
    ),
    "parmed": (
        Requirement("parmed", "python", "parmed", "topology and charge validation", "parmed"),
    ),
    "pyscf": (
        Requirement("pyscf", "python", "pyscf", "QMMESP and MCPB RESP fitting", "pyscf"),
    ),
}


@dataclass(frozen=True)
class Finding:
    requirement: Requirement
    found: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.requirement.name,
            "kind": self.requirement.kind,
            "conda_package": self.requirement.conda_package,
            "purpose": self.requirement.purpose,
            "found": self.found,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DoctorReport:
    findings: tuple[Finding, ...]
    scope: str

    @property
    def missing(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if not finding.found)

    @property
    def ok(self) -> bool:
        return not self.missing

    @property
    def install_command(self) -> str | None:
        packages = []
        for finding in self.missing:
            package = finding.requirement.conda_package
            if package not in packages:
                packages.append(package)
        if not packages:
            return None
        return "conda install -c conda-forge " + " ".join(packages)

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "ok": self.ok,
            "findings": [finding.to_dict() for finding in self.findings],
            "install_command": self.install_command,
        }


def _python_version(module: str) -> str | None:
    if importlib.util.find_spec(module) is None:
        return None
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:
        # Importable but with no distribution metadata is still usable.
        return "present"


def check_requirement(requirement: Requirement) -> Finding:
    if requirement.kind == "python":
        found = _python_version(requirement.probe)
        if found is None:
            return Finding(requirement, False, "not importable")
        return Finding(requirement, True, found)
    path = shutil.which(requirement.probe)
    if path is None:
        return Finding(requirement, False, "not on PATH")
    return Finding(requirement, True, path)


def required_packages(manifest: Mapping[str, Any] | None) -> list[str]:
    """Package names this manifest needs, or every one mdprep can use."""

    if manifest is None:
        return list(REQUIREMENTS)
    return [name for name in conda_packages_for(manifest) if name in REQUIREMENTS]


def run_doctor(manifest: Mapping[str, Any] | None = None, *, scope: str = "") -> DoctorReport:
    findings: list[Finding] = []
    for package in required_packages(manifest):
        for requirement in REQUIREMENTS[package]:
            findings.append(check_requirement(requirement))
    return DoctorReport(
        findings=tuple(findings),
        scope=scope or ("this manifest" if manifest is not None else "every mdprep feature"),
    )


def gxtb_note(manifest: Mapping[str, Any] | None) -> str | None:
    """A bonded metal site defaults to a g-xTB Hessian, which needs its own
    build. The xtb conda installs is GFN-only, so finding `xtb` on PATH is not
    enough to say this will work.
    """

    if manifest is None:
        return None
    from mdprep.metals.gxtb_hessian import (
        GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE,
        gxtb_executable_candidates,
    )

    wants_gxtb = False
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
            wants_gxtb = True
            configured = configured or settings.get("executable")
    if not wants_gxtb:
        return None

    found: list[str] = []
    for candidate in gxtb_executable_candidates(configured, "gxtb"):
        path = Path(candidate).expanduser()
        resolved = str(path) if path.is_file() else shutil.which(candidate)
        if resolved is None:
            continue
        found.append(resolved)
        if supports_gxtb(resolved):
            return None
    remedy = (
        "g-xTB is a separate build from the GFN-only xtb conda installs. Get "
        "it from https://github.com/grimme-lab/g-xtb/releases and set "
        f"{GXTB_EXECUTABLE_ENVIRONMENT_VARIABLE} to it, or set "
        "mcpb.pyscf.xtb.model: gfn2 -- though GFN2 describes transition-metal "
        "coordination poorly and may give a force constant mdprep refuses."
    )
    if found:
        return (
            "A bonded metal site uses a g-xTB Hessian. "
            f"{found[0]} exists but reports no --gxtb support. " + remedy
        )
    return f"A bonded metal site uses a g-xTB Hessian, and none was found. {remedy}"


def supports_gxtb(executable: str) -> bool:
    """Whether this binary actually offers --gxtb.

    Being named xtb says nothing: the conda package is GFN-only and is what
    an unqualified `xtb` on PATH almost always is, so the name alone would
    make this check pass everywhere and help nobody.
    """

    import subprocess

    try:
        result = subprocess.run(
            [executable, "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "--gxtb" in (result.stdout + result.stderr)


def ash_note(manifest: Mapping[str, Any] | None) -> str | None:
    """ASH cannot be installed by any command this tool could print.

    It is a manual checkout, and the `ash` distribution on PyPI is unrelated
    software, so installing that would quietly produce a broken environment.
    """

    if manifest is None:
        return None
    refinement = manifest.get("refinement") or {}
    if not refinement.get("enabled") or refinement.get("backend", "openmm") != "ash":
        return None
    if importlib.util.find_spec("ash") is not None:
        return None
    return (
        "refinement.backend: ash needs ASH, which no package manager here "
        "provides: install it from https://github.com/RagnarB83/ash and put it "
        "on PYTHONPATH. Do not `pip install ash`, which is unrelated software. "
        "refinement.backend: openmm is the default and needs none of this."
    )
