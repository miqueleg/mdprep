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
