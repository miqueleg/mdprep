"""Amber Li/Merz nonbonded ion parameter preparation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from mdprep.config.models import MetalSiteConfig
from mdprep.metals.coordination import ResolvedMetalIon, ResolvedMetalSite


class NonbondedMetalError(ValueError):
    """Raised when the requested Amber ion model is unavailable or inconsistent."""


@dataclass(frozen=True)
class AmberIonParameter:
    element: str
    charge: int
    atom_type: str
    mass: float
    rmin_over_2_angstrom: float
    epsilon_kcal_mol: float
    parameter_set: str
    water_model: str
    frcmod_name: str
    frcmod_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "element": self.element,
            "charge": self.charge,
            "atom_type": self.atom_type,
            "mass": self.mass,
            "rmin_over_2_angstrom": self.rmin_over_2_angstrom,
            "epsilon_kcal_mol": self.epsilon_kcal_mol,
            "parameter_set": self.parameter_set,
            "water_model": self.water_model,
            "frcmod_name": self.frcmod_name,
            "frcmod_path": str(self.frcmod_path),
        }


@dataclass(frozen=True)
class NonbondedIonArtifact:
    ion: ResolvedMetalIon
    parameter: AmberIonParameter
    mol2_path: Path
    leap_variable: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self.ion.to_dict(),
            "parameter": self.parameter.to_dict(),
            "mol2_path": str(self.mol2_path),
            "leap_variable": self.leap_variable,
        }


@dataclass(frozen=True)
class NonbondedSiteResult:
    site_id: str
    parameter_set: str
    ions: tuple[NonbondedIonArtifact, ...]
    leap_commands: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "model": "nonbonded",
            "parameter_set": self.parameter_set,
            "ions": [ion.to_dict() for ion in self.ions],
            "leap_commands": list(self.leap_commands),
        }


def prepare_nonbonded_site(
    site: ResolvedMetalSite,
    *,
    water_model: str,
    output_dir: str | Path,
    amberhome: str | Path | None = None,
) -> NonbondedSiteResult:
    if site.config.model != "nonbonded" or site.config.nonbonded is None:
        raise NonbondedMetalError(f"Metal site {site.config.id!r} is not a nonbonded site.")
    # Metal setup commands are composed into tleap scripts executed from
    # several different stage directories. Resolve generated templates once
    # so every preserved script remains independently runnable.
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    parameter_set = _canonical_parameter_set(site.config.nonbonded.parameter_set)
    amber_root = resolve_amberhome(amberhome)

    artifacts: list[NonbondedIonArtifact] = []
    commands: list[str] = []
    defined_types: set[str] = set()
    loaded_frcmods: set[Path] = set()
    residue_templates: dict[str, tuple[str, int, str]] = {}
    for index, ion in enumerate(site.ions, start=1):
        parameter = load_amber_ion_parameter(
            element=ion.element,
            charge=ion.charge,
            water_model=water_model,
            parameter_set=parameter_set,
            amberhome=amber_root,
        )
        residue_name = ion.residue.id.resname
        atom_name = ion.atom.name
        signature = (ion.element, ion.charge, atom_name)
        previous = residue_templates.get(residue_name)
        if previous is not None and previous != signature:
            raise NonbondedMetalError(
                f"Metal residue name {residue_name!r} is used with incompatible ion templates: "
                f"{previous} and {signature}. Use distinct PDB residue names."
            )
        residue_templates[residue_name] = signature
        mol2_path = output / f"{site.config.id}.{index}.{residue_name}.mol2"
        write_single_ion_mol2(ion, parameter, mol2_path)
        variable = _leap_variable(site.config, index, residue_name)
        if parameter.atom_type not in defined_types:
            commands.extend(
                [
                    "addAtomTypes {",
                    f'    {{ "{parameter.atom_type}" "{ion.element}" "sp3" }}',
                    "}",
                ]
            )
            defined_types.add(parameter.atom_type)
        if parameter.frcmod_path not in loaded_frcmods:
            commands.append(f"loadamberparams {parameter.frcmod_path}")
            loaded_frcmods.add(parameter.frcmod_path)
        if previous is None:
            commands.append(f"{variable} = loadmol2 {mol2_path}")
            if variable != residue_name:
                commands.append(f"{residue_name} = {variable}")
        artifacts.append(
            NonbondedIonArtifact(
                ion=ion,
                parameter=parameter,
                mol2_path=mol2_path,
                leap_variable=variable,
            )
        )
    return NonbondedSiteResult(
        site_id=site.config.id,
        parameter_set=parameter_set,
        ions=tuple(artifacts),
        leap_commands=tuple(commands),
    )


def ion_frcmod_name(*, charge: int, water_model: str, parameter_set: str) -> str:
    """Return the exact AmberTools Li/Merz frcmod filename."""

    if charge not in {1, 2, 3, 4}:
        raise NonbondedMetalError("Amber Li/Merz ion models support charges +1 through +4.")
    water = water_model.lower()
    if water not in {"tip3p", "opc"}:
        raise NonbondedMetalError(
            f"mdprep supports Li/Merz metal parameters only for configured TIP3P or OPC, not {water_model}."
        )
    family = _canonical_parameter_set(parameter_set)
    suffix = {"12_6": "126", "12_6_4": "1264", "hfe": "hfe", "iod": "iod"}[family]
    if water == "opc":
        return f"frcmod.ionslm_{suffix}_opc"
    if charge == 1:
        if family in {"12_6", "hfe"}:
            return "frcmod.ions1lm_126_tip3p"
        if family == "iod":
            return "frcmod.ions1lm_iod"
        return "frcmod.ions1lm_1264_tip3p"
    return f"frcmod.ions234lm_{suffix}_tip3p"


def load_amber_ion_parameter(
    *,
    element: str,
    charge: int,
    water_model: str,
    parameter_set: str,
    amberhome: str | Path,
) -> AmberIonParameter:
    name = ion_frcmod_name(
        charge=charge,
        water_model=water_model,
        parameter_set=parameter_set,
    )
    path = Path(amberhome) / "dat" / "leap" / "parm" / name
    if not path.is_file():
        raise NonbondedMetalError(f"Amber ion parameter file is missing: {path}")
    atom_type = f"{element}{charge}+"
    mass, radius, epsilon = _parse_frcmod_ion(path, atom_type)
    return AmberIonParameter(
        element=element,
        charge=charge,
        atom_type=atom_type,
        mass=mass,
        rmin_over_2_angstrom=radius,
        epsilon_kcal_mol=epsilon,
        parameter_set=_canonical_parameter_set(parameter_set),
        water_model=water_model,
        frcmod_name=name,
        frcmod_path=path.resolve(),
    )


def resolve_amberhome(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for variable in ("AMBERHOME", "CONDA_PREFIX"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value))
    tleap = shutil.which("tleap")
    if tleap:
        candidates.append(Path(tleap).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "dat" / "leap" / "parm").is_dir():
            return candidate.resolve()
    raise NonbondedMetalError(
        "Could not locate AmberTools data. Set AMBERHOME or run mdprep in an AmberTools environment."
    )


def write_single_ion_mol2(
    ion: ResolvedMetalIon,
    parameter: AmberIonParameter,
    path: str | Path,
) -> None:
    atom = ion.atom
    residue = ion.residue.id.resname
    text = (
        "@<TRIPOS>MOLECULE\n"
        f"{residue}\n"
        "1 0 1 0 0\n"
        "SMALL\n"
        "USER_CHARGES\n"
        "\n"
        "@<TRIPOS>ATOM\n"
        f"{1:7d} {atom.name:<8s} {atom.x:12.6f} {atom.y:12.6f} {atom.z:12.6f} "
        f"{parameter.atom_type:<8s} {1:4d} {residue:<8s} {float(ion.charge):12.6f}\n"
        "@<TRIPOS>BOND\n"
        "@<TRIPOS>SUBSTRUCTURE\n"
        f"{1:6d} {residue:<8s} {1:6d} GROUP             0 ****  ****    0 ROOT\n"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _parse_frcmod_ion(path: Path, atom_type: str) -> tuple[float, float, float]:
    section = ""
    mass: float | None = None
    radius: float | None = None
    epsilon: float | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped in {"MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONBON"}:
            section = stripped
            continue
        fields = stripped.split()
        if not fields or fields[0] != atom_type:
            continue
        try:
            if section == "MASS":
                mass = float(fields[1])
            elif section == "NONBON":
                radius = float(fields[1])
                epsilon = float(fields[2])
        except (IndexError, ValueError) as exc:
            raise NonbondedMetalError(
                f"Malformed {section} entry for {atom_type} in {path}."
            ) from exc
    if mass is None or radius is None or epsilon is None:
        raise NonbondedMetalError(
            f"Amber parameter set {path.name} has no complete MASS/NONBON entry for "
            f"{atom_type}; choose a supported element, oxidation state, and parameter family."
        )
    return mass, radius, epsilon


def _canonical_parameter_set(value: str) -> str:
    return "12_6" if value == "cm" else value


def _leap_variable(site: MetalSiteConfig, index: int, residue_name: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in site.id.upper())
    safe = safe.strip("_") or "METAL"
    if safe[0].isdigit():
        safe = "M_" + safe
    return f"{safe}_{index}_{residue_name}"
