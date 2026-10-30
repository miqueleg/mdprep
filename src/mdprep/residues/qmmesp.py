"""QMMESP charges for a noncanonical residue of a chain.

The residue is fitted the way it is parameterised (``parameterise.py``):
between an acetyl and a methylamide built from its neighbours' atoms. That
capped residue is the QM region. Around it sits the provisional system the
ligand QMMESP fits use -- the protonated protein, its crystal waters and
ions, every ligand with AM1-BCC charges and this residue with its own -- and
every atom of it is an MM point charge that polarises the PySCF density
(``pyscf.qmmm.mm_charge``), except the residue itself and the two residues
the caps stand in for. Those are left out whole: a cap's methyl carbon is
where a neighbour's CA is, its hydrogens almost where that neighbour's N,
HA and CB are, and leaving the whole residue out keeps the embedding's
charge a whole number.

The ESP of the polarised density is then fitted with Amber's two-stage RESP
(``respgen`` and ``resp``) under the constraints Amber's own protocol for a
new residue uses: the residue's backbone N, H, C and O held at the force
field's charges for a residue of its sign, and each cap summing to zero, so
the residue's own atoms carry its integer charge. Only those atoms' charges
are kept. The MM point charges are not fitted, not added to the ESP that is,
and not written anywhere but the QM directory's record of them.

The fitted charges replace the AM1-BCC ones in a library built with the same
atom types and the same frcmod: charges are all the fit changes.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, replace
from math import dist
from pathlib import Path

import numpy as np

from mdprep.ambertools.resp import AmberRespError, run_amber_resp_fit
from mdprep.charges.esp_grid import EspGridError, generate_merz_kollman_grid, write_grid_xyz
from mdprep.config.models import ManifestConfig, NoncanonicalResidueConfig
from mdprep.leap.forcefields import ResidueLibraryFiles
from mdprep.leap.runner import TLeapRun
from mdprep.qm.point_charges import (
    PointChargeError,
    PointChargeSelection,
    extract_residue_point_charges,
    write_point_charge_files,
)
from mdprep.qm.pyscf_esp import (
    BOHR_PER_ANGSTROM,
    PySCFEspError,
    evaluate_ligand_esp,
    write_esp_values,
)
from mdprep.qm.pyscf_runner import PySCFRunnerError, run_pyscf_scf
from mdprep.residues.capping import (
    CappedResidue,
    ResidueCappingError,
    capped_residue,
    chain_context,
)
from mdprep.residues.charges import (
    BACKBONE_CHARGES,
    ResidueChargeError,
    ResidueCharges,
    backbone_set,
    held_backbone_charges,
)
from mdprep.residues.parameterise import (
    ParameterisedResidue,
    ResidueParameterisationError,
    build_residue_library,
)
from mdprep.residues.workflow import (
    ResidueStageError,
    ResidueStageResult,
    block_residues,
    residue_directory,
    residue_leap_inputs,
    write_stage_failure,
)
from mdprep.structure.models import PdbStructure, ResidueRecord

QMMESP_RESIDUE_CONFIRMATION = (
    "The capped residue's PySCF single point was run as electrostatically embedded "
    "QM/MM using pyscf.qmmm.mm_charge. Fixed MM point charges from the provisional "
    "Amber system polarized its QM density; the residue and the two residues its caps "
    "stand in for were not among them. The RESP fitting target contains only the "
    "polarized capped residue's ESP, with the residue's backbone N, H, C and O held at "
    "the force field's charges and each cap held to a sum of zero; only the residue's "
    "own atoms' charges were kept. Environment point charges were not fitted and were "
    "not written to the residue's library."
)

# The residue's hydrogens are placed in the structure from the same SMILES
# and heavy atoms as the capped model's; anything further apart than this is
# not the same molecule.
COORDINATE_TOLERANCE_ANGSTROM = 0.01

# Further than this from its AM1-BCC charge, an atom's fitted charge is more
# likely RESP's freedom with a buried atom than polarisation: on osteocalcin's
# Gla the methine between the two carboxylates went from -0.27 to +1.19 e.
DEPARTURE_FROM_AM1BCC_E = 0.5


class ResidueQmmespError(ValueError):
    """Raised when a residue's QMMESP charges cannot be fitted safely."""


@dataclass
class ResidueQmmespResult:
    resname: str
    fitted_from: str
    method: str
    basis: str
    qm_dir: Path
    files: ResidueLibraryFiles
    residue_mol2_path: Path
    charges: ResidueCharges
    provisional_charges: dict[str, float]
    embedding: PointChargeSelection
    pyscf_result: dict[str, object]
    fit_result: dict[str, object]
    grid_point_count: int
    tleap: TLeapRun
    warnings: list[str] = field(default_factory=list)

    @property
    def point_charge_count(self) -> int:
        return self.embedding.total_after_cutoff

    @property
    def net_embedding_charge(self) -> float:
        return self.embedding.net_embedding_charge

    @property
    def excluded_residues(self) -> tuple[str, ...]:
        return self.embedding.excluded_residues

    @property
    def rms_error(self) -> float:
        return float(self.fit_result["rms_error"])

    @property
    def relative_rms_error(self) -> float | None:
        value = self.fit_result.get("relative_rms_error")
        return None if value is None else float(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "resname": self.resname,
            "fitted_from": self.fitted_from,
            "method": self.method,
            "basis": self.basis,
            "qm_dir": str(self.qm_dir),
            "library_path": str(self.files.library_path),
            "frcmod_path": str(self.files.frcmod_path),
            "residue_mol2_path": str(self.residue_mol2_path),
            "charges": self.charges.to_dict(),
            "provisional_am1bcc_charges": dict(self.provisional_charges),
            "embedding": self.embedding.to_dict(),
            "pyscf_result": self.pyscf_result,
            "fit": self.fit_result,
            "grid_point_count": self.grid_point_count,
            "tleap": self.tleap.to_dict(),
            "confirmation": QMMESP_RESIDUE_CONFIRMATION,
            "warnings": list(self.warnings),
        }


def run_residue_qmmesp_stage(
    structure: PdbStructure,
    manifest: ManifestConfig,
    residue_result: ResidueStageResult,
    *,
    output_dir: str | Path,
    prmtop: str | Path,
    inpcrd: str | Path,
) -> ResidueStageResult:
    """Fit every QMMESP block's charges in the provisional system given.

    ``structure`` is the protonated structure the provisional system was
    built from. Returns the stage result with each fit, and with the leap
    inputs of the final build, which take the fitted libraries.
    """

    configs = {block.resname: block for block in manifest.noncanonical_residues}
    try:
        for item in residue_result.blocks:
            block = configs[item.resname]
            if block.parameters != "qmmesp_pyscf":
                continue
            if item.parameterised is None:
                raise ResidueQmmespError(
                    f"noncanonical residue {block.resname} has no provisional AM1-BCC "
                    "parameterisation to fit QMMESP charges for."
                )
            item.qmmesp = fit_residue_qmmesp(
                block,
                structure,
                item.parameterised,
                manifest=manifest,
                prmtop=Path(prmtop),
                inpcrd=Path(inpcrd),
                output_dir=output_dir,
            )
            item.warnings.extend(item.qmmesp.warnings)
        inputs = residue_leap_inputs(manifest, output_dir)
    except (ResidueQmmespError, ResidueStageError) as exc:
        write_stage_failure(output_dir, exc)
        raise ResidueStageError(str(exc)) from exc
    return ResidueStageResult(blocks=residue_result.blocks, leap_inputs=inputs)


def fit_residue_qmmesp(
    block: NoncanonicalResidueConfig,
    structure: PdbStructure,
    provisional: ParameterisedResidue,
    *,
    manifest: ManifestConfig,
    prmtop: Path,
    inpcrd: Path,
    output_dir: str | Path,
) -> ResidueQmmespResult:
    """Fit ``block``'s charges from its first residue, embedded in ``prmtop``."""

    assert block.qmmesp is not None and block.smiles is not None
    assert block.net_charge is not None
    config = block.qmmesp
    name = block.resname
    try:
        residues = block_residues(structure, block)
    except ResidueStageError as exc:
        raise ResidueQmmespError(str(exc)) from exc
    first = residues[0]
    label = f"noncanonical residue {name} ({first.id.display()})"
    try:
        context = chain_context(structure.residues, first)
        capped = capped_residue(context, block.smiles, net_charge=block.net_charge, label=label)
    except ResidueCappingError as exc:
        raise ResidueQmmespError(str(exc)) from exc
    capped = at_structure_coordinates(capped, first, label=label)
    own = [atom.name for atom in capped.atoms if atom.role == "RES"]
    if set(own) != set(provisional.atom_types):
        raise ResidueQmmespError(
            f"{label}: its atoms {sorted(own)} are not those its provisional library was "
            f"made with, {sorted(provisional.atom_types)}."
        )

    elements = [atom.element.capitalize() for atom in capped.atoms]
    coordinates = np.asarray([[atom.x, atom.y, atom.z] for atom in capped.atoms], dtype=float)
    directory = residue_directory(output_dir, name)
    qm_dir = directory / "qm" / "qmmesp_pyscf"
    qm_dir.mkdir(parents=True, exist_ok=True)
    try:
        embedding = extract_residue_point_charges(
            prmtop=prmtop,
            inpcrd=inpcrd,
            resname=name,
            residue_coordinates={
                atom.name: (atom.x, atom.y, atom.z) for atom in capped.atoms if atom.role == "RES"
            },
            qm_coordinates=coordinates,
            qmmesp=config,
            manifest=manifest,
            label=label,
            expected_neighbours=[
                (context.previous.id.resname, context.previous.id.display()),
                (context.following.id.resname, context.following.id.display()),
            ],
        )
    except PointChargeError as exc:
        raise ResidueQmmespError(str(exc)) from exc
    if embedding.total_after_cutoff == 0:
        raise ResidueQmmespError(
            f"{label}: the embedding selected no MM point charges; refusing to run a "
            "gas-phase SCF under a QMMESP label. Widen the environment or the cutoff."
        )
    fixed, groups = fit_constraints(capped, block.net_charge)
    (qm_dir / "pyscf_input.json").write_text(
        json.dumps(
            {
                "residue": name,
                "fitted_from": first.id.display(),
                "elements": elements,
                "atom_names": [atom.name for atom in capped.atoms],
                "atom_roles": [atom.role for atom in capped.atoms],
                "coordinates_angstrom": coordinates.tolist(),
                "charge": block.net_charge,
                "spin": 0,
                "multiplicity": 1,
                "method": config.method,
                "basis": config.basis,
                "max_cycle": config.max_cycle,
                "conv_tol": config.conv_tol,
                "num_threads": config.num_threads,
                "max_memory_mb": config.max_memory_mb,
                "calculation_type": "electrostatically_embedded_qm_mm_single_point",
                "qm_region": "capped_residue",
                "caps": (
                    "acetyl on N from the previous residue's CA, C and O; methylamide "
                    "on C from the next residue's N and CA; hydrogens placed by RDKit"
                ),
                "qm_atom_count": len(capped.atoms),
                "mm_region": (
                    "provisional Amber system without the residue and the two residues "
                    "its caps stand in for"
                ),
                "mm_point_charge_count": embedding.total_after_cutoff,
                "embedding_operator": "pyscf.qmmm.mm_charge",
                "mm_bonded_and_lennard_jones_terms_in_scf": False,
                "direct_mm_potential_in_resp_target": False,
                "esp_grid": config.grid.model_dump(mode="json"),
                "resp_fitting": config.resp_fitting.model_dump(mode="json"),
                "resp_fixed_charges": {
                    capped.atoms[index].name: value for index, value in sorted(fixed.items())
                },
                "resp_charge_groups": [
                    {"atoms": [capped.atoms[index].name for index in members], "charge": charge}
                    for members, charge in groups
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_point_charge_files(
        embedding,
        csv_path=qm_dir / "mm_point_charges.csv",
        xyz_path=qm_dir / "mm_point_charges.xyz",
        summary_path=qm_dir / "embedding_summary.json",
    )
    try:
        grid = generate_merz_kollman_grid(
            elements=elements,
            coordinates=coordinates,
            vdw_scale_factors=config.grid.vdw_scale_factors,
            point_density_per_square_angstrom=config.grid.point_density_per_square_angstrom,
            exclude_inside_vdw_scale=config.grid.exclude_inside_vdw_scale,
            max_points=config.grid.max_points,
        )
        scf = run_pyscf_scf(
            elements=elements,
            coordinates=coordinates,
            charge=block.net_charge,
            spin=0,
            method=config.method,
            basis=config.basis,
            max_cycle=config.max_cycle,
            conv_tol=config.conv_tol,
            mm_charges=embedding.charge_array,
            mm_coordinates=embedding.coordinate_array,
            work_dir=qm_dir,
            checkpoint_path=qm_dir / "pyscf.chk",
            num_threads=config.num_threads,
            max_memory_mb=config.max_memory_mb,
        )
        (qm_dir / "pyscf_stdout.txt").write_text(scf.stdout, encoding="utf-8")
        (qm_dir / "pyscf_stderr.txt").write_text(scf.stderr, encoding="utf-8")
        (qm_dir / "pyscf_result.json").write_text(
            json.dumps(scf.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        esp = evaluate_ligand_esp(
            mol=scf.mol, mf=scf.mf, grid_coordinates_angstrom=grid.points
        )
        write_grid_xyz(grid, qm_dir / "esp_grid.xyz")
        write_esp_values(esp, str(qm_dir / "esp_values.dat"))
        fit = run_amber_resp_fit(
            provisional_mol2_path=provisional.antechamber_mol2_path,
            atom_coordinates_bohr=coordinates * BOHR_PER_ANGSTROM,
            grid_coordinates_bohr=grid.points * BOHR_PER_ANGSTROM,
            esp_values_au=esp,
            total_charge=block.net_charge,
            multiplicity=1,
            atom_types=block.atom_types,
            work_dir=qm_dir / "resp",
            fixed_charges=fixed,
            charge_groups=groups,
        )
    except (EspGridError, PySCFRunnerError, PySCFEspError, AmberRespError) as exc:
        raise ResidueQmmespError(f"{label}: {exc}") from exc
    try:
        charges = held_backbone_charges(
            [
                (atom.role, atom.name, float(charge))
                for atom, charge in zip(capped.atoms, fit.charges, strict=True)
            ],
            block.net_charge,
        )
    except ResidueChargeError as exc:
        raise ResidueQmmespError(f"{label}: {exc}") from exc

    # The provisional library's types and frcmod, with the fitted charges.
    frcmod = directory / f"{name}.frcmod"
    shutil.copyfile(provisional.files.frcmod_path, frcmod)
    try:
        library, residue_mol2, tleap = build_residue_library(
            capped,
            resname=name,
            types=[
                provisional.atom_types[atom.name] if atom.role == "RES" else atom.element
                for atom in capped.atoms
            ],
            charges=[
                charges.charges[atom.name] if atom.role == "RES" else 0.0
                for atom in capped.atoms
            ],
            protein_forcefield=manifest.protein.forcefield,
            atom_types=block.atom_types,
            frcmod=frcmod,
            label=label,
        )
    except ResidueParameterisationError as exc:
        raise ResidueQmmespError(str(exc)) from exc

    fit_report = {
        **fit.to_dict(),
        "grid_point_count": int(len(grid.points)),
        "fitted_charge_centers": [atom.name for atom in capped.atoms],
        "kept_charges": list(charges.charges),
        "mm_point_charge_count": embedding.total_after_cutoff,
        "external_mm_potential_included_in_fit": False,
        "confirmation": QMMESP_RESIDUE_CONFIRMATION,
    }
    (qm_dir / "fit_report.json").write_text(
        json.dumps(fit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    warnings = list(scf.warnings)
    departed = {
        atom: (charge, provisional.charges.charges[atom])
        for atom, charge in charges.charges.items()
        if abs(charge - provisional.charges.charges[atom]) > DEPARTURE_FROM_AM1BCC_E
    }
    if departed:
        warnings.append(
            f"{name}'s QMMESP charges depart from its AM1-BCC ones by more than "
            f"{DEPARTURE_FROM_AM1BCC_E} e at "
            + ", ".join(
                f"{atom} ({fitted:+.2f} against {am1bcc:+.2f})"
                for atom, (fitted, am1bcc) in departed.items()
            )
            + ". Polarisation does not move one atom's charge that far. A fit to one "
            "conformation lets an atom the ESP outside the molecule barely sees -- a "
            "carbon buried among its neighbours -- trade charge with the atoms around "
            "it. Compare the two sets in the residue report before simulating with "
            "these; parameters: am1bcc does not fail this way."
        )
    result = ResidueQmmespResult(
        resname=name,
        fitted_from=first.id.display(),
        method=config.method,
        basis=config.basis,
        qm_dir=qm_dir,
        files=ResidueLibraryFiles(resname=name, library_path=library, frcmod_path=frcmod),
        residue_mol2_path=residue_mol2,
        charges=charges,
        provisional_charges=dict(provisional.charges.charges),
        embedding=embedding,
        pyscf_result=scf.to_dict(),
        fit_result=fit_report,
        grid_point_count=int(len(grid.points)),
        tleap=tleap,
        warnings=warnings,
    )
    (directory / "parameters.json").write_text(
        json.dumps(
            {
                **provisional.to_dict(),
                "charge_method": "qmmesp_pyscf",
                "library_path": str(library),
                "frcmod_path": str(frcmod),
                "residue_mol2_path": str(residue_mol2),
                "charges": charges.to_dict(),
                "qmmesp": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def fit_constraints(
    capped: CappedResidue, net_charge: int
) -> tuple[dict[int, float], list[tuple[list[int], float]]]:
    """The RESP constraints of a capped residue, by atom index.

    The residue's backbone N, H, C and O held at the force field's charges
    for its sign, and each cap's atoms summing to zero.
    """

    index = {(atom.role, atom.name): number for number, atom in enumerate(capped.atoms)}
    backbone = BACKBONE_CHARGES[backbone_set(net_charge)]
    missing = [atom for atom in backbone if ("RES", atom) not in index]
    if missing:
        raise ResidueQmmespError(
            f"the capped residue has no {', '.join(missing)}, whose charges the fit holds"
        )
    fixed = {index[("RES", atom)]: value for atom, value in backbone.items()}
    groups = [
        ([number for number, atom in enumerate(capped.atoms) if atom.role == cap], 0.0)
        for cap in ("ACE", "NME")
    ]
    return fixed, groups


def at_structure_coordinates(
    capped: CappedResidue, residue: ResidueRecord, *, label: str
) -> CappedResidue:
    """``capped`` with the residue's atoms where the structure has them.

    The provisional system holds the residue as the structure does, and its
    embedding is found and measured from there. Its heavy atoms are the
    capped model's own; its hydrogens were placed from the same SMILES by the
    same procedure, so they agree too, and a residue whose atoms do not is
    not the one the model describes.
    """

    placed = {atom.name.strip(): atom for atom in residue.atoms}
    atoms = []
    for atom in capped.atoms:
        if atom.role != "RES":
            atoms.append(atom)
            continue
        source = placed.get(atom.name)
        if source is None:
            raise ResidueQmmespError(
                f"{label}: the structure has no {atom.name}, which its SMILES gives it."
            )
        moved = dist((atom.x, atom.y, atom.z), (source.x, source.y, source.z))
        if moved > COORDINATE_TOLERANCE_ANGSTROM:
            raise ResidueQmmespError(
                f"{label}: {atom.name} is {moved:.3f} A from where the capped model "
                "places it; the residue in the structure is not the one its SMILES "
                "describes."
            )
        atoms.append(replace(atom, x=source.x, y=source.y, z=source.z))
    extra = sorted(set(placed) - {atom.name for atom in capped.atoms if atom.role == "RES"})
    if extra:
        raise ResidueQmmespError(
            f"{label}: the structure gives it atoms its SMILES does not: {', '.join(extra)}."
        )
    return replace(capped, atoms=atoms)
