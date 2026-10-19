"""Crystal waters for a predicted model, taken from the closest X-ray structure.

A structure predicted with its cofactors -- Boltz, AlphaFold3 -- has no
waters. For most of the protein that does not matter: tleap's solvation box
fills it. It matters where the water is structure rather than solvent: a
buried water holding two loops together, or the water bound to the iron of an
alpha-ketoglutarate enzyme at rest, where O2 later binds. Those come from the
closest crystal structure: superpose it onto the model and keep the waters
that land somewhere they fit.

PyMOL's ``align`` pairs the residues -- sequence first, then structure, so a
point mutant and a homologue are both handled. The fit itself is done here,
from those pairs, by the Kabsch method with the same outlier rejection
``align`` uses; that keeps the arithmetic testable without PyMOL and keeps
PyMOL's object matrices out of the result. Every protein chain of the model
gets the reference chain that fits it best, and only the waters of that
reference chain, so a monomer model is not given the waters of a crystal's
three other copies.

A water bound to a metal is moved with the metal: the model's iron is rarely
exactly where the crystal's is, and a water 2.1 A from the crystal's iron can
be 2.9 A from the model's. Keeping its position relative to the metal is what
"the water bound to the metal" means; the clash test then decides whether the
model has room for it. Everything is reported.

The model's own coordinates never move. Reading and writing go through
mdprep's reader and writer, so a predicted model's long residue names and
mmCIF chains get the same treatment here as everywhere else.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from mdprep.structure.classify import (
    COORDINATING_METAL_ELEMENTS,
    WATER_RESIDUES,
    is_standard_protein_residue,
)
from mdprep.structure.models import AtomRecord, PdbStructure, build_residues
from mdprep.structure.reader import read_structure
from mdprep.structure.writer import write_pdb

# A water closer than this to a nitrogen, oxygen or halide of the model is
# inside it: the shortest hydrogen bond is about 2.6 A.
CLASH_POLAR = 2.5
# ...or closer than this to a carbon, sulfur or anything else. O and C are in
# contact at about 3.2 A.
CLASH_APOLAR = 2.9
# Farther than this from every atom of the model is bulk solvent, which the
# solvation box provides anyway.
WITHIN = 6.0
# A water oxygen this close to a metal is one of its ligands.
METAL_BOND = 2.6
# A crystal metal stands for a model metal of the same element up to this far
# away once the structures are superposed.
METAL_MATCH = 3.0
# PyMOL's align defaults: drop pairs further apart than twice the current
# RMSD after a fit and fit again, five times at most.
OUTLIER_CUTOFF = 2.0
OUTLIER_CYCLES = 5
# ...but never a pair closer than this: a fit that is already exact would
# otherwise reject its own rounding error.
OUTLIER_FLOOR = 0.5
# Below a TM-score of 0.5 two structures do not share a fold, and the
# superposition says nothing about where the reference's waters would go.
MIN_TM_SCORE = 0.5
# Below this, or below 30% identity, waters are poorly conserved: warned.
# Not higher: a model's His-tag and disordered termini count against the
# score of an otherwise excellent fit, because it is normalised by the
# model's length.
GOOD_TM_SCORE = 0.7

_POLAR = {"N", "O", "F", "CL", "BR", "I"}
_WATER_NAMES = WATER_RESIDUES | {"DOD"}


class WaterTransplantError(RuntimeError):
    """Raised when waters cannot be transplanted safely."""


# --------------------------------------------------------------------------
# Superposition
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Superposition:
    """``rotation`` and ``translation`` take reference coordinates onto the model.

    ``rmsd`` is over the ``aligned`` pairs left after outlier rejection;
    ``tm_score`` is over all ``pairs``, normalised by the model chain's
    length, and is what fits are ranked by: unlike the RMSD it does not
    reward a fit for having dropped most of the protein.
    """

    rotation: np.ndarray
    translation: np.ndarray
    rmsd: float
    aligned: int
    pairs: int
    tm_score: float

    def apply(self, xyz: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(xyz, dtype=float) @ self.rotation.T + self.translation


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rotation and translation that best put ``mobile`` onto ``target``."""

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("kabsch needs two equally long lists of 3D points")
    if len(mobile) < 3:
        raise WaterTransplantError("At least three aligned atoms are needed to superpose")
    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    # A reflection is not a superposition.
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    translation = target_centre - mobile_centre @ rotation.T
    return rotation, translation


def tm_score(deviations: np.ndarray, length: int) -> float:
    """Zhang and Skolnick's TM-score of paired deviations, over ``length``."""

    d0 = max(0.5, 1.24 * (max(length, 19) - 15) ** (1.0 / 3.0) - 1.8)
    return float(np.sum(1.0 / (1.0 + (np.asarray(deviations) / d0) ** 2)) / max(length, 1))


def fit(
    mobile: np.ndarray,
    target: np.ndarray,
    *,
    cutoff: float = OUTLIER_CUTOFF,
    cycles: int = OUTLIER_CYCLES,
    length: int | None = None,
) -> Superposition:
    """Kabsch fit with PyMOL-style outlier rejection.

    Pairs further apart than ``cutoff`` times the current RMSD are left out
    of the next fit, so a loop the prediction placed differently does not
    tilt the core.
    """

    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    used = np.ones(len(mobile), dtype=bool)
    rotation, translation = kabsch(mobile, target)
    for _ in range(cycles):
        deviation = np.linalg.norm(mobile @ rotation.T + translation - target, axis=1)
        rmsd = float(np.sqrt(np.mean(deviation[used] ** 2)))
        keep = used & (deviation <= max(cutoff * rmsd, OUTLIER_FLOOR))
        if keep.sum() < 3 or np.array_equal(keep, used):
            break
        used = keep
        rotation, translation = kabsch(mobile[used], target[used])
    deviation = np.linalg.norm(mobile @ rotation.T + translation - target, axis=1)
    return Superposition(
        rotation=rotation,
        translation=translation,
        rmsd=float(np.sqrt(np.mean(deviation[used] ** 2))),
        aligned=int(used.sum()),
        pairs=len(mobile),
        tm_score=tm_score(deviation, length or len(mobile)),
    )


# --------------------------------------------------------------------------
# What the reference contributes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceWater:
    """A reference water oxygen, already in the model's frame.

    ``model_chain`` is the model chain whose superposition placed it: a
    monomeric reference fitted to both chains of a dimer model yields every
    water twice, once per chain, under the same label.
    """

    label: str
    xyz: tuple[float, float, float]
    # The label of the reference metal it is bound to, if any.
    bound_to: str | None = None
    model_chain: str = ""


@dataclass(frozen=True)
class ReferenceMetal:
    label: str
    element: str
    xyz: tuple[float, float, float]
    model_chain: str = ""


def follow_metals(
    waters: Sequence[ReferenceWater],
    reference_metals: Sequence[ReferenceMetal],
    model_metals: Sequence[AtomRecord],
    *,
    match: float = METAL_MATCH,
) -> tuple[list[ReferenceWater], list[dict[str, object]]]:
    """Move each metal-bound water by however far its metal moved.

    A reference metal stands for the nearest model metal of the same element
    within ``match``. Returns the waters, moved where that applies, and one
    record per moved water.
    """

    partner: dict[tuple[str, str], tuple[ReferenceMetal, AtomRecord, np.ndarray]] = {}
    for metal in reference_metals:
        point = np.array(metal.xyz)
        candidates = [
            atom
            for atom in model_metals
            if (atom.element or "").strip().upper() == metal.element.upper()
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda atom: float(np.linalg.norm(np.array((atom.x, atom.y, atom.z)) - point)),
        )
        shift = np.array((nearest.x, nearest.y, nearest.z)) - point
        if float(np.linalg.norm(shift)) <= match:
            partner[(metal.model_chain, metal.label)] = (metal, nearest, shift)

    moved: list[ReferenceWater] = []
    records: list[dict[str, object]] = []
    for water in waters:
        key = (water.model_chain, water.bound_to or "")
        if water.bound_to is None or key not in partner:
            moved.append(water)
            continue
        metal, atom, shift = partner[key]
        moved.append(replace(water, xyz=tuple(float(v) for v in np.array(water.xyz) + shift)))
        records.append(
            {
                "water": water.label,
                "reference_metal": metal.label,
                "model_metal": _atom_label(atom),
                "shift_angstrom": round(float(np.linalg.norm(shift)), 2),
            }
        )
    return moved, records


# --------------------------------------------------------------------------
# Which waters fit
# --------------------------------------------------------------------------


@dataclass
class WaterSelection:
    kept: list[ReferenceWater] = field(default_factory=list)
    clashing: list[dict[str, object]] = field(default_factory=list)
    distant: list[str] = field(default_factory=list)
    outside_metal_radius: list[str] = field(default_factory=list)
    metal_bound: list[dict[str, object]] = field(default_factory=list)


def _heavy(atom: AtomRecord) -> bool:
    return (atom.element or "").strip().upper() not in {"H", "D"}


def _is_metal(atom: AtomRecord) -> bool:
    return (atom.element or "").strip().upper() in COORDINATING_METAL_ELEMENTS


def _atom_label(atom: AtomRecord) -> str:
    return f"{atom.chain_id}:{atom.resname}{atom.resid}:{atom.name.strip()}"


def select_waters(
    model: PdbStructure,
    waters: Sequence[ReferenceWater],
    *,
    clash_polar: float = CLASH_POLAR,
    clash_apolar: float = CLASH_APOLAR,
    within: float = WITHIN,
    around_metal: float | None = None,
    metal_bond: float = METAL_BOND,
) -> WaterSelection:
    """Decide which superposed reference waters belong in the model.

    A water is kept when some heavy atom of the model is within ``within``,
    and none is closer than the clash distance for its element -- including
    waters already kept, and any the model had. Metals are left out of the
    clash test: a water bound to the iron sits 2.1-2.3 A from it, and that is
    the water this exists to keep. Waters bound to a metal are tried first,
    so a crystal water next to one cannot crowd it out. With
    ``around_metal`` set, only waters that close to a metal are kept.
    """

    heavy = [atom for atom in model.atoms if _heavy(atom)]
    if not any(atom.resname.strip().upper() not in _WATER_NAMES for atom in heavy):
        raise WaterTransplantError("The model has no heavy atoms to place waters around")
    blockers = [atom for atom in heavy if not _is_metal(atom)]
    metals = [atom for atom in heavy if _is_metal(atom)]
    blocker_xyz = np.array([(a.x, a.y, a.z) for a in blockers]).reshape(-1, 3)
    limits = np.array(
        [
            clash_polar if (atom.element or "").strip().upper() in _POLAR else clash_apolar
            for atom in blockers
        ]
    )
    model_xyz = np.array(
        [(a.x, a.y, a.z) for a in heavy if a.resname.strip().upper() not in _WATER_NAMES]
    )
    metal_xyz = np.array([(a.x, a.y, a.z) for a in metals]).reshape(-1, 3)

    selection = WaterSelection()
    kept_xyz: list[np.ndarray] = []
    ordered = sorted(waters, key=lambda water: water.bound_to is None)
    for water in ordered:
        point = np.array(water.xyz, dtype=float)
        if around_metal is not None:
            if not len(metal_xyz) or np.linalg.norm(metal_xyz - point, axis=1).min() > around_metal:
                selection.outside_metal_radius.append(water.label)
                continue
        if np.linalg.norm(model_xyz - point, axis=1).min() > within:
            selection.distant.append(water.label)
            continue
        if len(blocker_xyz):
            distances = np.linalg.norm(blocker_xyz - point, axis=1)
            overlap = distances - limits
            worst = int(np.argmin(overlap))
            if overlap[worst] < 0:
                selection.clashing.append(
                    {
                        "water": water.label,
                        "nearest": _atom_label(blockers[worst]),
                        "distance_angstrom": round(float(distances[worst]), 2),
                        "metal_bound_in_reference": water.bound_to is not None,
                    }
                )
                continue
        if kept_xyz and np.linalg.norm(np.array(kept_xyz) - point, axis=1).min() < clash_polar:
            selection.clashing.append(
                {
                    "water": water.label,
                    "nearest": "another transplanted water",
                    "distance_angstrom": round(
                        float(np.linalg.norm(np.array(kept_xyz) - point, axis=1).min()), 2
                    ),
                    "metal_bound_in_reference": water.bound_to is not None,
                }
            )
            continue
        selection.kept.append(water)
        kept_xyz.append(point)
        if len(metal_xyz):
            metal_distances = np.linalg.norm(metal_xyz - point, axis=1)
            closest = int(np.argmin(metal_distances))
            if metal_distances[closest] <= metal_bond:
                selection.metal_bound.append(
                    {
                        "water": water.label,
                        "metal": _atom_label(metals[closest]),
                        "distance_angstrom": round(float(metal_distances[closest]), 2),
                    }
                )
    return selection


def add_waters(model: PdbStructure, waters: Sequence[ReferenceWater]) -> PdbStructure:
    """The model plus one HOH per water, in the nearest protein chain.

    Each water joins the chain of the protein atom nearest to it and is
    numbered after the highest residue number already in that chain, so a
    chain selection later keeps the waters that belong to it.
    """

    protein_atoms = [
        atom
        for residue in model.residues
        if is_standard_protein_residue(residue)
        for atom in residue.atoms
    ]
    if not protein_atoms:
        raise WaterTransplantError("The model has no protein to attach waters to")
    protein_xyz = np.array([(a.x, a.y, a.z) for a in protein_atoms])
    last_resid: dict[str, int] = {}
    for atom in model.atoms:
        last_resid[atom.chain_id] = max(last_resid.get(atom.chain_id, 0), atom.resid)

    added: list[AtomRecord] = []
    for water in waters:
        point = np.array(water.xyz, dtype=float)
        nearest = protein_atoms[int(np.argmin(np.linalg.norm(protein_xyz - point, axis=1)))]
        chain = nearest.chain_id
        last_resid[chain] = last_resid.get(chain, 0) + 1
        added.append(
            AtomRecord(
                serial=None,
                name="O",
                altloc="",
                resname="HOH",
                chain_id=chain,
                resid=last_resid[chain],
                icode="",
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
                occupancy=1.0,
                bfactor=0.0,
                element="O",
                record_name="HETATM",
                original_line="",
            )
        )
    atoms = list(model.atoms) + added
    return replace(model, atoms=atoms, residues=build_residues(atoms))


# --------------------------------------------------------------------------
# PyMOL
# --------------------------------------------------------------------------


def _pymol():
    try:
        from pymol import cmd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WaterTransplantError(
            "Transplanting waters uses PyMOL to pair the residues of the two "
            "structures. Install it with: conda install -c conda-forge "
            "pymol-open-source"
        ) from exc
    return cmd


def _chain(chain: str) -> str:
    return f"chain {chain}" if chain.strip() else "chain ''"


@dataclass
class ChainFit:
    model_chain: str
    reference_chain: str
    superposition: Superposition
    identity: float | None
    model_ca: int
    # "align" pairs residues by sequence, "super" by structure alone.
    method: str = "align"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_chain": self.model_chain,
            "reference_chain": self.reference_chain,
            "pairing": self.method,
            "tm_score": round(self.superposition.tm_score, 3),
            "rmsd_angstrom": round(self.superposition.rmsd, 3),
            "aligned_ca": self.superposition.aligned,
            "paired_ca": self.superposition.pairs,
            "model_ca": self.model_ca,
            "sequence_identity": None if self.identity is None else round(self.identity, 3),
        }


def _paired_ca(
    cmd, mobile: str, target: str, method: str
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """CA coordinates PyMOL pairs between two selections, and their identity.

    ``align`` pairs by sequence and then structure, which is right for a
    point mutant or a close homologue; ``super`` pairs by structure alone,
    which still works where the sequences have drifted too far for the
    sequence alignment to find the right register.
    """

    try:
        getattr(cmd, method)(mobile, target, cycles=0, transform=0, object="mdprep_aln")
    except Exception as exc:  # PyMOL raises a bare CmdException
        raise WaterTransplantError(f"PyMOL could not align {mobile} to {target}: {exc}") from exc
    pairs = cmd.get_raw_alignment("mdprep_aln")
    cmd.delete("mdprep_aln")
    if not pairs:
        return np.empty((0, 3)), np.empty((0, 3)), None
    coords: dict[tuple[str, int], tuple[float, float, float]] = {}
    names: dict[tuple[str, int], str] = {}
    cmd.iterate_state(
        1,
        f"({mobile}) or ({target})",
        "coords[(model, index)] = (x, y, z); names[(model, index)] = resn",
        space={"coords": coords, "names": names},
    )
    mobile_objects = set(cmd.get_object_list(mobile))
    mobile_xyz, target_xyz, same = [], [], 0
    for left, right in pairs:
        if left[0] not in mobile_objects:
            left, right = right, left
        if left not in coords or right not in coords:
            continue
        mobile_xyz.append(coords[left])
        target_xyz.append(coords[right])
        same += names.get(left) == names.get(right)
    if not mobile_xyz:
        return np.empty((0, 3)), np.empty((0, 3)), None
    return np.array(mobile_xyz), np.array(target_xyz), same / len(mobile_xyz)


@dataclass
class _Reference:
    fits: list[ChainFit]
    waters: list[ReferenceWater]
    metals: list[ReferenceMetal]
    total_waters: int
    # Reference chain -> its metal-bound waters, as "water on metal". Copies
    # of one enzyme in a crystal often differ exactly here.
    bound_by_chain: dict[str, list[str]] = field(default_factory=dict)


def _superpose(model_path: Path, reference_path: Path, reference_chain: str | None) -> _Reference:
    cmd = _pymol()
    cmd.reinitialize()
    # "model" is a reserved word in PyMOL's selection language.
    cmd.load(str(model_path), "mdprep_model")
    cmd.load(str(reference_path), "mdprep_xray")
    cmd.remove("mdprep_xray and not alt ''+A")
    ca = "polymer.protein and name CA"

    model_chains = sorted({a.chain for a in cmd.get_model(f"mdprep_model and {ca}").atom})
    if not model_chains:
        raise WaterTransplantError(f"{model_path.name} has no protein chain to superpose onto")
    reference_chains = sorted({a.chain for a in cmd.get_model(f"mdprep_xray and {ca}").atom})
    if reference_chain is not None:
        if reference_chain not in reference_chains:
            raise WaterTransplantError(
                f"{reference_path.name} has no protein chain {reference_chain!r}; "
                f"it has {', '.join(reference_chains) or 'none'}"
            )
        reference_chains = [reference_chain]
    if not reference_chains:
        raise WaterTransplantError(f"{reference_path.name} has no protein chain to superpose")

    fits: list[ChainFit] = []
    for model_chain in model_chains:
        target = f"mdprep_model and {_chain(model_chain)} and {ca}"
        model_ca = cmd.count_atoms(target)
        best: ChainFit | None = None
        for chain in reference_chains:
            for method in ("align", "super"):
                mobile_xyz, target_xyz, identity = _paired_ca(
                    cmd, f"mdprep_xray and {_chain(chain)} and {ca}", target, method
                )
                if len(mobile_xyz) < 3:
                    continue
                superposition = fit(mobile_xyz, target_xyz, length=model_ca)
                candidate = ChainFit(
                    model_chain, chain, superposition, identity, model_ca, method
                )
                # Copies of one chain in a crystal score within a hair of each
                # other, and a residue more or less resolved should not decide
                # between them: to two decimals they tie, and the tighter fit
                # wins.
                if best is None or (
                    round(superposition.tm_score, 2),
                    -superposition.rmsd,
                ) > (round(best.superposition.tm_score, 2), -best.superposition.rmsd):
                    best = candidate
        if best is None:
            raise WaterTransplantError(
                f"No chain of {reference_path.name} aligns to chain {model_chain!r} "
                f"of {model_path.name}"
            )
        fits.append(best)

    # Each reference water and metal belongs to the protein chain nearest it,
    # decided in the reference's own frame, where the crystal put it.
    protein = cmd.get_model("mdprep_xray and polymer.protein and not hydro").atom
    protein_xyz = np.array([atom.coord for atom in protein])
    protein_chain = [atom.chain for atom in protein]

    def owner(xyz) -> str:
        return protein_chain[int(np.argmin(np.linalg.norm(protein_xyz - np.array(xyz), axis=1)))]

    water_atoms = cmd.get_model(
        "mdprep_xray and resn " + "+".join(sorted(_WATER_NAMES)) + " and elem O"
    ).atom
    metal_selection = "+".join(sorted(element.capitalize() for element in COORDINATING_METAL_ELEMENTS))
    metal_atoms = cmd.get_model(f"mdprep_xray and elem {metal_selection}").atom

    def label(atom) -> str:
        return f"{atom.chain}:{atom.resn}{atom.resi}"

    metal_xyz = np.array([atom.coord for atom in metal_atoms]).reshape(-1, 3)
    bound_by_chain: dict[str, list[str]] = {}
    if len(metal_xyz):
        for atom in water_atoms:
            distances = np.linalg.norm(metal_xyz - np.array(atom.coord), axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= METAL_BOND:
                bound_by_chain.setdefault(owner(atom.coord), []).append(
                    f"{label(atom)} on {label(metal_atoms[nearest])}"
                )
    waters: list[ReferenceWater] = []
    metals: list[ReferenceMetal] = []
    for chain_fit in fits:
        mapped = chain_fit.reference_chain
        for atom in metal_atoms:
            if owner(atom.coord) == mapped:
                metals.append(
                    ReferenceMetal(
                        label=label(atom),
                        element=atom.symbol.upper(),
                        xyz=tuple(float(v) for v in chain_fit.superposition.apply(atom.coord)),
                        model_chain=chain_fit.model_chain,
                    )
                )
        for atom in water_atoms:
            if owner(atom.coord) != mapped:
                continue
            bound_to = None
            if len(metal_xyz):
                distances = np.linalg.norm(metal_xyz - np.array(atom.coord), axis=1)
                nearest = int(np.argmin(distances))
                if distances[nearest] <= METAL_BOND:
                    bound_to = label(metal_atoms[nearest])
            waters.append(
                ReferenceWater(
                    label=label(atom),
                    xyz=tuple(float(v) for v in chain_fit.superposition.apply(atom.coord)),
                    bound_to=bound_to,
                    model_chain=chain_fit.model_chain,
                )
            )
    cmd.delete("all")
    return _Reference(
        fits=fits,
        waters=waters,
        metals=metals,
        total_waters=len(water_atoms),
        bound_by_chain=bound_by_chain,
    )


# --------------------------------------------------------------------------
# The whole operation
# --------------------------------------------------------------------------


@dataclass
class WaterTransplantResult:
    model_path: Path
    reference_path: Path
    output_path: Path
    fits: list[ChainFit]
    reference_waters: int
    candidate_waters: int
    selection: WaterSelection
    moved_with_metal: list[dict[str, object]]
    metal_displacements: list[dict[str, object]]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": str(self.model_path),
            "reference": str(self.reference_path),
            "output": str(self.output_path),
            "chains": [chain_fit.to_dict() for chain_fit in self.fits],
            "reference_waters": self.reference_waters,
            "candidate_waters": self.candidate_waters,
            "transplanted": len(self.selection.kept),
            "rejected_clashing": self.selection.clashing,
            "rejected_distant": len(self.selection.distant),
            "rejected_outside_metal_radius": len(self.selection.outside_metal_radius),
            "metal_bound_waters": self.selection.metal_bound,
            "moved_with_metal": self.moved_with_metal,
            "metal_displacements": self.metal_displacements,
            "warnings": self.warnings,
        }


def transplant_waters(
    model_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    reference_chain: str | None = None,
    clash_polar: float = CLASH_POLAR,
    clash_apolar: float = CLASH_APOLAR,
    within: float = WITHIN,
    around_metal: float | None = None,
    metal_bond: float = METAL_BOND,
    follow_metal: bool = True,
) -> WaterTransplantResult:
    """Copy the reference's waters onto the model and write the result as PDB."""

    model_file = Path(model_path)
    reference_file = Path(reference_path)
    output = Path(output_path)
    model = read_structure(model_file)
    reference = _superpose(model_file, reference_file, reference_chain)
    if not reference.total_waters:
        raise WaterTransplantError(f"{reference_file.name} has no waters to transplant")

    for chain_fit in reference.fits:
        if chain_fit.superposition.tm_score < MIN_TM_SCORE:
            raise WaterTransplantError(
                f"{reference_file.name} does not share a fold with chain "
                f"{chain_fit.model_chain} of {model_file.name}: its best chain "
                f"({chain_fit.reference_chain}) superposes with a TM-score of "
                f"{chain_fit.superposition.tm_score:.2f}, and two structures with "
                "the same fold score above 0.5. Its waters would land at random. Use "
                "a closer structure: search the PDB with the model's sequence "
                "(mdprep closest-xray)."
            )

    warnings = list(model.warnings)
    for chain_fit in reference.fits:
        where = f"Chain {chain_fit.model_chain} (reference chain {chain_fit.reference_chain})"
        if chain_fit.superposition.tm_score < GOOD_TM_SCORE:
            warnings.append(
                f"{where} superposes with a TM-score of "
                f"{chain_fit.superposition.tm_score:.2f} (RMSD "
                f"{chain_fit.superposition.rmsd:.2f} A over "
                f"{chain_fit.superposition.aligned} CA). A close homologue scores "
                "above 0.8; below that the active site may differ enough that its "
                "waters do not fit. Check the result, or choose a closer structure."
            )
        if chain_fit.identity is not None and chain_fit.identity < 0.3:
            warnings.append(
                f"{where}: {chain_fit.identity:.0%} of the paired residues are identical. "
                "Waters are conserved poorly between distant homologues; look for a "
                "closer structure."
            )

    used_chains = {chain_fit.reference_chain for chain_fit in reference.fits}
    if not any(reference.bound_by_chain.get(chain) for chain in used_chains):
        elsewhere = {
            chain: bound
            for chain, bound in sorted(reference.bound_by_chain.items())
            if chain not in used_chains
        }
        if elsewhere and reference_chain is None:
            listed = "; ".join(
                f"chain {chain}: {', '.join(bound)}" for chain, bound in elsewhere.items()
            )
            warnings.append(
                f"The reference chain used ({', '.join(sorted(used_chains))}) has no water "
                f"on a metal, but other copies in the crystal do ({listed}). Copies of "
                "one enzyme often differ exactly here. If your model is in that state, "
                "transplant from that chain instead (reference chain option)."
            )

    model_metals = [atom for atom in model.atoms if _is_metal(atom)]
    waters = reference.waters
    moved: list[dict[str, object]] = []
    if follow_metal:
        waters, moved = follow_metals(waters, reference.metals, model_metals)

    displacements: list[dict[str, object]] = []
    for atom in model_metals:
        element = (atom.element or "").strip().upper()
        same = [metal for metal in reference.metals if metal.element == element]
        if not same:
            warnings.append(
                f"The model's {_atom_label(atom)} has no {element.capitalize()} in the "
                "reference to compare it with, so any water on it is the model's "
                "business, not the crystal's."
            )
            continue
        point = np.array((atom.x, atom.y, atom.z))
        nearest = min(same, key=lambda metal: float(np.linalg.norm(np.array(metal.xyz) - point)))
        distance = float(np.linalg.norm(np.array(nearest.xyz) - point))
        displacements.append(
            {
                "model_metal": _atom_label(atom),
                "reference_metal": nearest.label,
                "distance_angstrom": round(distance, 2),
            }
        )
        if distance > METAL_MATCH:
            warnings.append(
                f"The model's {_atom_label(atom)} sits {distance:.2f} A from the "
                f"reference's {nearest.label} after superposition: these are not the "
                "same site, and no water was moved with it."
            )

    selection = select_waters(
        model,
        waters,
        clash_polar=clash_polar,
        clash_apolar=clash_apolar,
        within=within,
        around_metal=around_metal,
        metal_bond=metal_bond,
    )
    lost = [entry for entry in selection.clashing if entry["metal_bound_in_reference"]]
    for entry in lost:
        warnings.append(
            f"{entry['water']} is bound to a metal in the reference but would sit "
            f"{entry['distance_angstrom']} A from {entry['nearest']} in the model, so "
            "it was not transplanted. Compare the two metal sites: the model may have "
            "put a ligand where the crystal has this water."
        )
    # Only a metal the model also has can lose its water: a crystal's surface
    # ion with no counterpart in the model has nothing to compare against.
    matched = {
        (water.model_chain, water.bound_to)
        for water in waters
        if water.bound_to is not None
        and any(
            metal.label == water.bound_to
            and metal.model_chain == water.model_chain
            and any(
                (atom.element or "").strip().upper() == metal.element
                and float(np.linalg.norm(np.array((atom.x, atom.y, atom.z)) - np.array(metal.xyz)))
                <= METAL_MATCH
                for atom in model_metals
            )
            for metal in reference.metals
        )
    }
    if matched and not selection.metal_bound and not lost:
        warnings.append(
            "The reference has a water on a metal the model also has, but none ends "
            "up bound to it in the model. Compare the two metal sites before going on."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    write_pdb(add_waters(model, selection.kept), output)
    return WaterTransplantResult(
        model_path=model_file,
        reference_path=reference_file,
        output_path=output,
        fits=reference.fits,
        reference_waters=reference.total_waters,
        candidate_waters=len(waters),
        selection=selection,
        moved_with_metal=moved,
        metal_displacements=displacements,
        warnings=warnings,
    )


def fetch_reference(pdb_id: str, directory: str | Path) -> Path:
    """Download an entry from the RCSB as mmCIF, which every entry has."""

    code = pdb_id.strip().upper()
    if len(code) != 4 or not code.isalnum():
        raise WaterTransplantError(f"{pdb_id!r} is not a four-character PDB ID")
    target = Path(directory) / f"{code}.cif"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{code}.cif"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed host
            target.write_bytes(response.read())
    except OSError as exc:
        raise WaterTransplantError(f"Could not download {code} from the RCSB: {exc}") from exc
    return target
