"""Electrostatically embedded QM/MM selection of a histidine tautomer.

The cluster backend answers "is this histidine HID or HIE?" by carving a few
hundred atoms out of the protein and handing the lot to xTB. That forces it to
state the carved fragment's total charge, which it computes by summing formal
per-residue charges -- and a CA-truncated, hydrogen-capped fragment does not
carry the charge of the residues it was cut from. On the BesD reference system
three of six clusters came out with a charge that contradicted their own atom
composition, and xTB refused them before the first SCF cycle.

Embedding removes the question rather than answering it better. The QM region
is the histidine side chain alone, capped at CA: 4-methylimidazole, neutral and
closed shell, the same atoms for both tautomers with one proton moved between
the ring nitrogens. There is no charge to get wrong. The surroundings enter as
fixed MM point charges, taken from the same ff14SB and TIP3P parameters tleap
will later use, so the environment is represented without being asked to be a
valid molecule.

The link atoms sit at CA, four bonds from either ring nitrogen, and the MM
environment is identical for the two tautomers. Whatever the caps get wrong
they get wrong equally in both, and the energy difference that decides the
tautomer is left alone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from mdprep.protonation.histidine_geometry import (
    TautomerClusterModel,
    XyzAtom,
    build_tautomer_cluster_model,
    format_constraint_block,
)
from mdprep.structure.models import PdbStructure, ResidueRecord
from mdprep.structure.pdb import ATOMIC_NUMBER


class HistidineQmmmError(ValueError):
    """Raised when an embedded QM/MM histidine model cannot be built."""


# xTB reads point-charge positions in bohr, which is not stated in its help
# output. Verified against the classical interaction of a +1 charge with GFN2's
# own partial charges on water; see tests/test_histidine_qmmm_embedding.py.
BOHR_PER_ANGSTROM = 1.0 / 0.52917721067

# The force field the MM charges come from, which is the one tleap uses for the
# protein and the water, so the embedding and the final system agree.
PROTEIN_FFXML = "amber14/protein.ff14SB.xml"
WATER_AND_ION_FFXML = "amber14/tip3p.xml"


@dataclass(frozen=True)
class EmbeddedPointCharge:
    charge: float
    x: float
    y: float
    z: float
    element: str
    residue: str
    atom_name: str


@dataclass(frozen=True)
class QmmmTautomerModel:
    """A histidine side chain in QM, everything else as fixed point charges."""

    tautomer: str
    qm_atoms: list[XyzAtom]
    fixed_atom_indices: list[int]
    point_charges: list[EmbeddedPointCharge]
    warnings: list[str]
    untemplated_residues: list[str]
    link_excluded_atoms: list[str]

    @property
    def qm_charge(self) -> int:
        """Always zero: a capped imidazole side chain is neutral."""
        return 0

    @property
    def qm_unpaired_electrons(self) -> int:
        """Always zero: a capped imidazole side chain is closed shell."""
        return 0

    @property
    def embedded_charge_total(self) -> float:
        return sum(charge.charge for charge in self.point_charges)

    def to_dict(self) -> dict[str, object]:
        return {
            "tautomer": self.tautomer,
            "qm_atom_count": len(self.qm_atoms),
            "qm_charge": self.qm_charge,
            "qm_unpaired_electrons": self.qm_unpaired_electrons,
            "qm_formula": _formula(self.qm_atoms),
            "point_charge_count": len(self.point_charges),
            "embedded_charge_total": round(self.embedded_charge_total, 6),
            "fixed_atom_indices": self.fixed_atom_indices,
            "untemplated_residues": self.untemplated_residues,
            "link_excluded_atoms": self.link_excluded_atoms,
            "warnings": self.warnings,
        }


@lru_cache(maxsize=1)
def residue_charge_templates() -> dict[str, dict[str, float]]:
    """Per-atom charges keyed by Amber residue state and PDB atom name.

    OpenMM's ffxml files carry the charge on the residue template itself, so
    this is a direct lookup and needs no topology, no bond perception and no
    complete residue -- which matters because it runs before anything has been
    parameterised.
    """

    try:
        import openmm.app
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise HistidineQmmmError(
            "Embedded QM/MM histidine selection needs OpenMM for its MM point "
            "charges; install the mdprep MD optional dependencies, or set "
            "protonation.histidine.xtb.embedding: cluster."
        ) from exc

    data_dir = Path(openmm.app.__file__).parent / "data"
    templates: dict[str, dict[str, float]] = {}
    for relative in (PROTEIN_FFXML, WATER_AND_ION_FFXML):
        path = data_dir / relative
        if not path.exists():
            raise HistidineQmmmError(f"OpenMM force field file not found: {path}")
        root = ET.parse(path).getroot()
        for residue in root.findall("./Residues/Residue"):
            name = residue.get("name")
            if name is None:
                continue
            charges: dict[str, float] = {}
            for atom in residue.findall("Atom"):
                atom_name = atom.get("name")
                charge = atom.get("charge")
                if atom_name is None or charge is None:
                    continue
                charges[atom_name.upper()] = float(charge)
            if charges:
                templates.setdefault(name.upper(), charges)
    return templates


def _formula(atoms: list[XyzAtom]) -> str:
    counts: dict[str, int] = {}
    for atom in atoms:
        element = atom.element.capitalize()
        counts[element] = counts.get(element, 0) + 1
    return "".join(
        f"{element}{counts[element] if counts[element] > 1 else ''}"
        for element in sorted(counts)
    )


WATER_RESNAMES = {"WAT", "HOH", "TIP3", "TP3", "SOL"}


def _template_names(residue: ResidueRecord, state: str | None) -> list[str]:
    """Candidate ff14SB template names for a residue, best first.

    Terminal residues carry their own templates because their charges differ:
    a C-terminal carboxylate and an N-terminal ammonium are not the charges the
    internal template gives. They are recognised from the atoms themselves --
    OXT for the C-terminus, a second and third backbone amide hydrogen for the
    N-terminus -- since nothing has told us where the chains end.
    """

    name = (state or residue.id.resname).strip().upper()
    if name in WATER_RESNAMES:
        return ["HOH"]

    atom_names = {atom.name.strip().upper() for atom in residue.atoms}
    candidates: list[str] = []
    if "OXT" in atom_names:
        candidates.append(f"C{name}")
    if {"H2", "H3"} <= atom_names:
        candidates.append(f"N{name}")
    candidates.append(name)
    if name in {"HIS", "HID", "HIE"}:
        # A neutral histidine's state has to match the proton it actually
        # carries, not the one it has been assigned. While the tautomers are
        # being decided, a neighbour's planned state can run ahead of its
        # coordinates; embedding it on the planned state would then look up a
        # proton that is not there and silently drop its charge. HIP is left
        # alone because its charge, not its geometry, is what matters.
        resolved = _histidine_state_from_hydrogens(atom_names)
        candidates = [
            candidate.replace(name, resolved) for candidate in candidates
        ] + candidates
    return candidates


def _histidine_state_from_hydrogens(atom_names: set[str]) -> str:
    """Which histidine state the ring protons present describe."""

    on_nd1 = "HD1" in atom_names
    on_ne2 = "HE2" in atom_names
    if on_nd1 and on_ne2:
        return "HIP"
    if on_nd1:
        return "HID"
    return "HIE"


def _water_atom_name(name: str) -> str:
    """TIP3P names its hydrogens H1 and H2; PDB water is less consistent."""
    upper = name.strip().upper()
    if upper in {"OW", "O"}:
        return "O"
    if upper in {"HW1", "H1", "1H"}:
        return "H1"
    if upper in {"HW2", "H2", "2H"}:
        return "H2"
    return upper


def _lookup_name(atom_name: str, *, template: str, is_water: bool) -> str:
    if is_water:
        return _water_atom_name(atom_name)
    upper = atom_name.strip().upper()
    # An N-terminal ammonium is H1/H2/H3 in ff14SB, but PDB writers often leave
    # the first one spelled H, as it is on every internal residue.
    if template.startswith("N") and upper == "H":
        return "H1"
    return upper


def residue_point_charges(
    residue: ResidueRecord,
    *,
    state: str | None = None,
    ion_charge: float | None = None,
    atom_charges: list[float] | None = None,
) -> tuple[list[EmbeddedPointCharge], str | None]:
    """Point charges for one residue, or a reason it has none.

    Returns an empty list and a reason when the residue has no template, which
    is the honest answer for a cofactor whose charges nothing has computed yet.
    An ``ion_charge`` from the manifest overrides the force field's default
    oxidation state, because the manifest is where the user declares it, and
    ``atom_charges`` supplies charges the force field has no template for at
    all -- a declared cofactor, computed elsewhere.
    """

    if atom_charges is not None:
        if len(atom_charges) != len(residue.atoms):
            return [], (
                f"{residue.id.display()} was given {len(atom_charges)} charges "
                f"for {len(residue.atoms)} atoms"
            )
        return [
            EmbeddedPointCharge(
                charge=float(charge),
                x=atom.x,
                y=atom.y,
                z=atom.z,
                element=(atom.element or "").strip() or "H",
                residue=residue.id.display(),
                atom_name=atom.name.strip(),
            )
            for atom, charge in zip(residue.atoms, atom_charges)
        ], None

    if len(residue.atoms) == 1:
        return _monatomic_ion_charge(residue, ion_charge)

    templates = residue_charge_templates()
    candidates = _template_names(residue, state)
    table: dict[str, float] | None = None
    template_name = candidates[-1]
    for candidate in candidates:
        found = templates.get(candidate)
        if found is not None:
            table, template_name = found, candidate
            break
    if table is None:
        return [], f"{residue.id.display()} has no {candidates[-1]} template"

    is_water = template_name == "HOH"
    charges: list[EmbeddedPointCharge] = []
    unknown: list[str] = []
    matched: set[str] = set()
    for atom in residue.atoms:
        lookup = _lookup_name(atom.name, template=template_name, is_water=is_water)
        if lookup not in table:
            unknown.append(atom.name.strip())
            continue
        matched.add(lookup)
        charges.append(
            EmbeddedPointCharge(
                charge=table[lookup],
                x=atom.x,
                y=atom.y,
                z=atom.z,
                element=(atom.element or "").strip() or "H",
                residue=residue.id.display(),
                atom_name=atom.name.strip(),
            )
        )
    reasons: list[str] = []
    if unknown:
        reasons.append(
            f"has no charge for {', '.join(sorted(unknown))}"
        )
    # An atom the template has but the residue does not takes its charge with
    # it, which is how an embedding ends up with a fractional total. A repaired
    # structure should have none of these.
    absent = sorted(name for name in table if name not in matched)
    if absent:
        reasons.append(f"is missing {', '.join(absent)}")
    if reasons:
        return charges, (
            f"{residue.id.display()} ({template_name}) " + " and ".join(reasons)
        )
    return charges, None


def _monatomic_ion_charge(
    residue: ResidueRecord,
    ion_charge: float | None,
) -> tuple[list[EmbeddedPointCharge], str | None]:
    """A one-atom residue is an ion, and its charge is the whole residue's.

    Matching is by residue name alone, not by atom name: the PDB calls the iron
    of an FE2 residue simply FE, while the force field names the atom after the
    oxidation state. With one atom on each side there is nothing to pair
    wrongly.
    """

    atom = residue.atoms[0]
    element = (atom.element or atom.name).strip() or "H"
    charge = ion_charge
    if charge is None:
        table = residue_charge_templates().get(residue.id.resname.strip().upper())
        if table is None or len(table) != 1:
            return [], (
                f"{residue.id.display()} is a single atom with no ion template; "
                "declare it under metals, or under ligands if it is not an ion"
            )
        charge = next(iter(table.values()))
    return [
        EmbeddedPointCharge(
            charge=float(charge),
            x=atom.x,
            y=atom.y,
            z=atom.z,
            element=element,
            residue=residue.id.display(),
            atom_name=atom.name.strip(),
        )
    ], None


def build_qmmm_tautomer_model(
    structure: PdbStructure,
    histidine: ResidueRecord,
    *,
    tautomer: str,
    residue_states: dict[int, str] | None = None,
    cutoff_angstrom: float = 12.0,
    link_exclusion_angstrom: float = 1.8,
    ion_charges: dict[int, float] | None = None,
    cofactor_atom_charges: dict[int, list[float]] | None = None,
    add_missing_water_hydrogens: bool = True,
    water_oh_distance_angstrom: float = 0.9572,
    water_hoh_angle_degrees: float = 104.52,
) -> QmmmTautomerModel:
    """Build the QM side chain and the MM charges that surround it."""

    states = {} if residue_states is None else residue_states
    qm_model: TautomerClusterModel = build_tautomer_cluster_model(
        [histidine],
        histidine,
        tautomer=tautomer,
        residue_states=states,
        add_missing_water_hydrogens=add_missing_water_hydrogens,
        water_oh_distance_angstrom=water_oh_distance_angstrom,
        water_hoh_angle_degrees=water_hoh_angle_degrees,
    )
    electrons = sum(
        ATOMIC_NUMBER[atom.element.strip().upper()] for atom in qm_model.atoms
    )
    if electrons % 2 != 0:
        raise HistidineQmmmError(
            f"The {tautomer} side chain of {histidine.id.display()} came out as "
            f"{_formula(qm_model.atoms)} with {electrons} electrons, which is "
            "an open shell. A capped histidine side chain is neutral and "
            "closed shell, so the residue is missing an atom or carries one it "
            "should not."
        )

    warnings = list(qm_model.warnings)
    untemplated: list[str] = []
    point_charges: list[EmbeddedPointCharge] = []
    for residue in _residues_within(structure, histidine, cutoff_angstrom):
        charges, reason = residue_point_charges(
            residue,
            state=states.get(id(residue)),
            ion_charge=(ion_charges or {}).get(id(residue)),
            atom_charges=(cofactor_atom_charges or {}).get(id(residue)),
        )
        if reason is not None:
            untemplated.append(reason)
        point_charges.extend(charges)

    point_charges, excluded = _drop_charges_on_top_of_link_atoms(
        point_charges,
        qm_model,
        link_exclusion_angstrom=link_exclusion_angstrom,
    )
    total = sum(charge.charge for charge in point_charges)
    if abs(total - round(total)) > 1.0e-4:
        # Every residue carries an integral charge, so the embedding's total
        # must be integral too. A fractional total means atoms were dropped
        # from a residue that was otherwise kept, which is exactly the kind of
        # silent partial-residue error the whole-residue cutoff exists to
        # prevent.
        warnings.append(
            f"The MM embedding around {histidine.id.display()} sums to "
            f"{total:+.4f}, which is not an integral charge. Some residue "
            "contributed only part of its atoms."
        )
    if untemplated:
        warnings.append(
            "Some neighbours of "
            f"{histidine.id.display()} contributed no MM point charges and are "
            "absent from the embedding: " + "; ".join(sorted(untemplated)) + ". "
            "Declare them under ligands or metals if they sit near this "
            "histidine."
        )
    return QmmmTautomerModel(
        tautomer=tautomer,
        qm_atoms=qm_model.atoms,
        fixed_atom_indices=qm_model.fixed_atom_indices,
        point_charges=point_charges,
        warnings=warnings,
        untemplated_residues=sorted(untemplated),
        link_excluded_atoms=excluded,
    )


def _residues_within(
    structure: PdbStructure,
    histidine: ResidueRecord,
    cutoff_angstrom: float,
) -> list[ResidueRecord]:
    """Whole residues with any atom inside the cutoff, the histidine excluded.

    Whole residues, because a cutoff that cuts through one leaves behind a
    fragment carrying a net charge that the residue does not have -- the same
    bookkeeping mistake the cluster backend makes, reintroduced through the
    back door.
    """

    cutoff_squared = cutoff_angstrom * cutoff_angstrom
    anchors = [(atom.x, atom.y, atom.z) for atom in histidine.atoms]
    near: list[ResidueRecord] = []
    for residue in structure.residues:
        if residue is histidine or residue.id == histidine.id:
            continue
        if any(
            (atom.x - ax) ** 2 + (atom.y - ay) ** 2 + (atom.z - az) ** 2
            <= cutoff_squared
            for atom in residue.atoms
            for ax, ay, az in anchors
        ):
            near.append(residue)
    return near


def _drop_charges_on_top_of_link_atoms(
    charges: list[EmbeddedPointCharge],
    qm_model: TautomerClusterModel,
    *,
    link_exclusion_angstrom: float,
) -> tuple[list[EmbeddedPointCharge], list[str]]:
    """Remove MM charges sitting within a bond length of a capping hydrogen.

    A cap hydrogen stands in for the backbone bond that was cut, so the real
    atoms it replaced are close enough that their point charges would drag
    electron density onto the cap. Their charge is redistributed over the rest
    of their own residue, which keeps that residue's total charge intact rather
    than quietly removing it from the system.
    """

    caps = [qm_model.atoms[index - 1] for index in qm_model.cap_atom_indices]
    if not caps:
        return charges, []

    cutoff_squared = link_exclusion_angstrom * link_exclusion_angstrom
    kept: list[EmbeddedPointCharge] = []
    excluded: list[str] = []
    orphaned: dict[str, float] = {}
    for charge in charges:
        too_close = any(
            (charge.x - cap.x) ** 2
            + (charge.y - cap.y) ** 2
            + (charge.z - cap.z) ** 2
            <= cutoff_squared
            for cap in caps
        )
        if too_close:
            excluded.append(f"{charge.residue}:{charge.atom_name}")
            orphaned[charge.residue] = orphaned.get(charge.residue, 0.0) + charge.charge
            continue
        kept.append(charge)

    if not orphaned:
        return kept, excluded

    counts: dict[str, int] = {}
    for charge in kept:
        if charge.residue in orphaned:
            counts[charge.residue] = counts.get(charge.residue, 0) + 1
    redistributed = [
        EmbeddedPointCharge(
            charge=charge.charge
            + (
                orphaned[charge.residue] / counts[charge.residue]
                if charge.residue in counts
                else 0.0
            ),
            x=charge.x,
            y=charge.y,
            z=charge.z,
            element=charge.element,
            residue=charge.residue,
            atom_name=charge.atom_name,
        )
        for charge in kept
    ]
    return redistributed, sorted(excluded)


def write_point_charge_file(
    charges: list[EmbeddedPointCharge],
    path: str | Path,
) -> None:
    """Write xTB's ``pcharge`` file: count, then charge, position, element.

    Positions go out in bohr, which is what xTB reads them as.
    """

    lines = [str(len(charges))]
    for charge in charges:
        element = charge.element.strip().upper() or "H"
        atomic_number = ATOMIC_NUMBER.get(element, 1)
        lines.append(
            f"{charge.charge: .10f} "
            f"{charge.x * BOHR_PER_ANGSTROM: .10f} "
            f"{charge.y * BOHR_PER_ANGSTROM: .10f} "
            f"{charge.z * BOHR_PER_ANGSTROM: .10f} "
            f"{atomic_number:d}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_embedding_xcontrol(
    path: str | Path,
    *,
    point_charge_filename: str = "pcharge",
    fixed_atom_indices: list[int] | None = None,
    constraint_force_constant: float = 1.0,
    embedding: bool = True,
) -> None:
    """Write the xcontrol file that turns electrostatic embedding on.

    ``embedding`` is false when there are no point charges to embed, because
    xTB rejects an empty ``pcharge`` file rather than treating it as no field.
    """

    blocks = []
    if fixed_atom_indices:
        blocks.append(
            format_constraint_block(fixed_atom_indices, constraint_force_constant)
        )
    if embedding:
        blocks.append(f"$embedding\n input={point_charge_filename}\n es=true\n$end")
    Path(path).write_text("\n".join(blocks) + "\n", encoding="utf-8")


def embedding_was_applied(stdout: str) -> bool:
    """Whether xTB reported reading the point charges.

    Silence here would mean the whole embedding had been ignored and both
    tautomers evaluated in vacuum, which would still produce an answer.
    """

    return "point charge" in stdout.lower()
