"""Regression tests for the Roe--Brooks restraint selection and coordinate frame.

Two defects made stage 2 die with "Particle coordinate is NaN" on a solvated
system: free ions were positionally restrained because Amber names them ``Na+``
and ``Cl-`` rather than ``NA``/``CL``, and the coordinates handed from one stage
to the next were wrapped into the periodic box while the restraint references
stayed unwrapped, so every wrapped restrained particle was pulled a full box
length toward its reference.
"""

from dataclasses import dataclass, field

import pytest

from mdprep.md.roe_brooks import (
    RoeBrooksError,
    monatomic_ion_atom_indices,
    periodic_output_state,
    restrained_atom_indices,
    stage_handoff_state,
)


@dataclass
class FakeElement:
    symbol: str


@dataclass
class FakeAtom:
    name: str
    index: int
    element: FakeElement | None
    residue: "FakeResidue" = field(default=None, repr=False)  # type: ignore[assignment]


@dataclass
class FakeResidue:
    name: str
    _atoms: list[FakeAtom] = field(default_factory=list)

    def atoms(self):
        return iter(self._atoms)


class FakeTopology:
    """Minimal stand-in for openmm.app.Topology."""

    def __init__(self, residue_spec: list[tuple[str, list[tuple[str, str]]]]):
        self._residues: list[FakeResidue] = []
        index = 0
        for resname, atom_spec in residue_spec:
            residue = FakeResidue(resname)
            for atom_name, symbol in atom_spec:
                atom = FakeAtom(atom_name, index, FakeElement(symbol) if symbol else None)
                atom.residue = residue
                residue._atoms.append(atom)
                index += 1
            self._residues.append(residue)

    def residues(self):
        return iter(self._residues)

    def atoms(self):
        for residue in self._residues:
            yield from residue.atoms()


def amber_like_topology() -> FakeTopology:
    return FakeTopology(
        [
            ("ALA", [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("HA", "H")]),
            ("LIG", [("C1", "C"), ("O1", "O"), ("H1", "H")]),
            ("Na+", [("Na+", "Na")]),
            ("Cl-", [("Cl-", "Cl")]),
            ("K+", [("K+", "K")]),
            ("Mg2+", [("Mg2+", "Mg")]),
            ("WAT", [("O", "O"), ("H1", "H"), ("H2", "H")]),
        ]
    )


def test_amber_charged_ion_names_are_recognised():
    """`Na+`.upper() is not `NA`, which is why every ion was being restrained."""
    topology = amber_like_topology()

    ion_indices = monatomic_ion_atom_indices(topology)

    names = {
        atom.residue.name for atom in topology.atoms() if atom.index in ion_indices
    }
    assert names == {"Na+", "Cl-", "K+", "Mg2+"}


def test_multi_atom_residue_is_not_mistaken_for_an_ion():
    """A ligand whose name normalises onto an element symbol stays solute."""
    topology = FakeTopology([("CA1", [("C1", "C"), ("C2", "C"), ("O1", "O")])])

    assert monatomic_ion_atom_indices(topology) == set()


def test_ions_hydrogens_and_water_are_not_restrained():
    topology = amber_like_topology()
    ion_indices = monatomic_ion_atom_indices(topology)

    indices = restrained_atom_indices(
        topology, selection="heavy", ion_atom_indices=ion_indices
    )

    by_index = {atom.index: atom for atom in topology.atoms()}
    assert [by_index[i].name for i in indices] == ["N", "CA", "C", "O", "CB", "C1", "O1"]
    assert not ion_indices & set(indices)


def test_backbone_selection_keeps_only_backbone_heavy_atoms():
    topology = amber_like_topology()

    indices = restrained_atom_indices(
        topology,
        selection="backbone",
        ion_atom_indices=monatomic_ion_atom_indices(topology),
    )

    by_index = {atom.index: atom for atom in topology.atoms()}
    assert [by_index[i].name for i in indices] == ["N", "CA", "C", "O"]


def test_unknown_restraint_selection_fails_clearly():
    topology = amber_like_topology()

    with pytest.raises(RoeBrooksError, match="Unknown restraint selection"):
        restrained_atom_indices(topology, selection="sidechain", ion_atom_indices=set())


class RecordingContext:
    """Captures the getState keyword arguments a caller asks for."""

    def __init__(self):
        self.calls: list[dict] = []

    def getState(self, **kwargs):
        self.calls.append(kwargs)
        return object()


def test_stage_handoff_never_wraps_into_the_periodic_box():
    """Wrapping here is what dragged restrained particles a box length away."""
    context = RecordingContext()

    stage_handoff_state(context)

    assert context.calls[0]["enforcePeriodicBox"] is False
    assert context.calls[0]["getPositions"] is True
    assert context.calls[0]["getVelocities"] is True


def test_written_structures_are_wrapped():
    context = RecordingContext()

    periodic_output_state(context)

    assert context.calls[0]["enforcePeriodicBox"] is True
