"""Transplanting waters with PyMOL doing the pairing, on a real deposition.

6NIE (BesD) has an iron on the surface of chain B with three waters on it.
Chain B alone, its waters stripped and the whole thing moved somewhere else,
plays the predicted model; the full deposition is the reference. The waters
bound to that iron have to come back onto it at the distance the crystal
has, whatever the model's frame.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymol")

from mdprep.structure.pdb import read_pdb  # noqa: E402
from mdprep.structure.waters import transplant_waters  # noqa: E402
from mdprep.structure.writer import write_pdb  # noqa: E402

pytestmark = [pytest.mark.external, pytest.mark.pymol]

DEPOSITION = Path("examples/tutorials/besd_6nie/6NIE.pdb")


def _moved_chain_b(path: Path) -> Path:
    structure = read_pdb(DEPOSITION)
    angle = np.radians(70.0)
    rotation = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]]
    )
    shift = np.array([25.0, -40.0, 12.0])
    atoms = []
    for atom in structure.atoms:
        if atom.chain_id != "B" or atom.resname == "HOH":
            continue
        x, y, z = np.array((atom.x, atom.y, atom.z)) @ rotation.T + shift
        atoms.append(replace(atom, x=float(x), y=float(y), z=float(z)))
    write_pdb(replace(structure, atoms=atoms, residues=[], conect_bonds=set()), path)
    return path


def test_waters_on_a_metal_come_back_onto_it(tmp_path):
    model = _moved_chain_b(tmp_path / "chain_b.pdb")

    result = transplant_waters(model, DEPOSITION, tmp_path / "with_waters.pdb")

    [chain_fit] = result.fits
    assert chain_fit.reference_chain == "B"
    assert chain_fit.superposition.rmsd < 0.01
    assert chain_fit.superposition.tm_score > 0.95
    bound = {entry["water"]: entry for entry in result.selection.metal_bound}
    assert {"B:HOH412", "B:HOH464"} <= set(bound)
    assert bound["B:HOH412"]["distance_angstrom"] == pytest.approx(2.24, abs=0.02)
    assert bound["B:HOH412"]["metal"].startswith("B:FE305")

    written = read_pdb(tmp_path / "with_waters.pdb")
    added = [residue for residue in written.residues if residue.id.resname == "HOH"]
    assert len(added) == len(result.selection.kept) > 50
    assert {residue.id.chain_id for residue in added} == {"B"}
