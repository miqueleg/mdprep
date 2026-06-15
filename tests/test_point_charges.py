import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mdprep.qm.point_charges import PointChargeError, extract_point_charges_from_prmtop
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


class FakeAtom:
    def __init__(self, idx, name, charge, residue):
        self.idx = idx
        self.name = name
        self.charge = charge
        self.residue = residue


class FakeResidue:
    def __init__(self, name, number):
        self.name = name
        self.number = number
        self.atoms = []


def qmmesp_ligand_config(**environment):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    entry = {
        **ligand_entry("sub_501", "B", "SUB", 501),
        "charge_method": "qmmesp_pyscf",
        "qmmesp": {
            "qm_engine": "pyscf",
            "method": "HF",
            "basis": "STO-3G",
            "embedding_cutoff_angstrom": 4.0,
            "resp_fitting": {"backend": "native"},
            "environment": {
                "include_protein": environment.get("include_protein", True),
                "include_waters": environment.get("include_waters", True),
                "include_other_ligands": environment.get("include_other_ligands", True),
                "exclude_self_ligand": True,
            },
        },
    }
    data["ligands"] = [entry]
    manifest = make_manifest(data)
    return manifest, manifest.ligands[0]


def fake_parmed_structure(include_ambiguous=False):
    protein = FakeResidue("ALA", 1)
    water = FakeResidue("WAT", 2)
    ligand = FakeResidue("SUB", 501)
    other = FakeResidue("COF", 601)
    residues = [protein, water, ligand, other]
    if include_ambiguous:
        residues.append(FakeResidue("SUB", 999))
    atoms = []
    for residue, atom_name, charge, coord in [
        (protein, "CA", 0.2, [0.0, 0.0, 0.0]),
        (water, "O", -0.8, [8.0, 0.0, 0.0]),
        (ligand, "C1", 0.1, [5.0, 5.0, 5.0]),
        (ligand, "O1", -0.1, [6.0, 5.0, 5.0]),
        (other, "N1", -0.2, [5.5, 6.0, 5.0]),
    ]:
        atom = FakeAtom(len(atoms), atom_name, charge, residue)
        residue.atoms.append(atom)
        atoms.append(atom)
    if include_ambiguous:
        residue = residues[-1]
        atom = FakeAtom(len(atoms), "C1", 0.0, residue)
        residue.atoms.append(atom)
        atoms.append(atom)
    return SimpleNamespace(
        residues=residues,
        atoms=atoms,
        coordinates=np.asarray([atom_coord for atom_coord in [
            [0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [5.0, 5.0, 5.0],
            [6.0, 5.0, 5.0],
            [5.5, 6.0, 5.0],
            [10.0, 10.0, 10.0],
        ][: len(atoms)]], dtype=float),
    )


def install_fake_parmed(monkeypatch, structure):
    monkeypatch.setitem(sys.modules, "parmed", SimpleNamespace(load_file=lambda *args: structure))


def test_point_charges_exclude_target_and_apply_cutoff(monkeypatch):
    manifest, ligand = qmmesp_ligand_config()
    install_fake_parmed(monkeypatch, fake_parmed_structure())

    selection = extract_point_charges_from_prmtop(
        prmtop="fake.prmtop",
        inpcrd="fake.inpcrd",
        ligand=ligand,
        manifest=manifest,
        target_coordinates=np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float),
    )

    assert selection.target_atom_indices == [2, 3]
    assert all(charge.residue_name != "SUB" for charge in selection.point_charges)
    assert selection.categories == {"ligand": 1}


def test_point_charges_respect_environment_excludes(monkeypatch):
    manifest, ligand = qmmesp_ligand_config(include_other_ligands=False)
    install_fake_parmed(monkeypatch, fake_parmed_structure())

    selection = extract_point_charges_from_prmtop(
        prmtop="fake.prmtop",
        inpcrd="fake.inpcrd",
        ligand=ligand,
        manifest=manifest,
        target_coordinates=np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float),
    )

    assert selection.total_after_cutoff == 0


def test_ambiguous_ligand_mapping_fails_clearly(monkeypatch):
    manifest, ligand = qmmesp_ligand_config()
    install_fake_parmed(monkeypatch, fake_parmed_structure(include_ambiguous=True))

    with pytest.raises(PointChargeError, match="uniquely"):
        extract_point_charges_from_prmtop(
            prmtop="fake.prmtop",
            inpcrd="fake.inpcrd",
            ligand=ligand,
            manifest=manifest,
            target_coordinates=np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float),
        )
