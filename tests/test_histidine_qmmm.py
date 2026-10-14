"""Embedded QM/MM histidine tautomer selection."""

from pathlib import Path

import pytest

from mdprep.protonation.histidine_qmmm import (
    BOHR_PER_ANGSTROM,
    HistidineQmmmError,
    build_qmmm_tautomer_model,
    residue_charge_templates,
    residue_point_charges,
    write_embedding_xcontrol,
    write_point_charge_file,
)
from mdprep.structure.pdb import ATOMIC_NUMBER, read_pdb


@pytest.fixture
def templates():
    """The ff14SB and TIP3P charge tables, which OpenMM ships.

    OpenMM is an optional dependency, so a run without it skips these rather
    than failing. The embedding itself refuses to proceed instead of guessing;
    that refusal is asserted in test_the_missing_dependency_is_reported.
    """

    pytest.importorskip(
        "openmm",
        reason="the MM point charges are read from OpenMM's force field files",
    )
    return residue_charge_templates()


def _structure():
    return read_pdb("tests/data/protein_histidine_ring_hydrogenated.pdb")


def _histidine(structure):
    return next(r for r in structure.residues if r.id.resname == "HIS")


def test_qm_region_is_neutral_closed_shell_for_both_tautomers():
    """The point of embedding: there is no cluster charge to get wrong."""

    structure = _structure()
    histidine = _histidine(structure)
    formulas = set()
    for tautomer in ("HID", "HIE"):
        model = build_qmmm_tautomer_model(structure, histidine, tautomer=tautomer)
        electrons = sum(
            ATOMIC_NUMBER[atom.element.strip().upper()] for atom in model.qm_atoms
        )
        assert model.qm_charge == 0
        assert model.qm_unpaired_electrons == 0
        assert electrons % 2 == 0
        formulas.add(model.to_dict()["qm_formula"])

    # Both tautomers are the same molecule with one proton moved, so the
    # energy difference cannot be an artefact of a different QM region.
    assert formulas == {"C5H8N2"}


def test_point_charge_positions_are_written_in_bohr(templates):
    """xTB reads pcharge positions in bohr, which its help output does not say."""

    structure = _structure()
    histidine = _histidine(structure)
    model = build_qmmm_tautomer_model(structure, histidine, tautomer="HID")
    charges, _ = residue_point_charges(histidine)
    assert charges, "the fixture histidine should have ff14SB charges"

    path = Path("/tmp") / "pcharge_units_check"
    write_point_charge_file(charges, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == str(len(charges))
    first = lines[1].split()
    assert float(first[1]) == pytest.approx(charges[0].x * BOHR_PER_ANGSTROM)
    assert float(first[1]) != pytest.approx(charges[0].x)
    assert int(first[4]) == ATOMIC_NUMBER[charges[0].element.strip().upper()]
    path.unlink()
    assert model.qm_charge == 0


def test_charges_come_from_the_force_field_tleap_will_use(templates):
    # ff14SB, checked against the published Amber charges.
    assert templates["ASP"]["OD1"] == pytest.approx(-0.8014)
    assert templates["ARG"]["NH1"] == pytest.approx(-0.8627)
    # TIP3P, from the matching water file.
    assert templates["HOH"]["O"] == pytest.approx(-0.834)
    assert templates["HOH"]["H1"] == pytest.approx(0.417)
    # Monatomic ions live in the same file.
    assert templates["CL"]["CL"] == pytest.approx(-1.0)
    assert templates["FE2"]["FE2"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "resname,formal_charge",
    [("ALA", 0), ("ASP", -1), ("GLU", -1), ("ARG", 1), ("LYS", 1), ("HIP", 1),
     ("HID", 0), ("HIE", 0), ("HOH", 0), ("CASP", -2), ("NLYS", 2)],
)
def test_a_complete_residue_contributes_its_formal_charge(templates, resname, formal_charge):
    """Whole residues carry integral charges, which is why the cutoff keeps
    them whole. A fractional embedding total means atoms went missing."""

    from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord

    template = templates[resname]
    pdb_resname = {"HOH": "HOH"}.get(resname, resname)
    atoms = [
        AtomRecord(
            serial=index, name=name, altloc=None, resname=pdb_resname, chain_id="A",
            resid=1, icode=None, x=float(index), y=0.0, z=0.0, occupancy=1.0,
            bfactor=0.0, element=name[0], record_name="ATOM", original_line="",
        )
        for index, name in enumerate(sorted(template), start=1)
    ]
    residue = ResidueRecord(
        id=ResidueId(chain_id="A", resname=pdb_resname, resid=1, icode=None),
        atoms=atoms, record_names={"ATOM"}, original_index=0,
    )

    charges, reason = residue_point_charges(residue, state=resname)
    assert reason is None, reason
    assert sum(charge.charge for charge in charges) == pytest.approx(
        formal_charge, abs=1e-4
    )


def test_a_residue_missing_an_atom_is_reported_not_silently_discharged(templates):
    """The fixture histidine has no ring N-H, so a template atom is absent.

    Its charge would otherwise vanish into a fractional embedding total with
    nothing to explain it.
    """

    structure = _structure()
    charges, reason = residue_point_charges(_histidine(structure))
    assert reason is not None and "missing" in reason and "HE2" in reason
    total = sum(charge.charge for charge in charges)
    assert total != pytest.approx(round(total), abs=1e-4)


def test_monatomic_ion_is_matched_by_residue_not_atom_name(templates):
    """A PDB FE2 residue names its atom FE; the force field names it FE2."""

    from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord

    atom = AtomRecord(
        serial=1, name="FE", altloc=None, resname="FE2", chain_id="A", resid=302,
        icode=None, x=0.0, y=0.0, z=0.0, occupancy=1.0, bfactor=0.0, element="Fe",
        record_name="HETATM", original_line="",
    )
    residue = ResidueRecord(
        id=ResidueId(chain_id="A", resname="FE2", resid=302, icode=None),
        atoms=[atom], record_names={"HETATM"}, original_index=0,
    )

    charges, reason = residue_point_charges(residue)
    assert reason is None
    assert len(charges) == 1
    assert charges[0].charge == pytest.approx(2.0)

    # A manifest declaring a different oxidation state overrides the default.
    charges, reason = residue_point_charges(residue, ion_charge=3.0)
    assert reason is None
    assert charges[0].charge == pytest.approx(3.0)


def test_undeclared_cofactor_is_reported_rather_than_silently_zeroed(templates):
    from mdprep.structure.models import AtomRecord, ResidueId, ResidueRecord

    atoms = [
        AtomRecord(
            serial=index, name=name, altloc=None, resname="AKG", chain_id="A",
            resid=304, icode=None, x=float(index), y=0.0, z=0.0, occupancy=1.0,
            bfactor=0.0, element="C", record_name="HETATM", original_line="",
        )
        for index, name in enumerate(("C1", "C2", "C3"), start=1)
    ]
    residue = ResidueRecord(
        id=ResidueId(chain_id="A", resname="AKG", resid=304, icode=None),
        atoms=atoms, record_names={"HETATM"}, original_index=0,
    )

    charges, reason = residue_point_charges(residue)
    assert charges == []
    assert reason is not None and "AKG" in reason


def test_xcontrol_omits_the_embedding_block_when_there_is_nothing_to_embed():
    path = Path("/tmp") / "xcontrol_no_embedding_check"
    write_embedding_xcontrol(path, embedding=False)
    text = path.read_text(encoding="utf-8")
    assert "$embedding" not in text
    write_embedding_xcontrol(path, embedding=True, fixed_atom_indices=[1, 2])
    text = path.read_text(encoding="utf-8")
    assert "$embedding" in text and "input=pcharge" in text and "es=true" in text
    assert "$fix" in text and "atoms: 1,2" in text
    path.unlink()


def test_open_shell_side_chain_is_refused(monkeypatch):
    """A backstop: an odd-electron QM region must never reach xTB.

    The geometry builder rejects most ways of producing one, so this drives the
    guard directly rather than relying on that check staying absent.
    """

    from mdprep.protonation import histidine_qmmm

    structure = _structure()
    histidine = _histidine(structure)
    good = build_qmmm_tautomer_model(structure, histidine, tautomer="HID")
    hydrogen = next(a for a in good.qm_atoms if a.element.strip().upper() == "H")

    class _OddFragment:
        # C5H8N2 plus one more hydrogen: 53 electrons.
        atoms = [*good.qm_atoms, hydrogen]
        fixed_atom_indices: list[int] = []
        cap_atom_indices: list[int] = []
        warnings: list[str] = []

    monkeypatch.setattr(
        histidine_qmmm, "build_tautomer_cluster_model", lambda *a, **k: _OddFragment()
    )
    with pytest.raises(HistidineQmmmError, match="open shell"):
        histidine_qmmm.build_qmmm_tautomer_model(
            structure, histidine, tautomer="HID"
        )


def test_the_missing_dependency_is_reported(monkeypatch):
    """Without OpenMM the embedding refuses rather than guessing charges.

    Falling back to the cluster backend on its own would quietly return to the
    charge arithmetic this replaces, so the error names the option instead.
    """

    import builtins

    real_import = builtins.__import__

    def no_openmm(name, *args, **kwargs):
        if name == "openmm" or name.startswith("openmm."):
            raise ImportError("No module named 'openmm'")
        return real_import(name, *args, **kwargs)

    residue_charge_templates.cache_clear()
    monkeypatch.setattr(builtins, "__import__", no_openmm)
    try:
        with pytest.raises(HistidineQmmmError, match="embedding: cluster"):
            residue_charge_templates()
    finally:
        monkeypatch.undo()
        residue_charge_templates.cache_clear()
