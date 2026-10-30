"""QMMESP charges for a noncanonical residue, without PySCF or AmberTools.

The fit itself is exercised in tests/test_external_noncanonical_residues.py.
Here: the manifest block, the RESP constraints, the embedding around a
residue of a chain, where its files go, and the orchestration with every
external step replaced by a fake that checks what it was given.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from mdprep.ambertools.resp import respgen_constraint_text, write_resp_qin
from mdprep.config.decisions import build_manifest, plan_decisions
from mdprep.config.models import ManifestConfig
from mdprep.qm.point_charges import (
    PointCharge,
    PointChargeError,
    PointChargeSelection,
    extract_residue_point_charges,
)
from mdprep.residues.capping import CappedAtom, CappedResidue
from mdprep.residues.charges import BACKBONE_CHARGES, ResidueChargeError, held_backbone_charges
from mdprep.residues.qmmesp import (
    QMMESP_RESIDUE_CONFIRMATION,
    ResidueQmmespError,
    fit_constraints,
)
from mdprep.residues.workflow import (
    ResidueStageError,
    parameterisation_directory,
    provisional_directory,
    residue_directory,
    residue_leap_inputs,
)
from mdprep.workflows.runtime import conda_packages_for
from tests.test_heterogen_groups import walk
from tests.test_noncanonical_residues import SEP57, _manifest, _smiles_block
from tests.test_point_charges import install_fake_parmed

QMMESP = {"method": "HF", "basis": "6-31G*"}


def _qmmesp_block(**changes) -> dict:
    return _smiles_block(**{"parameters": "qmmesp_pyscf", "qmmesp": dict(QMMESP), **changes})


# --------------------------------------------------------------------------
# The manifest block
# --------------------------------------------------------------------------


def test_a_qmmesp_block_validates():
    manifest = ManifestConfig.model_validate(_manifest([_qmmesp_block()]))

    (block,) = manifest.noncanonical_residues
    assert (block.qmmesp.method, block.qmmesp.basis) == ("HF", "6-31G*")


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (_smiles_block(parameters="qmmesp_pyscf"), "requires qmmesp"),
        (_smiles_block(qmmesp=dict(QMMESP)), "qmmesp belongs to parameters: qmmesp_pyscf"),
        (_qmmesp_block(net_charge=None), "requires smiles and net_charge"),
        (_qmmesp_block(qmmesp={**QMMESP, "scf_charge": -1}), "scf_charge must equal net_charge"),
        (_qmmesp_block(qmmesp={**QMMESP, "scf_spin": 1}), "closed shell"),
        (
            {"resname": "SEP", "residues": [SEP57], "parameters": "amber_library",
             "template": "SEP", "qmmesp": dict(QMMESP)},
            "a template brings its own chemistry",
        ),
    ],
)
def test_a_qmmesp_block_that_says_two_things_is_refused(block, message):
    with pytest.raises(ValidationError, match=message):
        ManifestConfig.model_validate(_manifest([block]))


def test_ff19sb_needs_a_parent_for_a_qmmesp_residue_too():
    block = _qmmesp_block()
    del block["parent"]

    with pytest.raises(ValidationError, match="needs a parent with ff19SB"):
        ManifestConfig.model_validate(_manifest([block], forcefield="ff19SB"))


# --------------------------------------------------------------------------
# RESP with the backbone held and the caps at zero
# --------------------------------------------------------------------------


def test_respgen_is_told_the_fixed_charges_and_the_groups():
    text = respgen_constraint_text(
        ["CAY", "CY", "N", "H"],
        fixed_charges={3: 0.2719, 2: -0.4157},
        charge_groups=[([0, 1], 0.0)],
    )

    assert text.splitlines()[1:] == [
        "CHARGE\t-0.415700\t3\tN",
        "CHARGE\t 0.271900\t4\tH",
        "//group 1: GROUP number_of_atoms net_charge, then its atoms",
        "GROUP\t2\t 0.00000",
        "ATOM\t1\tCAY",
        "ATOM\t2\tCY",
    ]


def test_resp_starts_from_eight_charges_a_line(tmp_path):
    path = write_resp_qin(tmp_path / "qin", [0.1] * 9)

    lines = path.read_text().splitlines()
    assert [len(line) for line in lines] == [80, 10]
    assert lines[1] == "  0.100000"


def _fitted(residue: dict[str, float], caps=(("ACE", "CY", 0.0), ("NME", "NT", 0.0))):
    return [*((role, name, charge) for role, name, charge in caps), *(("RES", n, c) for n, c in residue.items())]


def test_a_fit_that_held_the_backbone_is_kept_as_it_is():
    held = dict(BACKBONE_CHARGES["anionic"])
    residue = {**held, "CA": 0.1, "HA": 0.05, "CB": -1.0, "OG": -0.82, "P": 1.2, "O1P": -0.9, "O2P": -0.9}
    residue["CA"] += -2.0 - sum(residue.values()) + 0.0000004

    charges = held_backbone_charges(_fitted(residue), -2)

    assert {name: charges.charges[name] for name in held} == held
    assert round(sum(charges.charges.values()), 6) == -2.0
    assert charges.shifted_atoms == () and charges.backbone_set == "anionic"


@pytest.mark.parametrize(
    ("change", "message"),
    [({"N": -0.5}, "did not hold the backbone"), ({"CB": -0.9}, "the caps did not keep to zero")],
)
def test_a_fit_that_did_not_hold_is_refused(change, message):
    residue = {**BACKBONE_CHARGES["neutral"], "CA": 0.0, "CB": 0.0}
    residue["CA"] = -sum(residue.values())
    residue.update(change)

    with pytest.raises(ResidueChargeError, match=message):
        held_backbone_charges(_fitted(residue), 0)


def _capped(names_by_role) -> CappedResidue:
    atoms = [
        CappedAtom(role=role, name=name, element=name[0], x=float(index), y=0.0, z=0.0)
        for index, (role, name) in enumerate(names_by_role)
    ]
    return CappedResidue(molecule=None, atoms=atoms, bonds=[], formal_charge=0, smiles_form="capped")


def test_the_fit_holds_the_backbone_of_the_residues_sign_and_zeroes_each_cap():
    capped = _capped(
        [("ACE", "CAY"), ("ACE", "CY"), ("ACE", "OY"), ("RES", "N"), ("RES", "CA"),
         ("RES", "C"), ("RES", "O"), ("NME", "NT"), ("NME", "CAT"), ("RES", "H"),
         ("NME", "HNT")]
    )

    fixed, groups = fit_constraints(capped, +1)

    cationic = BACKBONE_CHARGES["cationic"]
    assert fixed == {3: cationic["N"], 5: cationic["C"], 6: cationic["O"], 9: cationic["H"]}
    assert groups == [([0, 1, 2], 0.0), ([7, 8, 10], 0.0)]


def test_a_capped_residue_without_its_amide_hydrogen_has_no_fit():
    capped = _capped([("ACE", "CY"), ("RES", "N"), ("RES", "CA"), ("RES", "C"), ("RES", "O"), ("NME", "NT")])

    with pytest.raises(ResidueQmmespError, match="has no H"):
        fit_constraints(capped, 0)


# --------------------------------------------------------------------------
# The embedding around a residue of a chain
# --------------------------------------------------------------------------


class _Residue:
    def __init__(self, idx, name, number):
        self.idx, self.name, self.number, self.atoms = idx, name, number, []


class _Atom:
    def __init__(self, idx, name, charge, residue):
        self.idx, self.name, self.charge, self.residue = idx, name, charge, residue
        self.bond_partners = []
        residue.atoms.append(self)


def _bond(first: _Atom, second: _Atom) -> None:
    first.bond_partners.append(second)
    second.bond_partners.append(first)


def _chain(*, extra_link=False, kept_name="TPO"):
    """MET - ALA - SEP - GLY - LYS in a chain, a water, another kept residue, an ion."""

    residues = [
        _Residue(0, "MET", 1), _Residue(1, "ALA", 2), _Residue(2, "SEP", 3),
        _Residue(3, "GLY", 4), _Residue(4, "LYS", 5), _Residue(5, "WAT", 6),
        _Residue(6, kept_name, 7), _Residue(7, "MG", 8),
    ]
    met, ala, sep, gly, lys, water, kept, ion = residues
    placed = [
        (met, "CA", 0.02, [-2.0, -3.0, 0.0]), (met, "HA", 0.09, [-2.5, -3.8, 0.0]),
        (met, "C", 0.6, [-1.0, -2.0, 0.0]),
        (met, "O", -0.57, [-1.0, -3.2, 0.0]),
        (ala, "N", -0.42, [-0.5, -1.0, 0.0]), (ala, "CA", 0.03, [0.0, 0.0, 0.0]),
        (ala, "C", 0.6, [1.0, 0.0, 0.0]),
        (sep, "N", -0.5, [2.0, 0.0, 0.0]), (sep, "CA", 0.1, [3.0, 0.0, 0.0]),
        (sep, "C", 0.5, [4.0, 0.0, 0.0]), (sep, "P", 1.2, [3.0, 1.5, 0.0]),
        (gly, "N", -0.4, [5.0, 0.0, 0.0]), (gly, "CA", -0.03, [6.0, 0.0, 0.0]),
        (gly, "C", 0.6, [6.5, -1.0, 0.0]),
        (lys, "N", -0.35, [7.5, -1.5, 0.0]), (lys, "H", 0.27, [8.0, -0.8, 0.0]),
        (lys, "CA", -0.24, [8.0, -2.8, 0.0]), (lys, "HA", 0.14, [8.6, -3.6, 0.0]),
        (water, "O", -0.83, [3.0, 4.0, 0.0]), (kept, "P", 1.1, [3.0, -6.0, 0.0]),
        (ion, "MG", 2.0, [3.0, 3.6, 0.0]),
    ]
    atoms = [_Atom(index, name, charge, residue) for index, (residue, name, charge, _) in enumerate(placed)]
    # The chain and the ion by name; the other kept residue may share SEP's.
    named = {(atom.residue.name, atom.name): atom for atom in atoms if atom.residue is not kept}
    for first, second in [
        (("MET", "CA"), ("MET", "HA")), (("LYS", "CA"), ("LYS", "HA")),
        (("MET", "CA"), ("MET", "C")), (("MET", "C"), ("MET", "O")), (("MET", "C"), ("ALA", "N")),
        (("ALA", "N"), ("ALA", "CA")), (("ALA", "CA"), ("ALA", "C")), (("ALA", "C"), ("SEP", "N")),
        (("SEP", "N"), ("SEP", "CA")), (("SEP", "CA"), ("SEP", "C")), (("SEP", "CA"), ("SEP", "P")),
        (("SEP", "C"), ("GLY", "N")), (("GLY", "N"), ("GLY", "CA")), (("GLY", "CA"), ("GLY", "C")),
        (("GLY", "C"), ("LYS", "N")), (("LYS", "N"), ("LYS", "H")), (("LYS", "N"), ("LYS", "CA")),
    ]:
        _bond(named[first], named[second])
    if extra_link:
        _bond(named[("SEP", "P")], named[("MG", "MG")])
    return SimpleNamespace(
        residues=residues,
        atoms=atoms,
        coordinates=np.asarray([xyz for *_, xyz in placed], dtype=float),
    )


def _select(monkeypatch, structure, *, environment=None, **options):
    install_fake_parmed(monkeypatch, structure)
    data = _manifest([_qmmesp_block()])
    if environment:
        data["noncanonical_residues"][0]["qmmesp"]["environment"] = environment
    manifest = ManifestConfig.model_validate(data)
    residue = {"N": (2.0, 0.0, 0.0), "CA": (3.0, 0.0, 0.0), "C": (4.0, 0.0, 0.0), "P": (3.0, 1.5, 0.0)}
    return extract_residue_point_charges(
        prmtop="fake.prmtop",
        inpcrd="fake.inpcrd",
        resname="SEP",
        residue_coordinates=options.pop("residue", residue),
        qm_coordinates=np.asarray(list(residue.values()) + [[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        qmmesp=manifest.noncanonical_residues[0].qmmesp,
        manifest=manifest,
        label="noncanonical residue SEP (B:SEP57)",
        **options,
    )


def test_the_residue_and_the_two_its_caps_stand_in_for_are_left_out_whole(monkeypatch):
    selection = _select(monkeypatch, _chain())

    assert selection.target_atom_indices == [7, 8, 9, 10]
    assert {(charge.residue_name, charge.atom_name) for charge in selection.point_charges} == {
        ("MET", "CA"), ("MET", "HA"), ("LYS", "CA"), ("LYS", "HA"),
        ("WAT", "O"), ("TPO", "P"), ("MG", "MG"),
    }
    assert selection.excluded_residues == ("ALA2", "GLY4")
    # And the peptide bonds that join those two to the rest of the chain.
    assert selection.boundary_atoms_excluded == ("MET@C", "MET@O", "LYS@N", "LYS@H")
    assert selection.net_embedding_charge == pytest.approx(0.02 + 0.09 - 0.24 + 0.14 - 0.83 + 1.1 + 2.0)
    report = selection.to_dict()
    assert report["target_residue_excluded_from_embedding"] is True
    assert "caps stand in for" in report["qmmesp_interpretation"]


def test_another_kept_residue_is_protein_not_a_ligand(monkeypatch):
    data_without = _select(
        monkeypatch,
        _chain(kept_name="SEP"),
        environment={"include_other_ligands": False},
    )

    assert data_without.categories == {"water": 1, "protein": 5}


def test_a_residue_bonded_to_more_than_its_neighbours_is_refused(monkeypatch):
    with pytest.raises(PointChargeError, match="to nothing more"):
        _select(monkeypatch, _chain(extra_link=True))


def test_an_mm_atom_on_top_of_the_qm_model_is_refused(monkeypatch):
    structure = _chain()
    structure.coordinates[18] = [3.0, 1.9, 0.0]  # the water's O, 0.4 A from SEP's P

    with pytest.raises(PointChargeError, match="not an environment but a clash"):
        _select(monkeypatch, structure)


def test_the_residue_is_found_by_its_atoms_wherever_tleap_put_them(monkeypatch):
    moved = {"N": (2.0, 0.0, 0.0), "CA": (3.0, 0.0, 0.0), "C": (4.0, 0.0, 0.0), "P": (3.0, 1.0, 0.0)}

    with pytest.raises(PointChargeError, match="atoms up to 0.500 A away"):
        _select(monkeypatch, _chain(), residue=moved)


# --------------------------------------------------------------------------
# Where its files go, and which a build takes
# --------------------------------------------------------------------------


def test_its_am1bcc_parameterisation_is_provisional_and_kept_apart(tmp_path):
    manifest = ManifestConfig.model_validate(_manifest([_qmmesp_block()]))
    (block,) = manifest.noncanonical_residues

    assert parameterisation_directory(tmp_path, block) == provisional_directory(tmp_path, "SEP")
    assert provisional_directory(tmp_path, "SEP").parent == residue_directory(tmp_path, "SEP")


def test_the_final_build_takes_only_the_fitted_library(tmp_path):
    manifest = ManifestConfig.model_validate(_manifest([_qmmesp_block()]))
    provisional = provisional_directory(tmp_path, "SEP")
    provisional.mkdir(parents=True)
    for suffix in ("lib", "frcmod"):
        (provisional / f"SEP.{suffix}").write_text("x")

    (files,) = residue_leap_inputs(manifest, tmp_path, provisional=True).files
    assert files.library_path == provisional / "SEP.lib"
    with pytest.raises(ResidueStageError, match="does not fall back on the provisional"):
        residue_leap_inputs(manifest, tmp_path)

    for suffix in ("lib", "frcmod"):
        (residue_directory(tmp_path, "SEP") / f"SEP.{suffix}").write_text("y")
    (files,) = residue_leap_inputs(manifest, tmp_path).files
    assert files.library_path == residue_directory(tmp_path, "SEP") / "SEP.lib"


def test_the_histidine_embedding_takes_its_provisional_charges(tmp_path):
    from mdprep.residues.embedding import _block_charges

    data = _manifest([_qmmesp_block()])
    data["project"]["output_dir"] = str(tmp_path)
    manifest = ManifestConfig.model_validate(data)
    directory = provisional_directory(tmp_path, "SEP")
    directory.mkdir(parents=True)
    (directory / "parameters.json").write_text(json.dumps({"charges": {"charges": {"N": -0.5163}}}))

    charges, _ = _block_charges(manifest.noncanonical_residues[0], manifest)

    assert charges == {"N": -0.5163}


def test_a_qmmesp_residue_installs_pyscf_and_rdkit():
    packages = conda_packages_for({"noncanonical_residues": [{"parameters": "qmmesp_pyscf"}]})

    assert {"pyscf", "rdkit"} <= set(packages)


# --------------------------------------------------------------------------
# The ligand stage builds the provisional system the fit needs
# --------------------------------------------------------------------------


def test_a_qmmesp_residue_alone_has_the_provisional_system_built(monkeypatch, tmp_path):
    import mdprep.ligands.workflow as ligands

    data = _manifest([_qmmesp_block()])
    data["project"]["output_dir"] = str(tmp_path)
    manifest = ManifestConfig.model_validate(data)
    built = []

    def provisional_system(structure, manifest, result, **options):
        built.append((result.ligands, options["context"]))
        return SimpleNamespace(prmtop_path=tmp_path / "p.prmtop", inpcrd_path=tmp_path / "p.inpcrd")

    monkeypatch.setattr(ligands, "_build_qmmesp_provisional_system", provisional_system)
    result = ligands.run_ligand_stage(
        SimpleNamespace(), manifest, output_dir=tmp_path, protonation_result=SimpleNamespace()
    )

    assert built == [([], "QMMESP provisional")]
    assert result.qmmesp_provisional_system.prmtop_path == tmp_path / "p.prmtop"


# --------------------------------------------------------------------------
# The planner writes a block that validates, for residues and ligands
# --------------------------------------------------------------------------


def test_choosing_qmmesp_for_a_ligand_writes_its_qm_level():
    plan = plan_decisions(Path("tests/data/cofactors/fad_2ar8.pdb"))
    answers = walk(
        plan,
        {
            "ligand.a_fad_600.charge_method": "qmmesp_pyscf",
            "structure.remove_unknown_heterogens": "false",
        },
    )

    manifest = build_manifest(plan, answers)

    (ligand,) = manifest["ligands"]
    assert ligand["qmmesp"] == QMMESP
    ManifestConfig.model_validate(manifest)


def test_the_fit_report_says_what_was_fitted():
    assert "only the residue's own atoms' charges were kept" in QMMESP_RESIDUE_CONFIRMATION
    assert "not written to the residue's library" in QMMESP_RESIDUE_CONFIRMATION


# --------------------------------------------------------------------------
# The fit itself, with PySCF, RESP and tleap replaced
# --------------------------------------------------------------------------


def test_the_fitted_charges_are_the_residues_and_the_caps_are_dropped(monkeypatch, tmp_path):
    pytest.importorskip("rdkit")
    import mdprep.residues.qmmesp as qmmesp
    from mdprep.structure.normalize import normalize_structure_stage

    data = _manifest([_qmmesp_block()])
    data["project"]["output_dir"] = str(tmp_path)
    manifest = ManifestConfig.model_validate(data)
    structure = normalize_structure_stage(manifest).normalized_structure
    (block,) = manifest.noncanonical_residues
    sep = next(residue for residue in structure.residues if residue.id.resname == "SEP")
    names = [atom.name.strip() for atom in sep.atoms]
    seen: dict[str, object] = {}

    def embedding(**options):
        seen["embedding"] = options
        return PointChargeSelection(
            target_atom_indices=[0],
            point_charges=[PointCharge(9.0, 9.0, 9.0, 2.0, "MG", 1, "MG", "ligand")],
            total_before_cutoff=1,
            total_after_cutoff=1,
            net_embedding_charge=2.0,
            min_distance=3.0,
            max_distance=3.0,
            categories={"ligand": 1},
            target_kind="residue",
            excluded_residues=("ASP56", "ILE58"),
        )

    def scf(**options):
        seen["scf"] = options
        return SimpleNamespace(
            mol=None, mf=None, stdout="", stderr="", warnings=[], to_dict=lambda: {"converged": True}
        )

    def resp(**options):
        seen["resp"] = options
        count = len(options["atom_coordinates_bohr"])
        charges = np.zeros(count)
        for index, value in options["fixed_charges"].items():
            charges[index] = value
        free = [
            index
            for index in range(count)
            if index not in options["fixed_charges"]
            and not any(index in members for members, _ in options["charge_groups"])
        ]
        charges[free] = (-2.0 - charges.sum()) / len(free)
        return SimpleNamespace(
            charges=charges, to_dict=lambda: {"rms_error": 0.001, "relative_rms_error": 0.1}
        )

    def library(capped, **options):
        seen["library"] = (capped, options)
        frcmod = options["frcmod"]
        return frcmod.with_suffix(".lib"), frcmod.with_suffix(".mol2"), SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(qmmesp, "extract_residue_point_charges", embedding)
    monkeypatch.setattr(qmmesp, "run_pyscf_scf", scf)
    monkeypatch.setattr(qmmesp, "evaluate_ligand_esp", lambda **options: np.zeros(len(options["grid_coordinates_angstrom"])))
    monkeypatch.setattr(qmmesp, "run_amber_resp_fit", resp)
    monkeypatch.setattr(qmmesp, "build_residue_library", library)
    frcmod = tmp_path / "provisional.frcmod"
    frcmod.write_text("junction terms\n")
    provisional = SimpleNamespace(
        atom_types={name: "XX" for name in names},
        antechamber_mol2_path=tmp_path / "SEP.capped.antechamber.mol2",
        files=SimpleNamespace(frcmod_path=frcmod),
        charges=SimpleNamespace(charges={name: -2.0 / len(names) for name in names}),
        to_dict=lambda: {"resname": "SEP"},
    )

    result = qmmesp.fit_residue_qmmesp(
        block, structure, provisional, manifest=manifest,
        prmtop=tmp_path / "p.prmtop", inpcrd=tmp_path / "p.inpcrd", output_dir=tmp_path,
    )

    # The QM model is the capped residue, the ESP fitted with its constraints.
    capped, options = seen["library"]
    roles = [atom.role for atom in capped.atoms]
    assert roles.count("ACE") == 6 and roles.count("NME") == 6
    assert seen["scf"]["charge"] == -2 and seen["scf"]["spin"] == 0
    assert seen["scf"]["mm_charges"].tolist() == [2.0]
    assert len(seen["resp"]["fixed_charges"]) == 4 and len(seen["resp"]["charge_groups"]) == 2
    assert seen["resp"]["provisional_mol2_path"] == provisional.antechamber_mol2_path
    # Only the residue's atoms reach the library, with the backbone held.
    assert set(result.charges.charges) == set(names)
    assert {name: result.charges.charges[name] for name in ("N", "H", "C", "O")} == BACKBONE_CHARGES["anionic"]
    assert round(sum(result.charges.charges.values()), 6) == -2.0
    assert options["frcmod"] == residue_directory(tmp_path, "SEP") / "SEP.frcmod"
    assert options["frcmod"].read_text() == "junction terms\n"
    written = json.loads((residue_directory(tmp_path, "SEP") / "parameters.json").read_text())
    assert written["charge_method"] == "qmmesp_pyscf"
    assert written["qmmesp"]["embedding"]["excluded_residues"] == ["ASP56", "ILE58"]
    record = json.loads((result.qm_dir / "pyscf_input.json").read_text())
    assert record["qm_region"] == "capped_residue" and record["direct_mm_potential_in_resp_target"] is False
    assert set(record["resp_fixed_charges"]) == {"N", "H", "C", "O"}


def test_a_four_letter_atom_name_is_read_from_its_columns():
    from mdprep.ambertools.resp import ac_atom_names

    text = (
        "CHARGE     -2.00 ( -2 )\n"
        "ATOM      1  CAY CGU     1       7.053  28.951  35.445 -0.172100        c3\n"
        "ATOM     12  OE11CGU     1      10.339  23.775  36.690 -0.874800         o\n"
        "BOND    1    1    2    1     CAY   CY\n"
    )

    assert ac_atom_names(text) == ["CAY", "OE11"]
