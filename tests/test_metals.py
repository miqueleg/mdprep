from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mdprep.config.models import ManifestConfig
from mdprep.config.loader import load_manifest
from mdprep.external.runner import CommandResult
from mdprep.leap.builder import TLeapOutputs, build_tleap_script
from mdprep.leap.forcefields import forcefield_sources
from mdprep.ligands.workflow import (
    LigandStageResult,
    LigandWorkflowItem,
    LigandWorkflowError,
    _qmmesp_provisional_metal_setup,
)
from mdprep.metals.c4 import _atom_type_mask
from mdprep.metals.coordination import MetalCoordinationError, resolve_metal_sites
from mdprep.metals.hydrogenate import _validate_preserved_input_coordinates
from mdprep.metals.mcpb import (
    collect_mcpb_refitted_ligands,
    run_mcpb_site,
    write_mcpb_numbered_pdb,
)
from mdprep.metals.nonbonded import (
    ion_frcmod_name,
    load_amber_ion_parameter,
    prepare_nonbonded_site,
)
from mdprep.metals.workflow import promote_mcpb_resp_ligands
from mdprep.protonation.apply import ProtonationApplicationError, apply_protonation_stage
from mdprep.structure.normalize import normalize_structure_stage
from mdprep.structure.pdb import read_pdb
from mdprep.structure.models import PdbStructure
from tests.test_ligand_workflow_mocked import qmmesp_block
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def atom(chain: str, resname: str, resid: int, atom_name: str) -> dict:
    return {
        "chain": chain,
        "resname": resname,
        "resid": resid,
        "icode": None,
        "atom_name": atom_name,
    }


def ion() -> dict:
    return {"selector": atom("Z", "ZN", 500, "ZN"), "element": "Zn", "charge": 2}


def nonbonded_site(parameter_set: str = "12_6") -> dict:
    return {
        "id": "zinc_site",
        "model": "nonbonded",
        "ions": [ion()],
        "nonbonded": {"parameter_set": parameter_set},
    }


def bonded_site(*, donor: str = "NE2", workflow: str = "prepare_inputs") -> dict:
    return {
        "id": "zinc_site",
        "model": "bonded_mcpb",
        "ions": [ion()],
        "mcpb": {
            "workflow": workflow,
            "provisional_nonbonded_parameter_set": "12_6",
            "bonds": [
                {
                    "ion": atom("Z", "ZN", 500, "ZN"),
                    "coordinator": atom("A", "HIS", 2, donor),
                }
            ],
            "cutoff_angstrom": 2.3,
            "force_constant_method": "seminario",
            "charge_restraint": "backbone_heavy",
            "software_version": "g16",
            "small_model_charge": 2,
            "small_model_spin": 1,
            "large_model_charge": 2,
            "large_model_spin": 1,
        },
    }


def metal_manifest(site: dict) -> ManifestConfig:
    data = manifest_data("tests/data/protein_histidine_zinc.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["protonation"]["method"] = "manual_only"
    data["metals"] = [site]
    return make_manifest(data)


def test_metal_model_settings_are_not_guessed():
    data = manifest_data("tests/data/protein_histidine_zinc.pdb")
    data["metals"] = [{"id": "bad", "model": "nonbonded", "ions": [ion()]}]
    with pytest.raises(ValidationError, match="requires nonbonded settings"):
        ManifestConfig.model_validate(data)


def test_configured_metal_is_retained_as_first_class_heterogen():
    result = normalize_structure_stage(metal_manifest(nonbonded_site()))
    assert result.configured_metal_ions_kept[0].element == "Zn"
    assert any(residue.id.resname == "ZN" for residue in result.normalized_structure.residues)


def test_ne2_coordination_assigns_hid_before_propka_or_xtb():
    manifest = metal_manifest(bonded_site(donor="NE2"))
    normalized = normalize_structure_stage(manifest)
    result = apply_protonation_stage(
        normalized.normalized_structure,
        manifest,
        input_normalized_pdb_path="normalized.pdb",
        output_protonation_pdb_path="protonated.pdb",
    )
    record = result.metal_coordination_assignments_applied[0]
    assert record.final_resname == "HID"
    assert record.source == "metal_coordination"
    assert "HID" in [residue.id.resname for residue in result.structure.residues]


def test_conflicting_manual_histidine_override_fails_instead_of_being_replaced():
    data = metal_manifest(bonded_site(donor="NE2")).model_dump(mode="json")
    data["protonation"]["overrides"] = [
        {
            "selector": {"chain": "A", "resname": "HIS", "resid": 2, "icode": None},
            "state": "HIE",
            "reason": "deliberate conflict",
        }
    ]
    manifest = ManifestConfig.model_validate(data)
    normalized = normalize_structure_stage(manifest)
    with pytest.raises(ProtonationApplicationError, match="requires HID"):
        apply_protonation_stage(
            normalized.normalized_structure,
            manifest,
            input_normalized_pdb_path="normalized.pdb",
            output_protonation_pdb_path="protonated.pdb",
        )


def test_nonhistidine_metal_donor_requires_manual_protonation_override():
    site = bonded_site()
    site["mcpb"]["bonds"] = [
        {
            "ion": atom("Z", "ZN", 500, "ZN"),
            "coordinator": atom("A", "ASP", 3, "OD1"),
        }
    ]
    site["mcpb"]["cutoff_angstrom"] = 0.5
    manifest = metal_manifest(site)
    normalized = normalize_structure_stage(manifest)
    with pytest.raises(ProtonationApplicationError, match="requires an explicit protonation override"):
        apply_protonation_stage(
            normalized.normalized_structure,
            manifest,
            input_normalized_pdb_path="normalized.pdb",
            output_protonation_pdb_path="protonated.pdb",
        )


def test_mcpb_cutoff_candidates_must_all_be_explicitly_approved():
    manifest = metal_manifest(bonded_site(donor="ND1"))
    normalized = normalize_structure_stage(manifest)
    with pytest.raises(MetalCoordinationError, match="undeclared bonds"):
        resolve_metal_sites(normalized.normalized_structure, manifest)


@pytest.mark.parametrize(
    ("water", "charge", "family", "expected"),
    [
        ("TIP3P", 2, "12_6", "frcmod.ions234lm_126_tip3p"),
        ("TIP3P", 1, "hfe", "frcmod.ions1lm_126_tip3p"),
        ("OPC", 2, "iod", "frcmod.ionslm_iod_opc"),
        ("OPC", 3, "12_6_4", "frcmod.ionslm_1264_opc"),
    ],
)
def test_exact_amber_ion_frcmod_mapping(water, charge, family, expected):
    assert ion_frcmod_name(charge=charge, water_model=water, parameter_set=family) == expected


def test_nonbonded_parameters_are_read_from_amber_file_not_hardcoded(tmp_path):
    parm = tmp_path / "dat" / "leap" / "parm"
    parm.mkdir(parents=True)
    frcmod = parm / "frcmod.ions234lm_126_tip3p"
    frcmod.write_text(
        "test\nMASS\nZn2+ 65.400\n\nNONBON\nZn2+ 1.271 0.00330286 source\n",
        encoding="utf-8",
    )
    parameter = load_amber_ion_parameter(
        element="Zn",
        charge=2,
        water_model="TIP3P",
        parameter_set="12_6",
        amberhome=tmp_path,
    )
    assert parameter.atom_type == "Zn2+"
    assert parameter.rmin_over_2_angstrom == pytest.approx(1.271)
    assert parameter.epsilon_kcal_mol == pytest.approx(0.00330286)


def test_nonbonded_site_generates_exact_charge_mol2_and_tleap_setup(tmp_path):
    manifest = metal_manifest(nonbonded_site())
    normalized = normalize_structure_stage(manifest)
    site = resolve_metal_sites(normalized.normalized_structure, manifest)[0]
    parm = tmp_path / "amber" / "dat" / "leap" / "parm"
    parm.mkdir(parents=True)
    (parm / "frcmod.ions234lm_126_tip3p").write_text(
        "test\nMASS\nZn2+ 65.400\n\nNONBON\nZn2+ 1.271 0.00330286\n",
        encoding="utf-8",
    )
    result = prepare_nonbonded_site(
        site,
        water_model="TIP3P",
        output_dir=tmp_path / "site",
        amberhome=tmp_path / "amber",
    )
    mol2 = result.ions[0].mol2_path.read_text(encoding="utf-8")
    assert "Zn2+" in mol2
    assert "2.000000" in mol2
    assert any(command.startswith("loadamberparams ") for command in result.leap_commands)


def test_qmmesp_provisional_topology_loads_configured_metal_model(monkeypatch, tmp_path):
    pdb = tmp_path / "protein_ligand_zinc.pdb"
    source = Path("tests/data/protein_two_ligands.pdb").read_text(encoding="utf-8")
    pdb.write_text(
        source.replace(
            "END\n",
            "HETATM    8 ZN    ZN Z 700       7.800   5.000   5.000  1.00 20.00          ZN\nEND\n",
        ),
        encoding="utf-8",
    )
    data = manifest_data(str(pdb))
    data["structure"]["remove_unknown_heterogens"] = True
    data["ligands"] = [
        {
            **ligand_entry("sub", "B", "SUB", 501),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": qmmesp_block(),
        }
    ]
    site = nonbonded_site()
    site["ions"][0]["selector"] = atom("Z", "ZN", 700, "ZN")
    data["metals"] = [site]
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    parm = tmp_path / "amber" / "dat" / "leap" / "parm"
    parm.mkdir(parents=True)
    (parm / "frcmod.ions234lm_126_tip3p").write_text(
        "test\nMASS\nZn2+ 65.400\n\nNONBON\nZn2+ 1.271 0.00330286\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mdprep.metals.nonbonded.resolve_amberhome",
        lambda explicit=None: tmp_path / "amber",
    )
    commands = _qmmesp_provisional_metal_setup(
        normalized.normalized_structure,
        manifest,
        output_dir=tmp_path / "qmmesp_metals",
    )
    assert any(command.startswith("loadamberparams ") for command in commands)
    ion_mol2 = next((tmp_path / "qmmesp_metals").rglob("*.mol2"))
    assert "2.000000" in ion_mol2.read_text(encoding="utf-8")


def test_qmmesp_target_cannot_be_refitted_by_mcpb(tmp_path):
    pdb = tmp_path / "protein_ligand_zinc.pdb"
    source = Path("tests/data/protein_two_ligands.pdb").read_text(encoding="utf-8")
    pdb.write_text(
        source.replace(
            "END\n",
            "HETATM    8 ZN    ZN Z 700       7.800   5.000   5.000  1.00 20.00          ZN\nEND\n",
        ),
        encoding="utf-8",
    )
    data = manifest_data(str(pdb))
    data["structure"]["remove_unknown_heterogens"] = True
    data["ligands"] = [
        {
            **ligand_entry("sub", "B", "SUB", 501),
            "charge_method": "qmmesp_pyscf",
            "qmmesp": qmmesp_block(),
        }
    ]
    site = bonded_site()
    site["ions"][0]["selector"] = atom("Z", "ZN", 700, "ZN")
    site["mcpb"]["bonds"] = [
        {
            "ion": atom("Z", "ZN", 700, "ZN"),
            "coordinator": atom("B", "SUB", 501, "O1"),
        }
    ]
    site["mcpb"]["cutoff_angstrom"] = 2.0
    data["metals"] = [site]
    manifest = make_manifest(data)
    normalized = normalize_structure_stage(manifest)
    with pytest.raises(LigandWorkflowError, match="would replace that ligand's QMMESP charges"):
        _qmmesp_provisional_metal_setup(
            normalized.normalized_structure,
            manifest,
            output_dir=tmp_path / "qmmesp_metals",
        )


def test_mcpb_prepare_inputs_records_external_step_and_artifacts(monkeypatch, tmp_path):
    manifest = metal_manifest(bonded_site())
    normalized = normalize_structure_stage(manifest)
    site = resolve_metal_sites(normalized.normalized_structure, manifest)[0]
    fake_exe = tmp_path / "MCPB.py"
    fake_exe.write_text("", encoding="utf-8")
    monkeypatch.setattr("mdprep.metals.mcpb.which_executable", lambda _: str(fake_exe))
    monkeypatch.setattr(
        "mdprep.metals.mcpb.resolve_amberhome",
        lambda _: tmp_path / "amber",
    )

    def fake_run(command, *, cwd=None, env=None, timeout=None, check=False):
        if command[-1] == "1":
            for name in (
                "zinc_site_small_opt.com",
                "zinc_site_small_fc.com",
                "zinc_site_large_mk.com",
            ):
                (Path(cwd) / name).write_text("qm input\n", encoding="utf-8")
        return CommandResult(tuple(command), str(cwd), 0, "step complete\n", "", 0.01)

    monkeypatch.setattr("mdprep.metals.mcpb.run_command", fake_run)
    result = run_mcpb_site(
        site,
        structure=normalized.normalized_structure,
        manifest=manifest,
        ligand_result=LigandStageResult(ligands=[]),
        output_dir=tmp_path / "mcpb",
    )
    assert not result.complete
    assert result.runs[0].step == "1"
    assert result.runs[0].result.command[-1] == "1"
    assert result.expected_qm_artifacts == (
        "zinc_site_large_mk.log",
        "zinc_site_small_opt.fchk",
    )
    text = result.input_path.read_text(encoding="utf-8")
    assert "smmodel_chg 2" in text
    assert "lgmodel_spin 1" in text


def test_gamess_seminario_requires_log_but_not_gaussian_fchk():
    site = bonded_site()
    site["mcpb"]["software_version"] = "gms"
    site["mcpb"]["workflow"] = "complete"
    site["mcpb"]["artifacts"] = {
        "small_fc_log": "small.log",
        "large_mk_log": "large.log",
    }
    manifest = metal_manifest(site)
    assert manifest.metals[0].mcpb is not None
    assert manifest.metals[0].mcpb.artifacts is not None
    assert manifest.metals[0].mcpb.artifacts.small_opt_fchk is None


def test_mcpb_resp_pyscf_can_resume_from_explicit_gaussian_artifacts():
    data = manifest_data("tests/data/protein_histidine_zinc.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [
        {
            **ligand_entry("cofactor", "A", "HIS", 2),
            "charge_method": "mcpb_resp_pyscf",
        }
    ]
    site = bonded_site(workflow="complete")
    site["mcpb"]["software_version"] = "gau"
    site["mcpb"]["artifacts"] = {
        "small_opt_fchk": "small.fchk",
        "small_fc_log": None,
        "large_mk_log": "large.log",
    }
    data["metals"] = [site]

    manifest = ManifestConfig.model_validate(data)

    assert manifest.metals[0].mcpb is not None
    assert manifest.metals[0].mcpb.workflow == "complete"


def test_mcpb_resp_pyscf_complete_requires_gaussian_compatible_artifacts():
    data = manifest_data("tests/data/protein_histidine_zinc.pdb")
    data["structure"]["remove_unknown_heterogens"] = False
    data["protonation"]["method"] = "manual_only"
    data["ligands"] = [
        {
            **ligand_entry("cofactor", "A", "HIS", 2),
            "charge_method": "mcpb_resp_pyscf",
        }
    ]
    site = bonded_site(workflow="complete")
    site["mcpb"]["software_version"] = "gms"
    site["mcpb"]["artifacts"] = {
        "small_fc_log": "small.log",
        "large_mk_log": "large.log",
    }
    data["metals"] = [site]

    with pytest.raises(ValidationError, match="software_version must be gau"):
        ManifestConfig.model_validate(data)


def test_gamess_z_matrix_is_rejected_before_external_execution():
    site = bonded_site()
    site["mcpb"]["software_version"] = "gms"
    site["mcpb"]["force_constant_method"] = "z_matrix"
    data = manifest_data("tests/data/protein_histidine_zinc.pdb")
    data["metals"] = [site]
    with pytest.raises(ValidationError, match="Z-matrix parser does not read GAMESS"):
        ManifestConfig.model_validate(data)


def test_mcpb_complete_runs_selected_steps_and_composes_generated_files(
    monkeypatch,
    tmp_path,
):
    small_fchk = tmp_path / "small.fchk"
    large_log = tmp_path / "large.log"
    small_fchk.write_text("formatted checkpoint\n", encoding="utf-8")
    large_log.write_text("ESP output\n", encoding="utf-8")
    site_data = bonded_site(workflow="complete")
    site_data["mcpb"]["artifacts"] = {
        "small_opt_fchk": str(small_fchk),
        "small_fc_log": None,
        "large_mk_log": str(large_log),
    }
    manifest = metal_manifest(site_data)
    normalized = normalize_structure_stage(manifest)
    site = resolve_metal_sites(normalized.normalized_structure, manifest)[0]
    fake_exe = tmp_path / "MCPB.py"
    fake_exe.write_text("", encoding="utf-8")
    monkeypatch.setattr("mdprep.metals.mcpb.which_executable", lambda _: str(fake_exe))
    monkeypatch.setattr(
        "mdprep.metals.mcpb.resolve_amberhome",
        lambda _: tmp_path / "amber",
    )

    def fake_run(command, *, cwd=None, env=None, timeout=None, check=False):
        work = Path(cwd)
        step = command[-1]
        if step == "1":
            for name in (
                "zinc_site_small_opt.com",
                "zinc_site_small_fc.com",
                "zinc_site_large_mk.com",
            ):
                (work / name).write_text("qm input\n", encoding="utf-8")
        elif step == "4b":
            (work / "zinc_site_mcpbpy.pdb").write_text(
                (work / "system.mcpb_input.pdb").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (work / "zinc_site_mcpbpy.frcmod").write_text(
                "MCPB parameters\n"
                "BOND\n"
                "M1-Y1  100.0  2.0000  Created by Seminario method using MCPB.py\n\n"
                "ANGL\n"
                "Y1-M1-Y2  50.0  110.0  Created by Seminario method using MCPB.py\n",
                encoding="utf-8",
            )
            (work / "Z1.mol2").write_text(
                (work / "ion_1_ZN.mol2").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (work / "zinc_site_tleap.in").write_text(
                "addAtomTypes {\n"
                '  { "M1" "Zn" "sp3" }\n'
                "}\n"
                "Z1 = loadmol2 Z1.mol2\n"
                "loadamberparams zinc_site_mcpbpy.frcmod\n"
                "mol = loadpdb zinc_site_mcpbpy.pdb\n"
                "bond mol.2.NE2 mol.4.ZN\n",
                encoding="utf-8",
            )
        return CommandResult(tuple(command), str(work), 0, "step complete\n", "", 0.01)

    monkeypatch.setattr("mdprep.metals.mcpb.run_command", fake_run)
    result = run_mcpb_site(
        site,
        structure=normalized.normalized_structure,
        manifest=manifest,
        ligand_result=LigandStageResult(ligands=[]),
        output_dir=tmp_path / "mcpb_complete",
    )
    assert result.complete
    assert [run.step for run in result.runs] == ["1", "2s", "3b", "4b"]
    assert result.restored_final_pdb_path is not None
    assert result.restored_final_pdb_path.is_file()
    assert result.leap_bond_commands == ("bond system.2.NE2 system.4.ZN",)
    assert any("zinc_site_mcpbpy.frcmod" in line for line in result.leap_setup_commands)


def test_mcpb_numbering_is_sequential_and_reversible(tmp_path):
    manifest = metal_manifest(bonded_site())
    normalized = normalize_structure_stage(manifest)
    mapping = write_mcpb_numbered_pdb(normalized.normalized_structure, tmp_path / "numbered.pdb")
    assert sorted(mapping.original_by_mcpb_atom_id) == list(
        range(1, len(normalized.normalized_structure.atoms) + 1)
    )
    assert mapping.mcpb_atom_id_by_original_identity[("Z", "ZN", 500, None, "ZN")] == 19


def test_metal_commands_are_composed_into_final_tleap_script(tmp_path):
    sources = forcefield_sources(protein_forcefield="ff19SB", water_model="TIP3P", ligands=[])
    outputs = TLeapOutputs(tmp_path / "a.prmtop", tmp_path / "a.inpcrd", tmp_path / "a.pdb")
    script = build_tleap_script(
        sources=sources,
        ligands=[],
        input_pdb=tmp_path / "input.pdb",
        disulfide_bonds=[],
        outputs=outputs,
        setup_commands=["loadamberparams /amber/frcmod.ions234lm_126_tip3p"],
        extra_bond_commands=["bond system.2.NE2 system.4.ZN"],
    )
    assert script.index("loadamberparams /amber") < script.index("system = loadpdb")
    assert script.index("bond system.2.NE2") > script.index("system = loadpdb")


def test_mcpb_joint_resp_mol2_is_validated_and_promoted(tmp_path):
    manifest = load_manifest("examples/tutorials/7E07_bonded_fe3/system.yaml")
    structure = read_pdb("examples/tutorials/7E07_bonded_fe3/input.pdb")
    residue = next(
        item
        for item in structure.residues
        if item.id.chain_id == "A" and item.id.resname == "AKG" and item.id.resid == 501
    )
    final_mol2 = tmp_path / "AG1.mol2"
    atom_lines = [
        f"{index:7d} {atom.name:<4} {atom.x:10.4f} {atom.y:10.4f} "
        f"{atom.z:10.4f} c3 1 AG1 {-2.0 / len(residue.atoms):12.6f}"
        for index, atom in enumerate(residue.atoms, start=1)
    ]
    final_mol2.write_text(
        "@<TRIPOS>MOLECULE\nAG1\n"
        f"{len(residue.atoms)} 0 1 0 0\nSMALL\nRESP Charge\n\n"
        "@<TRIPOS>ATOM\n"
        + "\n".join(atom_lines)
        + "\n@<TRIPOS>BOND\n@<TRIPOS>SUBSTRUCTURE\n"
        "1 AG1 1 TEMP 0 **** **** 0 ROOT\n",
        encoding="utf-8",
    )
    provisional = tmp_path / "alpha_ketoglutarate.am1bcc.mol2"
    provisional.write_text("provisional\n", encoding="utf-8")
    ligand_item = LigandWorkflowItem(
        ligand_id="alpha_ketoglutarate",
        selector={"chain": "A", "resname": "AKG", "resid": 501, "icode": None},
        residue_identity=residue.id.to_dict(),
        atom_count=len(residue.atoms),
        charge_method="mcpb_resp_pyscf",
        atom_types="gaff2",
        net_charge=-2,
        multiplicity=1,
        extracted_pdb_path=tmp_path / "akg.pdb",
        identity_path=tmp_path / "akg.identity.json",
        final_mol2_path=provisional,
        final_frcmod_path=tmp_path / "akg.frcmod",
        provisional_mol2_path=provisional,
        status="provisional_for_mcpb",
    )
    ligand_result = LigandStageResult(ligands=[ligand_item])
    renames = [
        {
            "chain": "A",
            "resid": 501,
            "icode": None,
            "original_resname": "AKG",
            "final_resname": "AG1",
            "source": "MCPB.py",
        }
    ]
    command = f"AG1 = loadmol2 {final_mol2.resolve()}"

    artifacts = collect_mcpb_refitted_ligands(
        manifest=manifest,
        structure=structure,
        ligand_result=ligand_result,
        residue_renames=renames,
        work_dir=tmp_path,
        leap_setup_commands=[command],
    )

    assert len(artifacts) == 1
    assert artifacts[0].final_mol2_path == final_mol2.resolve()
    assert artifacts[0].fragment_charge == pytest.approx(-2.0, abs=1.0e-5)
    final_frcmod = tmp_path / "cluster.frcmod"
    promoted = promote_mcpb_resp_ligands(
        ligand_result,
        SimpleNamespace(
            mcpb_site=SimpleNamespace(
                complete=True,
                refitted_ligands=tuple(artifacts),
                final_frcmod_path=final_frcmod,
            )
        ),
    )
    assert promoted.ligands[0].final_mol2_path == final_mol2.resolve()
    assert promoted.ligands[0].final_frcmod_path == final_frcmod
    assert promoted.ligands[0].provisional_mol2_path == provisional
    assert promoted.ligands[0].status == "ok"


def test_1264_mask_selects_explicit_amber_atom_types():
    assert _atom_type_mask(["Zn2+", "Mg2+", "Zn2+"]) == "@%Mg2+,Zn2+"


def test_pre_mcpb_coordinate_audit_treats_hoh_to_wat_as_representation_rename(tmp_path):
    reference = read_pdb("tests/data/protein_with_waters.pdb")
    observed_atoms = [
        replace(atom, resname="WAT") if atom.resname == "HOH" else atom
        for atom in reference.atoms
    ]
    observed = PdbStructure(
        path=tmp_path / "tleap.pdb",
        atoms=observed_atoms,
        residues=reference.residues,
        model_count=1,
    )

    assert _validate_preserved_input_coordinates(reference, observed) == 0.0
