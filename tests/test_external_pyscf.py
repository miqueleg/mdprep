from pathlib import Path
import hashlib

import numpy as np
import pytest

from mdprep.ambertools.resp import amber_resp_available
from mdprep.ligands.extract import extract_ligand
from mdprep.ligands.pyscf_charges import LigandPySCFChargeError, derive_pyscf_charges
from mdprep.qm.pyscf_runner import (
    pyscf_available,
    run_pyscf_analytic_hessian,
    run_pyscf_scf,
)
from mdprep.structure.normalize import normalize_structure_stage
from tests.test_ligand_workflow_mocked import qmmesp_block
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


pytestmark = [pytest.mark.external, pytest.mark.pyscf]


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_fixed_geometry_analytic_hessian_does_not_move_atoms():
    coordinates = np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float)
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=coordinates,
        charge=0,
        spin=0,
        method="HF",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-10,
        num_threads=1,
        max_memory_mb=512,
    )
    hessian = run_pyscf_analytic_hessian(result.mf, num_threads=1)

    assert hessian.shape == (2, 2, 3, 3)
    assert np.all(np.isfinite(hessian))
    assert result.mol.atom_coords(unit="Angstrom") == pytest.approx(coordinates)


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_point_charges_polarize_scf_density(tmp_path):
    common = {
        "elements": ["H", "H"],
        "coordinates": np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        "charge": 0,
        "spin": 0,
        "method": "HF",
        "basis": "STO-3G",
        "max_cycle": 50,
        "conv_tol": 1.0e-10,
        "num_threads": 1,
        "max_memory_mb": 512,
    }
    gas = run_pyscf_scf(**common)
    embedded = run_pyscf_scf(
        **common,
        mm_charges=np.asarray([-0.5], dtype=float),
        mm_coordinates=np.asarray([[2.5, 0.8, 0.0]], dtype=float),
        checkpoint_path=tmp_path / "embedded.chk",
    )

    assert embedded.electrostatic_embedding_applied is True
    assert embedded.mm_point_charge_count == 1
    assert embedded.embedding_operator == "pyscf.qmmm.mm_charge"
    assert np.linalg.norm(np.asarray(embedded.mf.make_rdm1()) - np.asarray(gas.mf.make_rdm1())) > 1.0e-8
    assert embedded.energy_hartree != pytest.approx(gas.energy_hartree, abs=1.0e-8)
    assert (tmp_path / "embedded.chk").exists()


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_explicit_newton_scf_records_protocol(tmp_path):
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        charge=0,
        spin=0,
        method="HF",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-10,
        num_threads=1,
        max_memory_mb=512,
        checkpoint_path=tmp_path / "newton.chk",
        scf_algorithm="newton",
        initial_guess="atom",
        level_shift_hartree=0.2,
        damping_factor=0.1,
        diis_space=10,
    )

    assert result.converged is True
    assert result.scf_algorithm == "newton"
    assert result.initial_guess == "atom"
    assert result.level_shift_hartree == pytest.approx(0.2)
    assert result.damping_factor == pytest.approx(0.1)
    assert result.diis_space == 10


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_adiis_preconditioned_newton_records_protocol(tmp_path):
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        charge=0,
        spin=0,
        method="B3LYP",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-9,
        num_threads=1,
        max_memory_mb=512,
        checkpoint_path=tmp_path / "adiis_then_newton.chk",
        scf_algorithm="adiis_then_newton",
        initial_guess="atom",
        adiis_precondition_cycles=2,
        adiis_precondition_dft_grid_level=0,
        dft_grid_level=1,
        level_shift_hartree=0.2,
    )

    assert result.converged is True
    assert result.scf_algorithm == "adiis_then_newton"
    assert result.precondition_cycles_completed >= 1
    assert result.adiis_precondition_dft_grid_level == 0
    assert result.dft_grid_level == 1


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_newton_can_use_pinned_matching_density_checkpoint(tmp_path):
    checkpoint = tmp_path / "seed.chk"
    common = {
        "elements": ["H", "H"],
        "coordinates": np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        "charge": 0,
        "spin": 0,
        "method": "HF",
        "basis": "STO-3G",
        "max_cycle": 50,
        "conv_tol": 1.0e-10,
        "num_threads": 1,
        "max_memory_mb": 512,
    }
    seed = run_pyscf_scf(**common, checkpoint_path=checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    restarted = run_pyscf_scf(
        **common,
        checkpoint_path=tmp_path / "restarted.chk",
        initial_density_checkpoint_path=checkpoint,
        expected_initial_density_checkpoint_sha256=digest,
        scf_algorithm="newton",
    )

    assert restarted.converged is True
    assert restarted.energy_hartree == pytest.approx(seed.energy_hartree, abs=1.0e-10)
    assert restarted.initial_density_checkpoint_path == str(checkpoint.resolve())
    assert restarted.initial_density_checkpoint_sha256 == digest
    assert restarted.initial_density_checkpoint_projected is False
    assert restarted.initial_density_checkpoint_max_displacement_angstrom == pytest.approx(0.0)


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_restart_projects_a_pinned_nearby_geometry(tmp_path):
    checkpoint = tmp_path / "seed_shift.chk"
    seed_coordinates = np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float)
    run_pyscf_scf(
        elements=["H", "H"],
        coordinates=seed_coordinates,
        charge=0,
        spin=0,
        method="HF",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-10,
        checkpoint_path=checkpoint,
        num_threads=1,
        max_memory_mb=512,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    shifted = seed_coordinates.copy()
    shifted[1, 0] += 0.001
    restarted = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=shifted,
        charge=0,
        spin=0,
        method="HF",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-10,
        checkpoint_path=tmp_path / "shifted.chk",
        initial_density_checkpoint_path=checkpoint,
        expected_initial_density_checkpoint_sha256=digest,
        initial_density_checkpoint_max_displacement_angstrom=0.01,
        scf_algorithm="newton",
        num_threads=1,
        max_memory_mb=512,
    )

    assert restarted.converged is True
    assert restarted.initial_density_checkpoint_projected is True
    assert restarted.initial_density_checkpoint_max_displacement_angstrom == pytest.approx(0.001)


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_density_fitting_is_explicit_and_recorded():
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        charge=0,
        spin=0,
        method="B3LYP",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-9,
        num_threads=1,
        max_memory_mb=512,
        density_fitting=True,
    )

    assert result.converged is True
    assert result.density_fitting is True
    assert result.auxiliary_basis is None


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_dynamic_level_shift_is_explicit_and_recorded():
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        charge=0,
        spin=0,
        method="HF",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-10,
        num_threads=1,
        max_memory_mb=512,
        level_shift_mode="dynamic",
        level_shift_hartree=0.5,
    )

    assert result.converged is True
    assert result.level_shift_mode == "dynamic"
    assert result.level_shift_hartree == pytest.approx(0.5)


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
def test_real_pyscf_adiis_then_diis_is_explicit_and_recorded():
    result = run_pyscf_scf(
        elements=["H", "H"],
        coordinates=np.asarray([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=float),
        charge=0,
        spin=0,
        method="B3LYP",
        basis="STO-3G",
        max_cycle=50,
        conv_tol=1.0e-9,
        num_threads=1,
        max_memory_mb=512,
        scf_algorithm="adiis_then_diis",
        adiis_precondition_cycles=2,
        adiis_precondition_dft_grid_level=0,
        dft_grid_level=1,
        level_shift_hartree=0.2,
    )

    assert result.converged is True
    assert result.scf_algorithm == "adiis_then_diis"
    assert result.adiis_precondition_cycles == 2
    assert result.adiis_precondition_dft_grid_level == 0
    assert result.precondition_cycles_completed >= 1


@pytest.mark.skipif(not pyscf_available(), reason="PySCF is not installed")
@pytest.mark.skipif(not amber_resp_available(), reason="antechamber, respgen, or resp is unavailable")
def test_real_pyscf_gas_resp_charge_derivation_tiny_ligand(tmp_path):
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    data["project"]["output_dir"] = str(tmp_path / "prepared")
    data["structure"]["remove_unknown_heterogens"] = True
    data["ligands"] = [
        {
            **ligand_entry("sub_501", "B", "SUB", 501),
            "charge_method": "gas_resp_pyscf",
            "qmmesp": {
                **qmmesp_block(),
                "basis": "STO-3G",
                "max_cycle": 50,
                "grid": {
                    "type": "merz_kollman",
                    "vdw_scale_factors": [1.6],
                    "point_density_per_square_angstrom": 0.5,
                    "exclude_inside_vdw_scale": 1.4,
                    "max_points": 200,
                },
                "resp_fitting": {"backend": "ambertools", "stage_2": True},
            },
        }
    ]
    manifest = make_manifest(data)
    structure = normalize_structure_stage(manifest).normalized_structure
    extracted = extract_ligand(structure, manifest.ligands[0], output_dir=tmp_path)

    try:
        result = derive_pyscf_charges(
            extracted=extracted,
            provisional_mol2_path=Path("tests/data/ligands/ligand_sub.good.mol2"),
            output_mol2_path=tmp_path / "sub.pyscf.mol2",
            output_dir=tmp_path,
            method_name="gas_resp_pyscf",
        )
    except LigandPySCFChargeError as exc:
        if "dyld cache" in str(exc):
            pytest.skip(f"local AmberTools RESP binary is not runnable reliably: {exc}")
        raise

    assert result.charged_mol2_path.exists()
    assert result.grid_point_count > len(extracted.atoms)
    assert result.fit_result["charge_sum_final"] == pytest.approx(0.0, abs=1.0e-6)
