import re

import numpy as np
import pytest

from mdprep.ambertools.resp import AmberRespError, amber_resp_available, run_amber_resp_fit
from mdprep.charges.esp_grid import generate_merz_kollman_grid
from mdprep.qm.pyscf_esp import BOHR_PER_ANGSTROM


pytestmark = [pytest.mark.external, pytest.mark.ambertools]


@pytest.mark.skipif(not amber_resp_available(), reason="antechamber, respgen, or resp is unavailable")
def test_real_amber_two_stage_resp_reads_every_written_esp_point(tmp_path):
    atoms_angstrom = np.asarray([[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], dtype=float)
    grid = generate_merz_kollman_grid(
        elements=["C", "O"],
        coordinates=atoms_angstrom,
        vdw_scale_factors=[1.4, 1.6, 1.8, 2.0],
        point_density_per_square_angstrom=1.0,
        exclude_inside_vdw_scale=1.4,
        max_points=8000,
    )
    atoms_bohr = atoms_angstrom * BOHR_PER_ANGSTROM
    points_bohr = grid.points * BOHR_PER_ANGSTROM
    source_charges = np.asarray([-0.3, 0.3], dtype=float)
    values = (
        1.0 / np.linalg.norm(points_bohr[:, None, :] - atoms_bohr[None, :, :], axis=2)
    ) @ source_charges

    try:
        result = run_amber_resp_fit(
            provisional_mol2_path="tests/data/ligands/ligand_sub.good.mol2",
            atom_coordinates_bohr=atoms_bohr,
            grid_coordinates_bohr=points_bohr,
            esp_values_au=values,
            total_charge=0,
            multiplicity=1,
            atom_types="gaff2",
            work_dir=tmp_path / "resp",
        )
    except AmberRespError as exc:
        if "dyld cache" in str(exc):
            pytest.skip(f"local AmberTools RESP binary is not runnable reliably: {exc}")
        raise

    output = result.stage_1_output_path.read_text(encoding="utf-8")
    match = re.search(r"total number of esp points\s*=\s*(\d+)", output, flags=re.IGNORECASE)
    assert match is not None
    assert int(match.group(1)) == len(grid.points)
    assert result.converged is True
    assert result.charge_sum_final == pytest.approx(0.0, abs=1.0e-6)
    assert result.rms_error < 0.01
