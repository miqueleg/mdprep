"""The xTB point-charge convention, verified against xTB itself.

xTB does not document the units of the positions in its ``pcharge`` file, and
reading them in the wrong unit does not fail -- it silently places every MM
charge 1.89 times too far away and still returns an energy. The tautomer
preference it then reports looks entirely normal. Nothing in mdprep would catch
that, so it is pinned here against the real program.
"""

from pathlib import Path

import pytest

from mdprep.external.discovery import which_executable
from mdprep.protonation.histidine_qmmm import (
    BOHR_PER_ANGSTROM,
    EmbeddedPointCharge,
    embedding_was_applied,
    write_embedding_xcontrol,
    write_point_charge_file,
)
from mdprep.external.runner import run_command

pytestmark = pytest.mark.external

# A water geometry and its GFN2 partial charges, from an unembedded run.
WATER_XYZ = """3
water
O   0.000000   0.000000   0.000000
H   0.758602   0.000000   0.504284
H   0.758602   0.000000  -0.504284
"""
GFN2_WATER_CHARGES = (
    (-0.622, 0.0, 0.0, 0.0),
    (0.311, 0.758602, 0.0, 0.504284),
    (0.311, 0.758602, 0.0, -0.504284),
)
HARTREE_PER_KCAL_MOL = 1.0 / 627.5095


def _total_energy(stdout: str) -> float:
    for line in stdout.splitlines():
        if "TOTAL ENERGY" in line.upper():
            return float(line.split()[-3])
    raise AssertionError(f"no total energy in xTB output:\n{stdout}")


def _run(work: Path, *extra: str) -> str:
    work.mkdir(parents=True, exist_ok=True)
    (work / "water.xyz").write_text(WATER_XYZ, encoding="utf-8")
    result = run_command(
        ["xtb", "water.xyz", "--gfn", "2", "--chrg", "0", "--uhf", "0", *extra],
        cwd=work,
        env={**__import__("os").environ, "OMP_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _classical_interaction_hartree(distance_angstrom: float) -> float:
    """Charge-to-partial-charge energy with everything converted to bohr."""
    total = 0.0
    probe = distance_angstrom * BOHR_PER_ANGSTROM
    for charge, x, y, z in GFN2_WATER_CHARGES:
        separation = (
            (probe - x * BOHR_PER_ANGSTROM) ** 2
            + (y * BOHR_PER_ANGSTROM) ** 2
            + (z * BOHR_PER_ANGSTROM) ** 2
        ) ** 0.5
        total += charge * 1.0 / separation
    return total


def test_point_charge_positions_are_read_by_xtb_as_bohr(tmp_path):
    if which_executable("xtb") is None:
        pytest.skip("xtb is required for this external integration test")

    reference = _total_energy(_run(tmp_path / "plain"))
    for distance_angstrom in (5.0, 10.0):
        work = tmp_path / f"embedded_{distance_angstrom}"
        work.mkdir(parents=True, exist_ok=True)
        probe = EmbeddedPointCharge(
            charge=1.0,
            x=distance_angstrom,
            y=0.0,
            z=0.0,
            element="O",
            residue="A:PRB1",
            atom_name="PRB",
        )
        write_point_charge_file([probe], work / "pcharge")
        write_embedding_xcontrol(work / "xcontrol")
        stdout = _run(work, "--input", "xcontrol")

        assert embedding_was_applied(stdout), (
            "xTB did not report the point charges, so the embedding was ignored"
        )
        shift = _total_energy(stdout) - reference
        expected = _classical_interaction_hartree(distance_angstrom)
        # Agreement to a few percent, the remainder being the polarisation that
        # a fixed-charge estimate cannot capture. Reading the file as angstrom
        # instead would be wrong by a factor of roughly two.
        assert shift == pytest.approx(expected, rel=0.08), (
            f"at {distance_angstrom} A the embedded shift was {shift:.6f} Eh "
            f"against {expected:.6f} Eh expected for bohr positions"
        )


def test_an_empty_embedding_block_is_not_written(tmp_path):
    """xTB rejects an empty pcharge file rather than treating it as no field."""

    write_embedding_xcontrol(tmp_path / "xcontrol", embedding=False)
    assert "$embedding" not in (tmp_path / "xcontrol").read_text(encoding="utf-8")
