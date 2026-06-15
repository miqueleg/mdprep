import numpy as np

from mdprep.qm.pyscf_esp import evaluate_ligand_esp


class FakeMol:
    def __init__(self):
        self.origin = None

    def atom_coords(self):
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float)

    def atom_charges(self):
        return np.asarray([1.0], dtype=float)

    def set_rinv_origin(self, origin):
        self.origin = np.asarray(origin, dtype=float)

    def intor(self, name):
        assert name == "int1e_rinv"
        return np.asarray([[0.25]], dtype=float)


class FakeMf:
    def make_rdm1(self):
        return np.asarray([[0.5]], dtype=float)


def test_esp_evaluation_returns_expected_shape_and_target_only_potential():
    values = evaluate_ligand_esp(
        mol=FakeMol(),
        mf=FakeMf(),
        grid_coordinates_angstrom=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float),
    )

    assert values.shape == (2,)
    assert np.all(np.isfinite(values))
    assert values[0] > values[1]
