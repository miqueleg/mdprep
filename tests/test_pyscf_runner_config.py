import builtins

import numpy as np
import pytest

from mdprep.qm.pyscf_runner import PySCFRunnerError, run_pyscf_scf, scf_class_name


def test_hf_spin_zero_selects_rhf():
    assert scf_class_name("HF", 0) == "RHF"


def test_hf_spin_positive_selects_uhf():
    assert scf_class_name("HF", 1) == "UHF"


def test_non_hf_spin_zero_selects_rks():
    assert scf_class_name("B3LYP", 0) == "RKS"


def test_non_hf_spin_positive_selects_uks():
    assert scf_class_name("B3LYP", 1) == "UKS"


def test_missing_pyscf_produces_clear_optional_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(PySCFRunnerError, match="PySCF is required"):
        run_pyscf_scf(
            elements=["H"],
            coordinates=np.asarray([[0.0, 0.0, 0.0]], dtype=float),
            charge=0,
            spin=1,
            method="HF",
            basis="STO-3G",
            max_cycle=1,
            conv_tol=1.0e-6,
        )
