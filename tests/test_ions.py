import pytest

from mdprep.leap.ions import IonPlanError, build_ion_plan, neutralizing_ion_count, salt_pair_count


def test_neutralization_ion_selection():
    assert neutralizing_ion_count(-2.0, positive_ion="Na+", negative_ion="Cl-") == ("Na+", 2)
    assert neutralizing_ion_count(1.0, positive_ion="Na+", negative_ion="Cl-") == ("Cl-", 1)
    assert neutralizing_ion_count(0.0, positive_ion="Na+", negative_ion="Cl-") == (None, 0)


def test_noninteger_total_charge_fails():
    with pytest.raises(IonPlanError):
        neutralizing_ion_count(0.20, positive_ion="Na+", negative_ion="Cl-")


def test_salt_pair_count_formula():
    assert salt_pair_count(0.15, 100000.0) == round(0.15 * 100000.0 * 0.000602214076)


def test_build_ion_plan_commands():
    plan = build_ion_plan(
        total_charge=-1.0,
        neutralize=True,
        positive_ion="Na+",
        negative_ion="Cl-",
        salt_concentration_molar=0.15,
        volume_a3=100000.0,
    )
    assert plan.commands[0] == "addionsrand system Na+ 1"
    assert "addionsrand system Cl-" in plan.commands[-1]


def test_zero_salt_adds_no_salt_pairs():
    plan = build_ion_plan(
        total_charge=0.0,
        neutralize=True,
        positive_ion="Na+",
        negative_ion="Cl-",
        salt_concentration_molar=0.0,
        volume_a3=None,
    )
    assert plan.commands == []
    assert plan.salt_pairs == 0
