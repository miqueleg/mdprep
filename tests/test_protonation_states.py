import pytest

from mdprep.protonation.states import ProtonationStateError, validate_state_transition


def assert_allowed(current: str, requested: str):
    validate_state_transition(current_resname=current, requested_state=requested, selector="A:1")


def test_asp_family_validation_allows_asp_ash():
    assert_allowed("ASP", "ASH")
    assert_allowed("ASH", "ASP")


def test_glu_family_validation_allows_glu_glh():
    assert_allowed("GLU", "GLH")
    assert_allowed("GLH", "GLU")


def test_his_family_validation_allows_his_states():
    for state in ["HIS", "HID", "HIE", "HIP"]:
        assert_allowed("HIS", state)


def test_cys_family_validation_allows_cys_states():
    for state in ["CYS", "CYM", "CYX"]:
        assert_allowed("CYS", state)


def test_asp_to_hie_fails():
    with pytest.raises(ProtonationStateError):
        assert_allowed("ASP", "HIE")


def test_glu_to_ash_fails():
    with pytest.raises(ProtonationStateError):
        assert_allowed("GLU", "ASH")


def test_unknown_state_fails():
    with pytest.raises(ProtonationStateError):
        assert_allowed("ASP", "BAD")

