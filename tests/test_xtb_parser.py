import pytest

from mdprep.protonation.xtb_parser import (
    HARTREE_TO_KCAL_MOL,
    XtbParseError,
    compare_hid_hie_energies,
    parse_xtb_energy_file,
    parse_xtb_energy_text,
)


def test_parse_total_energy_from_fixture_stdout():
    assert parse_xtb_energy_file("tests/data/xtb_hid_stdout.txt") == -40.01


def test_compare_energies_selects_lower_hid_and_computes_delta():
    comparison = compare_hid_hie_energies(
        hid_energy_hartree=-40.01,
        hie_energy_hartree=-40.00,
        close_call_kcal_mol=0.5,
    )

    assert comparison.selected_state == "HID"
    assert comparison.delta_kcal_mol == pytest.approx(-0.01 * HARTREE_TO_KCAL_MOL)
    assert not comparison.close_call


def test_compare_energies_selects_lower_hie():
    comparison = compare_hid_hie_energies(
        hid_energy_hartree=-40.00,
        hie_energy_hartree=-40.01,
        close_call_kcal_mol=0.5,
    )

    assert comparison.selected_state == "HIE"


def test_close_call_warning_is_set():
    comparison = compare_hid_hie_energies(
        hid_energy_hartree=-40.0001,
        hie_energy_hartree=-40.0000,
        close_call_kcal_mol=0.5,
    )

    assert comparison.close_call
    assert comparison.warnings


def test_missing_energy_fails_clearly():
    with pytest.raises(XtbParseError) as excinfo:
        parse_xtb_energy_text("no energy here", source="fake stdout")

    assert "fake stdout" in str(excinfo.value)
