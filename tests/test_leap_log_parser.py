import pytest

from mdprep.leap.log_parser import LeapLogError, assert_tleap_success, parse_tleap_log_text


def test_parse_warnings_errors_and_charge():
    text = """
Warning: Close contact
Unknown residue: XXX
Could not find type: zz
Could not find bond parameter for c1-o
Total unperturbed charge:   -1.000000
Created a new atom named H1
"""
    summary = parse_tleap_log_text(text, returncode=0)
    assert summary.total_charge == -1.0
    assert summary.warnings
    assert summary.unknown_residues
    assert summary.missing_atom_types
    assert summary.missing_parameters
    assert summary.atoms_created


def test_fail_on_warnings_behavior():
    summary = parse_tleap_log_text("Warning: something", returncode=0)
    assert_tleap_success(summary, fail_on_warnings=False, context="dry")
    with pytest.raises(LeapLogError):
        assert_tleap_success(summary, fail_on_warnings=True, context="dry")


def test_nonzero_returncode_error_includes_log_tail():
    summary = parse_tleap_log_text(
        "\n".join(
            [
                "Loading parameters",
                "Could not find type: ca",
                "Fatal Error!",
                "Quit",
            ]
        ),
        returncode=31,
    )

    with pytest.raises(LeapLogError) as excinfo:
        assert_tleap_success(summary, fail_on_warnings=False, context="QMMESP provisional")

    message = str(excinfo.value)
    assert "QMMESP provisional tleap exited with code 31" in message
    assert "missing atom types" in message
    assert "tleap log tail" in message
    assert "Fatal Error!" in message
