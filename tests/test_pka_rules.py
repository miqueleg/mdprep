import pytest

from mdprep.protonation.pka_rules import PkaRuleError, decide_residue_state
from mdprep.protonation.propka_parser import PropkaRecord
from mdprep.structure.models import ResidueId, ResidueRecord


def residue(resname: str, resid: int = 1) -> ResidueRecord:
    return ResidueRecord(
        id=ResidueId(chain_id="A", resname=resname, resid=resid, icode=None),
        atoms=[],
        record_names={"ATOM"},
        original_index=resid,
    )


def record(resname: str, pka: float, resid: int = 1) -> PropkaRecord:
    return PropkaRecord(resname=resname, resid=resid, chain_id="A", pka=pka, raw_line="")


@pytest.mark.parametrize(
    ("resname", "pka", "ph", "expected"),
    [
        ("ASP", 8.0, 7.0, "ASH"),
        ("ASP", 4.0, 7.0, "ASP"),
        ("GLU", 8.0, 7.0, "GLH"),
        ("GLU", 4.0, 7.0, "GLU"),
        ("CYS", 8.0, 7.0, "CYS"),
        ("CYS", 6.0, 7.0, "CYM"),
        ("LYS", 10.0, 7.0, "LYS"),
        ("LYS", 6.0, 7.0, "LYN"),
        ("HIS", 8.0, 7.0, "HIP"),
    ],
)
def test_pka_state_rules(resname, pka, ph, expected):
    decision = decide_residue_state(
        residue(resname),
        record=record("HIS" if resname == "HIS" else resname, pka),
        ph=ph,
        method="propka",
    )

    assert decision is not None
    assert decision.final_state == expected


def test_neutral_his_under_propka_fails_without_manual_resolution():
    with pytest.raises(PkaRuleError) as excinfo:
        decide_residue_state(residue("HIS"), record=record("HIS", 6.0), ph=7.0, method="propka")

    assert "requires HID/HIE assignment" in str(excinfo.value)


def test_neutral_his_under_propka_xtb_is_sent_to_xtb():
    decision = decide_residue_state(
        residue("HIS"),
        record=record("HIS", 6.0),
        ph=7.0,
        method="propka_xtb_his",
    )

    assert decision is not None
    assert decision.needs_xtb
    assert decision.final_state is None


def test_arg_remains_arg_and_warns_if_predicted_deprotonated():
    decision = decide_residue_state(residue("ARG"), record=record("ARG", 6.0), ph=7.0, method="propka")

    assert decision is not None
    assert decision.final_state == "ARG"
    assert any("ARG deprotonation" in warning for warning in decision.warnings)


def test_pka_close_to_ph_generates_warning():
    decision = decide_residue_state(residue("ASP"), record=record("ASP", 7.2), ph=7.0, method="propka")

    assert decision is not None
    assert any("within 0.5" in warning for warning in decision.warnings)


@pytest.mark.parametrize("input_state", ["HID", "HIE", "ASH", "GLH", "LYN", "CYM", "CYX"])
def test_input_amber_state_is_preserved(input_state):
    propka_name = {"ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "CYM": "CYS", "CYX": "CYS"}.get(
        input_state,
        "HIS",
    )
    decision = decide_residue_state(
        residue(input_state),
        record=record(propka_name, 1.0),
        ph=7.0,
        method="propka_xtb_his",
    )

    assert decision is not None
    assert decision.final_state == input_state
    assert decision.source == "input_state"
