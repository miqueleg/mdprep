

# --------------------------------------------------------------------------
# Two flags every generated manifest carried, that nothing read
# --------------------------------------------------------------------------


def test_a_guessed_force_field_parameter_is_noticed(tmp_path):
    """parmchk2 writes "ATTN, need revision" beside a term it had to
    estimate rather than look up. Nothing read the frcmod files, so a ligand
    built on invented parameters looked exactly like one built on real
    ones -- while the manifest promised fail_on_missing_parameters: true."""

    from mdprep.validation.topology import guessed_parameters

    good = tmp_path / "clean.frcmod"
    good.write_text("BOND\nc3-c3  303.1   1.535\n", encoding="utf-8")
    bad = tmp_path / "guessed.frcmod"
    bad.write_text(
        "BOND\nc3-xx  300.0   1.500   same as c3-c3, penalty score=  8.0\n"
        "ANGLE\nc3-xx-c3   50.0  109.5   ATTN, need revision\n",
        encoding="utf-8",
    )

    found = dict(guessed_parameters([good, bad, tmp_path / "absent.frcmod"]))

    assert str(good) not in found
    assert found[str(bad)] == 1
