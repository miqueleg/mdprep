from mdprep.config.models import HistidineXtbConfig
from mdprep.protonation.xtb_runner import build_xtb_command


def test_gfn2_opt_command_contains_gfn_and_opt():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gfn2", mode="opt", opt_level="loose"),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="xtb",
    )

    assert command[:4] == ["xtb", "HID.xyz", "--gfn", "2"]
    assert command[4:6] == ["--opt", "loose"]


def test_gfn2_sp_command_has_no_opt():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gfn2", mode="sp"),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="xtb",
    )

    assert "--gfn" in command
    assert "--opt" not in command


def test_gxtb_sp_command_contains_gxtb_and_no_opt():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gxtb", mode="sp"),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="gxtb",
    )

    assert "--gxtb" in command
    assert "--opt" not in command


def test_gxtb_opt_command_is_supported():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gxtb", mode="opt", opt_level="normal"),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="gxtb",
    )

    assert "--gxtb" in command
    assert command[command.index("--opt") + 1] == "normal"


def test_solvent_adds_alpb_and_extra_args_are_last():
    command = build_xtb_command(
        config=HistidineXtbConfig(extra_args=["--parallel", "2"], solvent="water"),
        xyz_path="HID.xyz",
        cluster_charge=-1,
        executable="xtb",
    )

    assert command[command.index("--alpb") + 1] == "water"
    assert command[-2:] == ["--parallel", "2"]


def test_duplicate_gxtb_is_not_added():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gxtb", mode="sp", extra_args=["--gxtb"]),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="gxtb",
    )

    assert command.count("--gxtb") == 1
