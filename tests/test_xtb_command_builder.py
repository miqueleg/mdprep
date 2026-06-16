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


def test_scf_iterations_and_optional_etemp_are_included():
    command = build_xtb_command(
        config=HistidineXtbConfig(
            scf_iterations=750,
            electronic_temperature_kelvin=1000.0,
        ),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="xtb",
    )

    assert command[command.index("--iterations") + 1] == "750"
    assert command[command.index("--etemp") + 1] == "1000.0"


def test_opt_command_can_include_xcontrol_input_before_extra_args():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gfn2", mode="opt", extra_args=["--parallel", "2"]),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="xtb",
        input_path="HID_xtb.inp",
    )

    assert command[command.index("--input") + 1] == "HID_xtb.inp"
    assert command[-2:] == ["--parallel", "2"]


def test_duplicate_gxtb_is_not_added():
    command = build_xtb_command(
        config=HistidineXtbConfig(model="gxtb", mode="sp", extra_args=["--gxtb"]),
        xyz_path="HID.xyz",
        cluster_charge=0,
        executable="gxtb",
    )

    assert command.count("--gxtb") == 1
