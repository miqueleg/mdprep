import pytest

from mdprep.ambertools.antechamber import build_antechamber_command, run_antechamber
from mdprep.ambertools.commands import AmberToolsError
from mdprep.ambertools.parmchk2 import build_parmchk2_command
from mdprep.external.runner import CommandResult
from tests.test_structure_normalize import ligand_entry, make_manifest, manifest_data


def ligand_config():
    data = manifest_data("tests/data/protein_two_ligands.pdb")
    entry = ligand_entry("sub_501", "B", "SUB", 501)
    entry["net_charge"] = -1
    entry["multiplicity"] = 2
    data["ligands"] = [entry]
    return make_manifest(data).ligands[0]


def test_antechamber_am1bcc_command_contains_required_flags():
    command = build_antechamber_command(
        executable="antechamber",
        input_pdb="ligand.pdb",
        output_mol2="ligand.antechamber.mol2",
        residue_name="SUB",
        ligand=ligand_config(),
    )

    assert command[command.index("-fi") + 1] == "pdb"
    assert command[command.index("-fo") + 1] == "mol2"
    assert command[command.index("-c") + 1] == "bcc"
    assert command[command.index("-nc") + 1] == "-1"
    assert command[command.index("-m") + 1] == "2"
    assert command[command.index("-at") + 1] == "gaff2"


def test_parmchk2_command_contains_required_flags():
    command = build_parmchk2_command(
        executable="parmchk2",
        input_mol2="ligand.mol2",
        output_frcmod="ligand.frcmod",
        ligand=ligand_config(),
    )

    assert command[command.index("-f") + 1] == "mol2"
    assert command[command.index("-s") + 1] == "gaff2"


def test_antechamber_failure_includes_command_and_log_tails(monkeypatch, tmp_path):
    ligand = ligand_config()

    monkeypatch.setattr("mdprep.ambertools.antechamber.which_executable", lambda name: "/bin/antechamber")

    def fake_run_command(command, *, cwd=None, **kwargs):
        return CommandResult(
            command=tuple(command),
            cwd=str(cwd),
            returncode=1,
            stdout="\n".join(f"stdout {index}" for index in range(30)),
            stderr="bad chemistry\nmissing bond information",
            runtime_seconds=0.1,
        )

    monkeypatch.setattr("mdprep.ambertools.antechamber.run_command", fake_run_command)

    with pytest.raises(AmberToolsError) as excinfo:
        run_antechamber(
            ligand=ligand,
            input_pdb=tmp_path / "ligand.pdb",
            output_mol2=tmp_path / "ligand.mol2",
            residue_name="SUB",
            work_dir=tmp_path,
        )

    message = str(excinfo.value)
    assert "Command: /bin/antechamber" in message
    assert "stdout tail:" in message
    assert "stdout 29" in message
    assert "stderr tail:" in message
    assert "missing bond information" in message


def test_antechamber_run_uses_absolute_paths_when_workdir_differs(monkeypatch, tmp_path):
    ligand = ligand_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mdprep.ambertools.antechamber.which_executable", lambda name: "/bin/antechamber")
    input_pdb = tmp_path / "out" / "prepared" / "ligands" / "sub" / "input" / "sub.pdb"
    output_mol2 = tmp_path / "out" / "prepared" / "ligands" / "sub" / "parameters" / "sub.mol2"
    input_pdb.parent.mkdir(parents=True)
    output_mol2.parent.mkdir(parents=True)
    input_pdb.write_text("END\n", encoding="utf-8")

    def fake_run_command(command, *, cwd=None, **kwargs):
        assert command[command.index("-i") + 1] == str(input_pdb)
        assert command[command.index("-o") + 1] == str(output_mol2)
        assert str(command[command.index("-i") + 1]).startswith("/")
        output_mol2.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")
        return CommandResult(tuple(command), str(cwd), 0, "", "", 0.1)

    monkeypatch.setattr("mdprep.ambertools.antechamber.run_command", fake_run_command)

    run_antechamber(
        ligand=ligand,
        input_pdb="out/prepared/ligands/sub/input/sub.pdb",
        output_mol2="out/prepared/ligands/sub/parameters/sub.mol2",
        residue_name="SUB",
        work_dir="out/prepared/ligands/sub/parameters",
    )


def test_parmchk2_run_uses_absolute_paths_when_workdir_differs(monkeypatch, tmp_path):
    ligand = ligand_config()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mdprep.ambertools.parmchk2.which_executable", lambda name: "/bin/parmchk2")
    input_mol2 = tmp_path / "out" / "prepared" / "ligands" / "sub" / "parameters" / "sub.mol2"
    output_frcmod = tmp_path / "out" / "prepared" / "ligands" / "sub" / "parameters" / "sub.frcmod"
    input_mol2.parent.mkdir(parents=True)
    input_mol2.write_text("@<TRIPOS>MOLECULE\n", encoding="utf-8")

    def fake_run_command(command, *, cwd=None, **kwargs):
        assert command[command.index("-i") + 1] == str(input_mol2)
        assert command[command.index("-o") + 1] == str(output_frcmod)
        assert str(command[command.index("-i") + 1]).startswith("/")
        output_frcmod.write_text("MASS\n", encoding="utf-8")
        return CommandResult(tuple(command), str(cwd), 0, "", "", 0.1)

    monkeypatch.setattr("mdprep.ambertools.parmchk2.run_command", fake_run_command)

    from mdprep.ambertools.parmchk2 import run_parmchk2

    run_parmchk2(
        ligand=ligand,
        input_mol2="out/prepared/ligands/sub/parameters/sub.mol2",
        output_frcmod="out/prepared/ligands/sub/parameters/sub.frcmod",
        work_dir="out/prepared/ligands/sub/parameters",
    )
