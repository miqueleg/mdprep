from pathlib import Path

from mdprep.config.loader import load_manifest


def test_execution_paths_resolve_from_manifest_directory(tmp_path: Path):
    manifest_dir = tmp_path / "project"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "system.yaml"
    manifest_path.write_text(
        """
project:
  name: portable
  input_structure: input/system.pdb
  output_dir: output
structure: {}
protein: {forcefield: ff14SB, water_model: TIP3P}
protonation: {method: manual_only}
disulfides: {}
ligands:
  - id: ligand
    selector: {resname: LIG}
    net_charge: 0
    atom_types: gaff2
    charge_method: user_mol2
    user_mol2: parameters/ligand.mol2
    user_frcmod: parameters/ligand.frcmod
metals: []
solvation: {}
validation: {}
""",
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path, resolve_paths=True)

    assert manifest.project.input_structure == str(
        (manifest_dir / "input/system.pdb").resolve()
    )
    assert manifest.project.output_dir == str((manifest_dir / "output").resolve())
    assert manifest.ligands[0].user_mol2 == str(
        (manifest_dir / "parameters/ligand.mol2").resolve()
    )


def test_executable_names_remain_path_independent(tmp_path: Path):
    manifest_path = tmp_path / "system.yaml"
    manifest_path.write_text(
        """
project: {name: portable, input_structure: input.pdb, output_dir: prepared}
structure: {}
protein: {forcefield: ff14SB, water_model: TIP3P}
protonation:
  method: propka_xtb_his
  propka: {executable: propka3}
  histidine:
    neutral_tautomer_method: xtb
    xtb: {executable: xtb}
disulfides: {}
ligands: []
metals: []
solvation: {}
validation: {}
""",
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path, resolve_paths=True)

    assert manifest.protonation.propka.executable == "propka3"
    assert manifest.protonation.histidine.xtb.executable == "xtb"
