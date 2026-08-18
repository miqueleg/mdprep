import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

NOTEBOOK = Path("examples/mdprep_interactive_colab.ipynb")
PIPELINE = Path("examples/colab/mdprep_colab_pipeline.py")


def _notebook() -> dict[str, object]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    return notebook


def _source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def _all_source() -> str:
    return "\n".join(_source(cell) for cell in _notebook()["cells"])


def _embedded_pipeline() -> str:
    cells = [
        cell
        for cell in _notebook()["cells"]
        if "mdprep-colab-source" in cell.get("metadata", {}).get("tags", [])
    ]
    assert [cell["metadata"]["tags"][1] for cell in cells] == ["pipeline"]
    assert cells[0]["metadata"]["cellView"] == "form"
    assert cells[0]["metadata"]["jupyter"]["source_hidden"] is True
    return "".join(_source(cell) for cell in cells)


def _load_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MDPREP_COLAB_WORKDIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "py3Dmol", types.ModuleType("py3Dmol"))
    ipython = types.ModuleType("IPython")
    display_module = types.ModuleType("IPython.display")
    display_module.HTML = lambda value: value
    display_module.display = lambda value: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    name = f"mdprep_colab_pipeline_{id(tmp_path)}"
    spec = importlib.util.spec_from_file_location(name, PIPELINE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _manifest_options() -> dict[str, object]:
    return {
        "project_name": "bonded_colab_test",
        "forcefield": "ff14SB",
        "water_model": "TIP3P",
        "protonation_method": "propka_xtb_his",
        "ph": 7.0,
        "histidine_method": "xtb",
        "aa_overrides": "",
        "solvate": True,
        "buffer_angstrom": 10.0,
        "salt_molar": 0.15,
        "md_mode": "none",
        "production_steps": 500000,
        "md_platform": "auto",
    }


def _bonded_chemistry(module, backend: str):
    akg = module.ComponentKey("A", "AKG", 402)
    iron = module.ComponentKey("A", "FE2", 403)
    chemistry = {
        akg: {
            "role": "cofactor",
            "charge": -2,
            "multiplicity": 1,
            "formula": "C5H4O5",
            "atom_types": "gaff2",
            "charge_method": "mcpb_resp_pyscf",
        },
        iron: {
            "role": "metal",
            "metal_model": "bonded_mcpb",
            "charge": 2,
            "multiplicity": 5,
            "element": "Fe",
            "atom_name": "FE",
            "provisional_parameter_set": "12_6_4",
            "cutoff_angstrom": 2.8,
            "coordinators": [
                {
                    "chain": "A",
                    "resname": "AKG",
                    "resid": 402,
                    "icode": None,
                    "atom_name": "O1",
                },
                {
                    "chain": "A",
                    "resname": "HIS",
                    "resid": 134,
                    "icode": None,
                    "atom_name": "NE2",
                },
            ],
            "additional_residues": [],
            "qm_component_keys": [akg],
            "hessian_backend": backend,
            "pyscf_threads": 2,
            "pyscf_memory_mb": 4000,
            "scf_algorithm": "diis",
            "movable_atoms": "active_region_hydrogens",
            "allow_unusual_links": True,
            "charge_restraint": "backbone_heavy",
        },
    }
    return chemistry


def test_colab_uses_only_native_sequential_cells() -> None:
    source = _all_source()

    assert "#@param" in source
    assert "from google.colab import files" in source
    assert "files.upload()" in source
    assert "files.download" in source
    assert "input(" in source
    assert "show_structure(" in source
    assert "py3Dmol" in source
    assert "ipywidgets" not in source
    assert "launch()" not in source
    assert "enable_custom_widget_manager" not in source
    assert "_WIZARD_GZ_B64" not in source
    assert "base64" not in source


def test_notebook_contains_readable_reviewed_pipeline() -> None:
    embedded = _embedded_pipeline()

    reviewed = PIPELINE.read_text(encoding="utf-8")
    assert ast.dump(ast.parse(embedded)) == ast.dump(ast.parse(reviewed))
    compile(embedded, str(PIPELINE), "exec")
    assert embedded.startswith("#@title mdprep helper functions - run once")
    assert "display(HTML(view._make_html()))" in embedded


def test_ordinary_code_cells_compile() -> None:
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = _source(cell)
        if "%pip install" in source:
            continue
        compile(source, f"{NOTEBOOK}:cell-{index}", "exec")


def test_pipeline_requires_explicit_reviews_and_validates_manifest() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    assert "Set CONFIRM_CLEANUP=True" in source
    assert 'charge = _required_int("Net charge")' in source
    assert 'multiplicity = _required_int("Multiplicity"' in source
    assert "Accept this component formula and hydrogenation" in source
    assert "template_charge != charge" in source
    assert "Hydrogen-template action" in source
    assert "RDKit reads the SMILES net formal charge" in source
    assert "O=C([O-])C(=O)CCC(=O)[O-]" in source
    assert "resonance-equivalent atoms; that warning alone is not a failure" in source
    assert "downloaded RCSB AKG CCD entry is the neutral acid" in source
    assert "Accept exactly these MCPB bonds" in source
    assert "no bond is selected automatically" in source
    assert "Explicit MCPB coordinator atoms, comma separated" in source
    assert "Additional complete MCPB residues, comma separated" in source
    assert "MCPB small-model total charge" not in source
    assert "Complete QM-region multiplicity for GFN2" not in source
    assert "Automatically derived electronic state" in source
    assert 'value.split(";")' not in source
    assert '"mcpb_resp_pyscf"' in source
    assert "one independent bonded MCPB site" in source
    assert '"remove_unknown_heterogens": False' in source
    assert "ManifestConfig.model_validate(manifest)" in source


def test_akg_smiles_charge_is_explicit_and_detected_before_mapping() -> None:
    pytest.importorskip("rdkit")
    source = PIPELINE.read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_smiles_formal_charge"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PIPELINE), "exec"), namespace)
    formal_charge = namespace["_smiles_formal_charge"]

    assert formal_charge("O=C(O)C(=O)CCC(=O)O") == 0
    assert formal_charge("O=C([O-])C(=O)CCC(=O)[O-]") == -2


@pytest.mark.parametrize(
    ("backend", "expected_backend", "expected_xtb_model"),
    [
        ("gxtb", "xtb", "gxtb"),
        ("b3lyp_6-31g*", "pyscf", None),
    ],
)
def test_colab_builds_valid_bonded_mcpb_hessian_manifests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    expected_backend: str,
    expected_xtb_model: str | None,
) -> None:
    module = _load_pipeline(monkeypatch, tmp_path)
    manifest = module.build_manifest(
        _bonded_chemistry(module, backend),
        _manifest_options(),
    )

    site = manifest["metals"][0]
    assert site["model"] == "bonded_mcpb"
    assert site["mcpb"]["workflow"] == "pyscf"
    assert len(site["mcpb"]["bonds"]) == 2
    assert site["mcpb"]["small_model_charge"] == 0
    assert site["mcpb"]["large_model_charge"] == 0
    assert site["mcpb"]["small_model_spin"] == 5
    assert site["mcpb"]["large_model_spin"] == 5
    pyscf = site["mcpb"]["pyscf"]
    assert pyscf["geometry_source"] == "qmmm_refinement"
    assert pyscf["hessian_backend"] == expected_backend
    assert pyscf["method"] == "B3LYP"
    assert pyscf["basis"] == "6-31G*"
    if expected_xtb_model is None:
        assert "xtb" not in pyscf
    else:
        assert pyscf["xtb"]["model"] == expected_xtb_model
        assert pyscf["xtb"]["expected_executable_sha256"] == (
            "1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26"
        )
    assert manifest["refinement"]["enabled"] is True
    assert manifest["refinement"]["movable_atoms"] == "active_region_hydrogens"
    assert manifest["refinement"]["qm_components"]["ligands"] == [
        "cofactor_A_AKG_402"
    ]
    assert manifest["refinement"]["total_qm_multiplicity"] == 5


def test_colab_derives_besd_cluster_state_from_components_and_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_pipeline(monkeypatch, tmp_path)
    chemistry = _bonded_chemistry(module, "gxtb")
    iron = module.ComponentKey("A", "FE2", 403)
    akg = module.ComponentKey("A", "AKG", 402)
    substrate = module.ComponentKey("A", "LYS", 401)
    chemistry[substrate] = {
        "role": "ligand",
        "charge": 1,
        "multiplicity": 1,
        "formula": "C6H15N2O2",
        "atom_types": "gaff2",
        "charge_method": "am1bcc",
    }
    chemistry[iron]["coordinators"].append(
        {
            "chain": "A",
            "resname": "ASP",
            "resid": 144,
            "icode": None,
            "atom_name": "OD1",
        }
    )
    chemistry[iron]["qm_component_keys"] = [akg, substrate]
    options = _manifest_options()
    options["aa_overrides"] = "A:ASP:144=ASP"

    derived = module._derive_bonded_electronic_state(
        iron,
        chemistry[iron],
        chemistry,
        module.parse_overrides(options["aa_overrides"]),
    )
    assert derived["total_qm_charge"] == 0

    manifest = module.build_manifest(chemistry, options)

    mcpb = manifest["metals"][0]["mcpb"]
    assert mcpb["small_model_charge"] == -1
    assert mcpb["large_model_charge"] == -1
    assert mcpb["small_model_spin"] == 5
    assert manifest["refinement"]["total_qm_multiplicity"] == 5
    assert manifest["protonation"]["overrides"] == [
        {
            "selector": {
                "chain": "A",
                "resname": "ASP",
                "resid": 144,
                "icode": None,
            },
            "state": "ASP",
            "reason": "Explicitly selected in the mdprep Colab workflow",
        }
    ]


def test_colab_refuses_to_guess_open_shell_coupling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_pipeline(monkeypatch, tmp_path)
    chemistry = _bonded_chemistry(module, "gxtb")
    akg = module.ComponentKey("A", "AKG", 402)
    chemistry[akg]["multiplicity"] = 3

    with pytest.raises(ValueError, match="spin coupling"):
        module.build_manifest(chemistry, _manifest_options())


def test_colab_pins_and_functionally_verifies_official_gxtb() -> None:
    source = _all_source()

    assert "grimme-lab/g-xtb/releases/download/v2.0.1" in source
    assert "cdfb81164f8d3300fc451dbbf2414528b533e71f7a30ec824b564cd30d6d9e5c" in source
    assert "1b4e30b68ed4e88b4075f60d97f4756ee440fe92d3294cc20826d53f8121cd26" in source
    assert 'platform.system() != "Linux"' in source
    assert '[GXTB_EXECUTABLE, "h2.xyz", "--gxtb", "--hess"' in source
    assert '"hessian_backend": "xtb"' in source
    assert '"model": "gxtb"' in source
    assert '"expected_executable_sha256": GXTB_EXECUTABLE_SHA256' in source
    assert "gxtb_hessian_smoke.log" in source


def test_colab_runtime_pins_and_import_checks_ash() -> None:
    source = _all_source()

    assert "git+https://github.com/RagnarB83/ash.git@{ASH_COMMIT}" in source
    assert "9f6fefc83dfaa8a7d5ba7b382f4a6ea2052b6d97" in source
    assert "import ash, geometric, openmm" in source
    assert "ASH/OpenMM/geomeTRIC imports: PASS" in source
