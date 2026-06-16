"""Pydantic models for mdprep YAML manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared manifest keys."""

    model_config = ConfigDict(extra="forbid")


class ResidueSelector(StrictModel):
    chain: str
    resname: str
    resid: int
    icode: str | None = None


class ProjectConfig(StrictModel):
    name: str
    input_structure: str
    output_dir: str


class StructureConfig(StrictModel):
    keep_crystal_waters: bool = True
    altloc_policy: Literal["highest_occupancy", "first", "fail"] = "highest_occupancy"
    remove_unknown_heterogens: bool = False
    preserve_chain_ids: bool = True
    remove_input_hydrogens: bool = True


class ProteinConfig(StrictModel):
    forcefield: Literal["ff14SB", "ff19SB"]
    water_model: Literal["TIP3P", "OPC"]


ProtonationState = Literal[
    "ASP",
    "ASH",
    "GLU",
    "GLH",
    "LYS",
    "LYN",
    "ARG",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "CYS",
    "CYM",
    "CYX",
]


class ProtonationOverride(StrictModel):
    selector: ResidueSelector
    state: ProtonationState
    reason: str


class HistidineXtbConfig(StrictModel):
    executable: str = "xtb"
    model: Literal["gfn2", "gxtb"] = "gfn2"
    mode: Literal["sp", "opt"] = "opt"
    opt_level: Literal["loose", "normal", "tight"] = "loose"
    solvent: str | None = "water"
    cutoff_angstrom: float = Field(default=5.0, gt=0)
    extra_args: list[str] = Field(default_factory=list)
    energy_close_call_kcal_mol: float = Field(default=0.5, ge=0)
    add_missing_water_hydrogens: bool = True
    water_oh_distance_angstrom: float = Field(default=0.9572, gt=0)
    water_hoh_angle_degrees: float = Field(default=104.52, gt=0, lt=180)
    scf_iterations: int = Field(default=500, ge=1)
    electronic_temperature_kelvin: float | None = Field(default=1000.0, gt=0)


class HistidineConfig(StrictModel):
    neutral_tautomer_method: Literal["manual", "xtb"] = "xtb"
    xtb: HistidineXtbConfig = Field(default_factory=HistidineXtbConfig)


class PropkaConfig(StrictModel):
    executable: str | None = None
    fallback_executables: list[str] = Field(default_factory=lambda: ["propka3", "propka"])
    extra_args: list[str] = Field(default_factory=list)
    require_success: bool = True


class ProtonationConfig(StrictModel):
    ph: float = 7.0
    method: Literal["manual_only", "propka", "propka_xtb_his"]
    propka: PropkaConfig = Field(default_factory=PropkaConfig)
    overrides: list[ProtonationOverride] = Field(default_factory=list)
    histidine: HistidineConfig = Field(default_factory=HistidineConfig)


class DisulfidePair(StrictModel):
    a: ResidueSelector
    b: ResidueSelector
    reason: str | None = None


class DisulfideConfig(StrictModel):
    auto_detect: bool = True
    detection_cutoff_angstrom: float = Field(default=2.2, gt=0)
    force: list[DisulfidePair] = Field(default_factory=list)
    forbid: list[DisulfidePair] = Field(default_factory=list)


class QmEspGridConfig(StrictModel):
    type: Literal["connolly"] = "connolly"
    vdw_scale_factors: list[float] = Field(default_factory=lambda: [1.4, 1.6, 1.8, 2.0])
    points_per_atom_per_shell: int = Field(default=60, ge=1)
    exclude_inside_vdw_scale: float = Field(default=1.2, gt=0)
    max_points: int = Field(default=8000, ge=10)


class RespFittingConfig(StrictModel):
    backend: Literal["auto", "psiresp", "native"] = "native"
    total_charge_constraint: bool = True
    restraint: Literal["none", "resp"] = "resp"
    restraint_a: float = Field(default=0.0005, ge=0)
    restraint_b: float = Field(default=0.1, gt=0)
    max_iter: int = Field(default=25, ge=1)
    convergence: float = Field(default=1.0e-6, gt=0)
    stage_2: bool = True


class QmmespEnvironmentConfig(StrictModel):
    include_protein: bool = True
    include_waters: bool = True
    include_other_ligands: bool = True
    exclude_self_ligand: bool = True

    @model_validator(mode="after")
    def require_target_exclusion(self) -> "QmmespEnvironmentConfig":
        if not self.exclude_self_ligand:
            raise ValueError("qmmesp.environment.exclude_self_ligand must be true; target ligand self-embedding is not allowed")
        return self


class QmmespConfig(StrictModel):
    qm_engine: Literal["pyscf"] = "pyscf"
    method: str = "HF"
    basis: str = "6-31G*"
    embedding_cutoff_angstrom: float = Field(default=12.0, gt=0)
    scf_charge: int | None = None
    scf_spin: int | None = Field(default=None, ge=0)
    max_cycle: int = Field(default=100, ge=1)
    conv_tol: float = Field(default=1.0e-9, gt=0)
    grid: QmEspGridConfig = Field(default_factory=QmEspGridConfig)
    resp_fitting: RespFittingConfig = Field(default_factory=RespFittingConfig)
    environment: QmmespEnvironmentConfig = Field(default_factory=QmmespEnvironmentConfig)


class LigandConfig(StrictModel):
    id: str
    selector: ResidueSelector
    net_charge: int
    multiplicity: int = Field(default=1, ge=1)
    atom_types: Literal["gaff", "gaff2"]
    charge_method: Literal["am1bcc", "gas_resp_pyscf", "qmmesp_pyscf", "user_mol2"]
    user_mol2: str | None = None
    user_frcmod: str | None = None
    preserve_atom_names: bool = True
    preserve_coordinates: bool = True
    allow_atom_renaming: bool = False
    allow_coordinate_changes: bool = False
    qmmesp: QmmespConfig | None = None

    @model_validator(mode="after")
    def validate_charge_inputs(self) -> "LigandConfig":
        if self.charge_method == "user_mol2" and not self.user_mol2:
            raise ValueError("charge_method: user_mol2 requires user_mol2")
        if self.charge_method in {"gas_resp_pyscf", "qmmesp_pyscf"} and self.qmmesp is None:
            raise ValueError(f"charge_method: {self.charge_method} requires qmmesp")
        return self


class SolvationConfig(StrictModel):
    enabled: bool = True
    box: Literal["truncated_octahedron", "rectangular"] = "truncated_octahedron"
    buffer_angstrom: float = Field(default=10.0, gt=0)
    neutralize: bool = True
    salt_concentration_molar: float = Field(default=0.15, ge=0)
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"


class ValidationConfig(StrictModel):
    run_openmm_energy_check: bool = True
    fail_on_warnings: bool = False
    fail_on_missing_parameters: bool = True
    fail_on_noninteger_ligand_charge: bool = True


class ManifestConfig(StrictModel):
    project: ProjectConfig
    structure: StructureConfig
    protein: ProteinConfig
    protonation: ProtonationConfig
    disulfides: DisulfideConfig
    ligands: list[LigandConfig] = Field(default_factory=list)
    solvation: SolvationConfig
    validation: ValidationConfig
