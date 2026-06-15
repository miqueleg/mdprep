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

    @model_validator(mode="after")
    def validate_gxtb_mode(self) -> "HistidineXtbConfig":
        if self.model == "gxtb" and self.mode == "opt":
            raise ValueError(
                "g-xTB optimization mode is not supported in mdprep v0.1; use mode: sp."
            )
        return self


class HistidineConfig(StrictModel):
    neutral_tautomer_method: Literal["manual", "xtb"] = "xtb"
    xtb: HistidineXtbConfig = Field(default_factory=HistidineXtbConfig)


class ProtonationConfig(StrictModel):
    ph: float = 7.0
    method: Literal["manual_only", "propka", "propka_xtb_his"]
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


class RespFittingConfig(StrictModel):
    backend: Literal["auto", "psiresp", "native"] = "auto"
    total_charge_constraint: bool = True
    stage_2: bool = True


class QmmespConfig(StrictModel):
    qm_engine: Literal["pyscf"] = "pyscf"
    method: str = "HF"
    basis: str = "6-31G*"
    embedding_cutoff_angstrom: float = Field(default=12.0, gt=0)
    resp_fitting: RespFittingConfig = Field(default_factory=RespFittingConfig)


class LigandConfig(StrictModel):
    id: str
    selector: ResidueSelector
    net_charge: int
    multiplicity: int = Field(default=1, ge=1)
    atom_types: Literal["gaff", "gaff2"]
    charge_method: Literal["am1bcc", "gas_resp_pyscf", "qmmesp_pyscf", "user_mol2"]
    user_mol2: str | None = None
    qmmesp: QmmespConfig | None = None

    @model_validator(mode="after")
    def validate_charge_inputs(self) -> "LigandConfig":
        if self.charge_method == "user_mol2" and not self.user_mol2:
            raise ValueError("charge_method: user_mol2 requires user_mol2")
        if self.charge_method == "qmmesp_pyscf" and self.qmmesp is None:
            raise ValueError("charge_method: qmmesp_pyscf requires qmmesp")
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

