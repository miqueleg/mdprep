"""Pydantic models for mdprep YAML manifests."""

from __future__ import annotations

from math import isfinite
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


class LigandResidueSelector(StrictModel):
    chain: str | None = None
    resname: str
    resid: int | None = None
    icode: str | None = None


class AtomSelector(ResidueSelector):
    """Select one named atom in one explicitly identified residue."""

    atom_name: str = Field(min_length=1)


class ProjectConfig(StrictModel):
    name: str
    input_structure: str
    output_dir: str


class StructureRepairConfig(StrictModel):
    """Explicit, reproducible repair of standard-polymer coordinates."""

    backend: Literal["none", "pdbfixer"] = "none"
    sequence_source: str | None = None
    add_missing_heavy_atoms: bool = False
    missing_residues: list[ResidueSelector] = Field(default_factory=list)
    random_seed: int = Field(default=20260722, ge=0)
    platform: Literal["Reference", "CPU"] = "Reference"
    peptide_bond_min_angstrom: float = Field(default=1.10, gt=0)
    peptide_bond_max_angstrom: float = Field(default=1.50, gt=0)
    require_trans_peptide_bonds: bool = True
    trans_peptide_min_abs_degrees: float = Field(default=120.0, ge=0, le=180)

    @model_validator(mode="after")
    def validate_repair_request(self) -> "StructureRepairConfig":
        if self.peptide_bond_min_angstrom >= self.peptide_bond_max_angstrom:
            raise ValueError(
                "structure.repair.peptide_bond_min_angstrom must be smaller than "
                "peptide_bond_max_angstrom"
            )
        configured = bool(
            self.sequence_source
            or self.add_missing_heavy_atoms
            or self.missing_residues
        )
        if self.backend == "none" and configured:
            raise ValueError(
                "structure.repair settings require backend: pdbfixer"
            )
        if self.backend == "pdbfixer" and not (
            self.add_missing_heavy_atoms or self.missing_residues
        ):
            raise ValueError(
                "structure.repair backend: pdbfixer must explicitly request "
                "add_missing_heavy_atoms and/or list missing_residues"
            )
        if self.missing_residues and self.sequence_source is None:
            raise ValueError(
                "structure.repair.missing_residues requires sequence_source so "
                "PDBFixer can validate residue identity and insertion position"
            )
        selectors = [item.model_dump_json() for item in self.missing_residues]
        if len(selectors) != len(set(selectors)):
            raise ValueError("structure.repair.missing_residues must be unique")
        return self


class StructureConfig(StrictModel):
    keep_crystal_waters: bool = True
    altloc_policy: Literal["highest_occupancy", "first", "fail"] = "highest_occupancy"
    remove_unknown_heterogens: bool = False
    preserve_chain_ids: bool = True
    remove_input_hydrogens: bool = True
    repair: StructureRepairConfig = Field(default_factory=StructureRepairConfig)


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
    add_missing_protein_hydrogens: bool = True
    temporary_hydrogen_random_seed: int = Field(default=20260722, ge=0)
    add_missing_water_hydrogens: bool = True
    water_oh_distance_angstrom: float = Field(default=0.9572, gt=0)
    water_hoh_angle_degrees: float = Field(default=104.52, gt=0, lt=180)
    scf_iterations: int = Field(default=500, ge=1)
    electronic_temperature_kelvin: float | None = Field(default=1000.0, gt=0)
    # xTB otherwise opens one OpenMP thread per hardware thread for every
    # tautomer cluster, and several clusters run concurrently, so the machine
    # oversubscribes badly on clusters far too small to use it.
    num_threads: int = Field(default=1, ge=1)
    omp_stacksize: str = "1G"


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
    type: Literal["merz_kollman"] = "merz_kollman"
    vdw_scale_factors: list[float] = Field(default_factory=lambda: [1.4, 1.6, 1.8, 2.0])
    point_density_per_square_angstrom: float = Field(default=1.0, gt=0)
    exclude_inside_vdw_scale: float = Field(default=1.4, gt=0)
    # Amber RESP reads atom and ESP-point counts with a 2I5 format, so 99,999
    # is the largest representable count. Grid generation fails instead of
    # silently thinning a requested Merz-Kollman surface.
    max_points: int = Field(default=99999, ge=10, le=99999)

    @model_validator(mode="after")
    def validate_layers(self) -> "QmEspGridConfig":
        if not self.vdw_scale_factors:
            raise ValueError("qmmesp.grid.vdw_scale_factors must contain at least one layer")
        if any(not isfinite(scale) or scale <= 0 for scale in self.vdw_scale_factors):
            raise ValueError("qmmesp.grid.vdw_scale_factors must all be finite and positive")
        if len(set(self.vdw_scale_factors)) != len(self.vdw_scale_factors):
            raise ValueError("qmmesp.grid.vdw_scale_factors must not contain duplicate layers")
        if self.vdw_scale_factors != sorted(self.vdw_scale_factors):
            raise ValueError("qmmesp.grid.vdw_scale_factors must be ordered from inner to outer")
        return self


class RespFittingConfig(StrictModel):
    """Canonical Amber two-stage RESP settings.

    QMMESP and gas-phase RESP calculations intentionally do not expose a
    simplified in-process approximation.  ``respgen`` supplies Amber's
    standard stage-one/stage-two restraints and equivalence constraints, and
    ``resp`` performs both fits.
    """

    backend: Literal["ambertools"] = "ambertools"
    stage_2: Literal[True] = True


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
    # None matches the reference QMMESP workflow: every selected non-target
    # atom in the provisional Amber system participates in electrostatic
    # embedding.  A finite cutoff is an explicit user approximation.
    embedding_cutoff_angstrom: float | None = Field(default=None, gt=0)
    scf_charge: int | None = None
    scf_spin: int | None = Field(default=None, ge=0)
    max_cycle: int = Field(default=100, ge=1)
    conv_tol: float = Field(default=1.0e-9, gt=0)
    num_threads: int = Field(default=1, ge=1)
    max_memory_mb: int = Field(default=4000, ge=256)
    grid: QmEspGridConfig = Field(default_factory=QmEspGridConfig)
    resp_fitting: RespFittingConfig = Field(default_factory=RespFittingConfig)
    environment: QmmespEnvironmentConfig = Field(default_factory=QmmespEnvironmentConfig)


class LigandConfig(StrictModel):
    id: str
    selector: LigandResidueSelector
    net_charge: int
    multiplicity: int = Field(default=1, ge=1)
    expected_formula: str | None = Field(
        default=None,
        pattern=r"^(?:[A-Z][a-z]?[0-9]*)+$",
        description=(
            "Exact element formula expected in the input PDB, including explicit hydrogens "
            "(for example C5H9NO2)."
        ),
    )
    atom_types: Literal["gaff", "gaff2"]
    charge_method: Literal[
        "am1bcc",
        "gas_resp_pyscf",
        "qmmesp_pyscf",
        "mcpb_resp_pyscf",
        "user_mol2",
    ]
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
        if self.charge_method == "mcpb_resp_pyscf":
            if self.selector.chain is None or self.selector.resid is None:
                raise ValueError(
                    "charge_method: mcpb_resp_pyscf requires an exact ligand selector "
                    "with chain and resid"
                )
            if self.qmmesp is not None:
                raise ValueError(
                    "charge_method: mcpb_resp_pyscf obtains its QM settings from "
                    "metals[].mcpb.pyscf and may not include ligand qmmesp settings"
                )
        if self.qmmesp is not None:
            if self.qmmesp.scf_charge is not None and self.qmmesp.scf_charge != self.net_charge:
                raise ValueError(
                    "qmmesp.scf_charge must equal ligand net_charge; RESP may not fit a different "
                    "electronic charge state"
                )
            expected_spin = self.multiplicity - 1
            if self.qmmesp.scf_spin is not None and self.qmmesp.scf_spin != expected_spin:
                raise ValueError(
                    "qmmesp.scf_spin must equal multiplicity - 1; RESP may not fit a different "
                    "electronic spin state"
                )
        return self


class MetalIonConfig(StrictModel):
    """Identity and formal charge of one metal atom.

    The charge is deliberately mandatory: neither a PDB residue name nor an
    element identifies an oxidation state safely.
    """

    selector: AtomSelector
    element: str = Field(pattern=r"^[A-Z][a-z]?$", min_length=1, max_length=2)
    charge: int = Field(ge=1, le=4)
    multiplicity: int | None = Field(default=None, ge=1)


class NonbondedMetalConfig(StrictModel):
    """Li/Merz Amber ion model selection."""

    parameter_set: Literal["12_6", "12_6_4", "hfe", "iod", "cm"]


class McpbQmArtifactsConfig(StrictModel):
    """Completed external QM calculations consumed by MCPB.py."""

    small_opt_fchk: str | None = None
    small_fc_log: str | None = None
    large_mk_log: str


class McpbBondConfig(StrictModel):
    """One explicitly approved metal--coordinator bond."""

    ion: AtomSelector
    coordinator: AtomSelector


class McpbXtbHessianConfig(StrictModel):
    """Pinned xTB settings for a fixed-geometry MCPB small-model Hessian."""

    model: Literal["gfn2", "gxtb"] = "gfn2"
    executable: str = Field(default="xtb", min_length=1)
    release_tag: str | None = Field(default=None, min_length=1)
    expected_executable_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    accuracy: float = Field(default=0.001, gt=0)
    num_threads: int = Field(default=1, ge=1)
    max_scf_iterations: int | None = Field(default=None, ge=1)


class McpbGxtbHessianConfig(McpbXtbHessianConfig):
    """Backward-compatible g-xTB-only form used by older manifests."""

    model: Literal["gxtb"] = "gxtb"


class McpbMacePolar1HessianConfig(StrictModel):
    """MACE-POLAR-1 settings for a fixed-geometry MCPB Hessian."""

    python_executable: str = Field(default="python", min_length=1)
    model: Literal["polar-1-s", "polar-1-m", "polar-1-l"] = "polar-1-m"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_threads: int = Field(default=1, ge=1)
    expected_model_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    finite_difference_step_angstrom: float = Field(default=0.001, gt=0)
    max_hessian_vector_relative_error: float = Field(default=0.01, gt=0)
    accept_model_license: bool = False


class McpbParameterComparisonConfig(StrictModel):
    """Optional comparison of generated MCPB terms with a reviewed reference."""

    reference_frcmod: str = Field(min_length=1)
    candidate_label: str = Field(default="xTB MCPB", min_length=1)
    reference_label: str = Field(default="MCPB reference", min_length=1)
    require_exact_term_set: bool = True
    max_bond_force_constant_relative_rmse: float | None = Field(
        default=None, ge=0
    )
    max_angle_force_constant_relative_rmse: float | None = Field(
        default=None, ge=0
    )
    max_bond_equilibrium_distance_rmse_angstrom: float | None = Field(
        default=None, ge=0
    )
    max_angle_equilibrium_value_rmse_degrees: float | None = Field(
        default=None, ge=0
    )
    fail_on_thresholds: bool = True


class McpbPySCFConfig(StrictModel):
    """Fixed-geometry calculations that complete the MCPB.py QM stages.

    The geometry can be supplied as an external, identity-preserving optimized
    MCPB small-model PDB, or inherited from mdprep's completed ASH QM/MM
    refinement. An explicit ``test_unoptimized`` status exists only for
    end-to-end workflow tests and is carried into reports as a scientific
    warning. PySCF remains the backend for the embedded large-model ESP. The
    small-model Cartesian Hessian may come from PySCF, GFN2-xTB, g-xTB, or
    MACE-POLAR-1.
    """

    geometry_source: Literal["external_pdb", "qmmm_refinement"] = "external_pdb"
    optimized_small_model_pdb: str | None = Field(default=None, min_length=1)
    small_model_geometry_status: Literal[
        "user_optimized",
        "test_unoptimized",
    ] | None = None
    hessian_backend: Literal["pyscf", "xtb", "gxtb", "mace_polar1"] = "pyscf"
    xtb: McpbXtbHessianConfig | None = None
    gxtb: McpbGxtbHessianConfig | None = None
    mace_polar1: McpbMacePolar1HessianConfig | None = None
    method: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    max_cycle: int = Field(default=150, ge=1)
    conv_tol: float = Field(default=1.0e-9, gt=0)
    scf_algorithm: Literal[
        "diis",
        "newton",
        "adiis_then_diis",
        "adiis_then_newton",
    ] = "diis"
    initial_guess: Literal["minao", "atom", "huckel"] = "minao"
    large_model_restart_checkpoint: str | None = Field(default=None, min_length=1)
    large_model_restart_checkpoint_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    large_model_restart_max_displacement_angstrom: float = Field(
        default=0.0,
        ge=0,
    )
    adiis_precondition_cycles: int = Field(default=15, ge=1)
    adiis_precondition_dft_grid_level: int | None = Field(
        default=None, ge=0, le=9
    )
    level_shift_mode: Literal["static", "dynamic"] = "static"
    level_shift_hartree: float = Field(default=0.0, ge=0)
    damping_factor: float = Field(default=0.0, ge=0, lt=1)
    diis_space: int = Field(default=8, ge=2)
    large_model_density_fitting: bool = False
    large_model_auxbasis: str | None = Field(default=None, min_length=1)
    dft_grid_level: int = Field(default=3, ge=0, le=9)
    num_threads: int = Field(default=1, ge=1)
    max_memory_mb: int = Field(default=8000, ge=256)
    embedding_cutoff_angstrom: float | None = Field(default=None, gt=0)
    embedding_min_distance_angstrom: float = Field(default=1.2, gt=0)
    esp_batch_size: int = Field(default=256, ge=1)
    grid: QmEspGridConfig = Field(default_factory=QmEspGridConfig)

    @model_validator(mode="after")
    def validate_hessian_backend(self) -> "McpbPySCFConfig":
        restart_values = (
            self.large_model_restart_checkpoint,
            self.large_model_restart_checkpoint_sha256,
        )
        if any(value is not None for value in restart_values) and not all(
            value is not None for value in restart_values
        ):
            raise ValueError(
                "mcpb.pyscf large_model_restart_checkpoint and "
                "large_model_restart_checkpoint_sha256 must be provided together"
            )
        if (
            self.large_model_restart_checkpoint is not None
            and self.scf_algorithm != "newton"
        ):
            raise ValueError(
                "mcpb.pyscf large_model_restart_checkpoint requires "
                "scf_algorithm: newton"
            )
        if (
            self.large_model_restart_checkpoint is None
            and self.large_model_restart_max_displacement_angstrom != 0.0
        ):
            raise ValueError(
                "mcpb.pyscf large_model_restart_max_displacement_angstrom "
                "requires large_model_restart_checkpoint"
            )
        if self.scf_algorithm != "diis" and self.level_shift_mode == "dynamic":
            raise ValueError(
                "mcpb.pyscf.level_shift_mode: dynamic is available only with "
                "scf_algorithm: diis"
            )
        if (
            self.adiis_precondition_dft_grid_level is not None
            and self.scf_algorithm
            not in {"adiis_then_diis", "adiis_then_newton"}
        ):
            raise ValueError(
                "mcpb.pyscf.adiis_precondition_dft_grid_level requires "
                "scf_algorithm: adiis_then_diis or adiis_then_newton"
            )
        if self.large_model_auxbasis is not None and not self.large_model_density_fitting:
            raise ValueError(
                "mcpb.pyscf.large_model_auxbasis requires "
                "large_model_density_fitting: true"
            )
        if self.geometry_source == "external_pdb":
            if self.optimized_small_model_pdb is None:
                raise ValueError(
                    "mcpb.pyscf geometry_source: external_pdb requires "
                    "optimized_small_model_pdb"
                )
            if self.small_model_geometry_status is None:
                raise ValueError(
                    "mcpb.pyscf geometry_source: external_pdb requires "
                    "small_model_geometry_status"
                )
        else:
            if self.optimized_small_model_pdb is not None:
                raise ValueError(
                    "mcpb.pyscf.optimized_small_model_pdb may not be supplied when "
                    "geometry_source: qmmm_refinement"
                )
            if self.small_model_geometry_status is not None:
                raise ValueError(
                    "mcpb.pyscf.small_model_geometry_status may not be supplied when "
                    "geometry_source: qmmm_refinement; completion is verified from the "
                    "refinement stage"
                )
        if self.hessian_backend == "xtb" and self.xtb is None:
            raise ValueError(
                "mcpb.pyscf hessian_backend: xtb requires explicit xtb settings"
            )
        if self.hessian_backend != "xtb" and self.xtb is not None:
            raise ValueError(
                "mcpb.pyscf.xtb settings may only be supplied when hessian_backend: xtb"
            )
        if self.hessian_backend == "gxtb" and self.gxtb is None:
            raise ValueError(
                "mcpb.pyscf hessian_backend: gxtb requires explicit gxtb settings"
            )
        if self.hessian_backend != "gxtb" and self.gxtb is not None:
            raise ValueError(
                "mcpb.pyscf.gxtb settings may only be supplied when "
                "hessian_backend: gxtb"
            )
        if self.hessian_backend == "mace_polar1" and self.mace_polar1 is None:
            raise ValueError(
                "mcpb.pyscf hessian_backend: mace_polar1 requires explicit "
                "mace_polar1 settings"
            )
        if self.hessian_backend != "mace_polar1" and self.mace_polar1 is not None:
            raise ValueError(
                "mcpb.pyscf.mace_polar1 settings may only be supplied when "
                "hessian_backend: mace_polar1"
            )
        if (
            self.mace_polar1 is not None
            and not self.mace_polar1.accept_model_license
        ):
            raise ValueError(
                "MACE-POLAR-1 checkpoints use the Academic Software License; set "
                "mcpb.pyscf.mace_polar1.accept_model_license: true only after "
                "reviewing and accepting its terms"
            )
        return self


class McpbConfig(StrictModel):
    """Explicit MCPB.py bonded-model workflow configuration."""

    executable: str = "MCPB.py"
    workflow: Literal["prepare_inputs", "complete", "pyscf"] = "prepare_inputs"
    provisional_nonbonded_parameter_set: Literal[
        "12_6", "12_6_4", "hfe", "iod", "cm"
    ] | None = None
    bonds: list[McpbBondConfig] = Field(min_length=1)
    additional_residues: list[ResidueSelector] = Field(default_factory=list)
    cutoff_angstrom: float = Field(default=2.8, gt=0)
    force_constant_method: Literal[
        "seminario",
        "modified_seminario",
        "empirical",
        "z_matrix",
    ] = "seminario"
    charge_restraint: Literal[
        "all_ligating",
        "backbone_heavy",
        "backbone_all",
        "backbone_and_cb",
    ] = "backbone_heavy"
    software_version: Literal["g03", "g09", "g16", "gau", "gms"] = "g16"
    small_model_charge: int
    small_model_spin: int = Field(ge=1)
    large_model_charge: int
    large_model_spin: int = Field(ge=1)
    scale_factor: float = Field(default=1.0, gt=0)
    large_opt: Literal[0, 1, 2] = 0
    artifacts: McpbQmArtifactsConfig | None = None
    pyscf: McpbPySCFConfig | None = None
    parameter_comparison: McpbParameterComparisonConfig | None = None

    @model_validator(mode="after")
    def validate_complete_artifacts(self) -> "McpbConfig":
        if self.force_constant_method == "z_matrix" and self.software_version == "gms":
            raise ValueError(
                "mcpb force_constant_method: z_matrix is only supported with Gaussian output; "
                "MCPB.py's Z-matrix parser does not read GAMESS output"
            )
        if self.workflow == "pyscf":
            if self.artifacts is not None:
                raise ValueError("mcpb.artifacts may not be supplied with workflow: pyscf")
            if self.pyscf is None:
                raise ValueError("mcpb workflow: pyscf requires mcpb.pyscf settings")
            if self.software_version != "gau":
                raise ValueError(
                    "mcpb workflow: pyscf requires software_version: gau because mdprep "
                    "writes MCPB.py-compatible formatted-checkpoint and MK-ESP adapters"
                )
            if self.force_constant_method not in {"seminario", "modified_seminario"}:
                raise ValueError(
                    "mcpb workflow: pyscf supports the Hessian-based seminario and "
                    "modified_seminario force-constant methods"
                )
            if self.large_opt != 0:
                raise ValueError(
                    "mcpb workflow: pyscf currently performs the MCPB large-model embedded "
                    "single point and therefore requires large_opt: 0"
                )
            return self
        if self.pyscf is not None:
            raise ValueError("mcpb.pyscf settings may only be supplied with workflow: pyscf")
        if self.workflow != "complete":
            if self.artifacts is not None:
                raise ValueError(
                    "mcpb.artifacts may only be supplied with workflow: complete"
                )
            return self
        if self.artifacts is None:
            raise ValueError("mcpb workflow: complete requires artifacts")
        needs_small_log = self.force_constant_method == "z_matrix" or (
            self.force_constant_method in {"seminario", "modified_seminario"}
            and self.software_version == "gms"
        )
        if needs_small_log:
            if not self.artifacts.small_fc_log:
                raise ValueError(
                    f"mcpb force_constant_method: {self.force_constant_method} requires "
                    "artifacts.small_fc_log"
                )
        needs_fchk = (
            self.force_constant_method in {"seminario", "modified_seminario"}
            and self.software_version != "gms"
        )
        if needs_fchk:
            if not self.artifacts.small_opt_fchk:
                raise ValueError(
                    f"mcpb force_constant_method: {self.force_constant_method} requires "
                    "artifacts.small_opt_fchk"
                )
        return self


class MetalSiteConfig(StrictModel):
    """One independent mono- or multi-metal parameterization site."""

    id: str = Field(min_length=1)
    model: Literal["nonbonded", "bonded_mcpb"]
    ions: list[MetalIonConfig] = Field(min_length=1)
    nonbonded: NonbondedMetalConfig | None = None
    mcpb: McpbConfig | None = None

    @model_validator(mode="after")
    def validate_model_settings(self) -> "MetalSiteConfig":
        if self.model == "nonbonded":
            if self.nonbonded is None:
                raise ValueError("metal model: nonbonded requires nonbonded settings")
            if self.mcpb is not None:
                raise ValueError("metal model: nonbonded may not include mcpb settings")
        else:
            if self.mcpb is None:
                raise ValueError("metal model: bonded_mcpb requires mcpb settings")
            if self.nonbonded is not None:
                raise ValueError("metal model: bonded_mcpb may not include nonbonded settings")
            if self.mcpb.provisional_nonbonded_parameter_set is None:
                raise ValueError(
                    "metal model: bonded_mcpb requires "
                    "mcpb.provisional_nonbonded_parameter_set for the pre-MCPB tleap "
                    "hydrogenation build"
                )
        selectors = [ion.selector.model_dump_json() for ion in self.ions]
        if len(selectors) != len(set(selectors)):
            raise ValueError("metal site ion selectors must be unique")
        return self


class RefinementQmComponentsConfig(StrictModel):
    """Explicit non-MM components of the QM region.

    Protein coordinating residues are named per selected metal site. Bonded
    MCPB sites may instead obtain them from their already explicit bond list.
    """

    ligands: list[str] = Field(default_factory=list)
    metal_sites: list[str] = Field(default_factory=list)
    metal_coordinating_residues: dict[str, list[ResidueSelector]] = Field(
        default_factory=dict
    )


class AshRefinementConfig(StrictModel):
    """Executable and optimizer settings for the isolated ASH environment."""

    python_executable: str = Field(min_length=1)
    xtb_executable: str = Field(default="xtb", min_length=1)
    allow_unusual_link_boundaries: bool = False
    max_iterations: int = Field(default=250, ge=1)
    num_cores: int = Field(default=1, ge=1)
    platform: Literal["CPU", "Reference"] = "CPU"
    xtb_max_iterations: int = Field(default=500, ge=1)
    electronic_temperature_kelvin: float = Field(default=300.0, gt=0)
    accuracy: float = Field(default=0.1, gt=0)


class MacePolar1RefinementConfig(StrictModel):
    """Pinned MACE-POLAR-1 settings for mechanical-embedding refinement."""

    model: Literal["polar-1-s", "polar-1-m", "polar-1-l"] = "polar-1-m"
    model_path: str = Field(min_length=1)
    expected_model_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    python_search_paths: list[str] = Field(default_factory=list)
    accept_model_license: bool = False


class RefinementConfig(StrictModel):
    """Optional pre-parameterization active-site QM/MM refinement."""

    enabled: bool = False
    backend: Literal["ash"] = "ash"
    qm_method: Literal["gfn2_xtb", "gxtb", "mace_polar1"] = "gfn2_xtb"
    embedding: Literal["electrostatic", "mechanical"] = "electrostatic"
    qm_components: RefinementQmComponentsConfig = Field(
        default_factory=RefinementQmComponentsConfig
    )
    active_region_cutoff_angstrom: Literal[4.0] = 4.0
    active_water_cutoff_angstrom: Literal[8.0] = 8.0
    include_contact_waters: Literal[True] = True
    movable_atoms: Literal[
        "all_active_region", "active_region_hydrogens"
    ] = "all_active_region"
    post_refinement_protonation: Literal["rerun", "reuse_initial"] = "rerun"
    total_qm_multiplicity: int | None = Field(default=None, ge=1)
    ash: AshRefinementConfig | None = None
    mace_polar1: MacePolar1RefinementConfig | None = None

    @model_validator(mode="after")
    def validate_method_embedding(self) -> "RefinementConfig":
        if (
            self.movable_atoms == "active_region_hydrogens"
            and self.post_refinement_protonation != "reuse_initial"
        ):
            raise ValueError(
                "refinement.movable_atoms: active_region_hydrogens requires "
                "post_refinement_protonation: reuse_initial so the optimized "
                "hydrogen coordinates are not discarded"
            )
        if self.qm_method == "gfn2_xtb":
            if self.embedding != "electrostatic":
                raise ValueError(
                    "refinement qm_method: gfn2_xtb requires embedding: electrostatic"
                )
            if self.mace_polar1 is not None:
                raise ValueError(
                    "refinement.mace_polar1 settings may only be supplied when "
                    "qm_method: mace_polar1"
                )
            return self
        if self.embedding != "mechanical":
            raise ValueError(
                f"refinement qm_method: {self.qm_method} requires explicit "
                "embedding: mechanical; electrostatic embedding is not supported "
                "by this backend"
            )
        if self.qm_method == "mace_polar1":
            if self.mace_polar1 is None:
                raise ValueError(
                    "refinement qm_method: mace_polar1 requires explicit "
                    "refinement.mace_polar1 settings"
                )
            if not self.mace_polar1.accept_model_license:
                raise ValueError(
                    "MACE-POLAR-1 checkpoints use the Academic Software License; set "
                    "refinement.mace_polar1.accept_model_license: true only after "
                    "reviewing and accepting its terms"
                )
        elif self.mace_polar1 is not None:
            raise ValueError(
                "refinement.mace_polar1 settings may only be supplied when "
                "qm_method: mace_polar1"
            )
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


class RoeBrooksDensityConfig(StrictModel):
    """Density-plateau settings for Roe--Brooks step 10."""

    increment_ns: float = Field(default=1.0, gt=0)
    report_interval_ps: float = Field(default=1.0, gt=0)
    window_ps: float = Field(default=300.0, gt=0)
    slope_threshold_g_ml_ps: float = Field(default=1.0e-6, gt=0)
    mean_difference_threshold_g_ml: float = Field(default=0.02, gt=0)
    minimum_points: int = Field(default=50, ge=3)
    maximum_duration_ns: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def validate_sampling_window(self) -> "RoeBrooksDensityConfig":
        if self.window_ps > self.maximum_duration_ns * 1000.0:
            raise ValueError(
                "molecular_dynamics.density.window_ps may not exceed "
                "maximum_duration_ns"
            )
        return self


class ProductionMdConfig(StrictModel):
    """User-selected production length and reporting cadence in MD steps."""

    steps: int = Field(ge=1)
    timestep_fs: Literal[2.0] = 2.0
    trajectory_interval_steps: int = Field(default=10_000, ge=1)
    state_interval_steps: int = Field(default=5_000, ge=1)
    checkpoint_interval_steps: int = Field(default=50_000, ge=1)

    @model_validator(mode="after")
    def validate_intervals(self) -> "ProductionMdConfig":
        for name in (
            "trajectory_interval_steps",
            "state_interval_steps",
            "checkpoint_interval_steps",
        ):
            if getattr(self, name) > self.steps:
                raise ValueError(
                    f"molecular_dynamics.production.{name} may not exceed production.steps"
                )
        return self


class MolecularDynamicsConfig(StrictModel):
    """Optional OpenMM Roe--Brooks equilibration and production stage."""

    enabled: bool = False
    protocol: Literal["roe_brooks_2020"] = "roe_brooks_2020"
    temperature_kelvin: float = Field(default=300.0, gt=0)
    pressure_atmosphere: float = Field(default=1.0, gt=0)
    platform: Literal[
        "auto", "CUDA", "HIP", "OpenCL", "Metal", "CPU", "Reference"
    ] = "auto"
    device_index: int | None = Field(default=None, ge=0)
    precision: Literal["single", "mixed", "double"] = "mixed"
    cpu_threads: int | None = Field(default=None, ge=1)
    random_seed: int = Field(default=20260817, ge=0)
    step08_ensemble: Literal["NPT", "NVT"] = "NPT"
    density: RoeBrooksDensityConfig = Field(default_factory=RoeBrooksDensityConfig)
    production: ProductionMdConfig | None = None

    @model_validator(mode="after")
    def validate_enabled_stage(self) -> "MolecularDynamicsConfig":
        if self.enabled and self.production is None:
            raise ValueError(
                "molecular_dynamics.enabled: true requires an explicit production block "
                "with the requested number of MD steps"
            )
        if not self.enabled and self.production is not None:
            raise ValueError(
                "molecular_dynamics.production may only be supplied when enabled: true"
            )
        if self.device_index is not None and self.platform in {"CPU", "Reference"}:
            raise ValueError(
                "molecular_dynamics.device_index is only valid for auto or accelerator platforms"
            )
        return self


class ManifestConfig(StrictModel):
    project: ProjectConfig
    structure: StructureConfig
    protein: ProteinConfig
    protonation: ProtonationConfig
    disulfides: DisulfideConfig
    ligands: list[LigandConfig] = Field(default_factory=list)
    metals: list[MetalSiteConfig] = Field(default_factory=list)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    solvation: SolvationConfig
    validation: ValidationConfig
    molecular_dynamics: MolecularDynamicsConfig = Field(
        default_factory=MolecularDynamicsConfig
    )

    @model_validator(mode="after")
    def validate_unique_workflow_ids(self) -> "ManifestConfig":
        ligand_ids = [ligand.id for ligand in self.ligands]
        if len(ligand_ids) != len(set(ligand_ids)):
            raise ValueError("ligand ids must be unique")
        metal_ids = [site.id for site in self.metals]
        if len(metal_ids) != len(set(metal_ids)):
            raise ValueError("metal site ids must be unique")
        ion_selectors = [
            ion.selector.model_dump_json()
            for site in self.metals
            for ion in site.ions
        ]
        if len(ion_selectors) != len(set(ion_selectors)):
            raise ValueError("a metal ion atom may belong to only one metal site")
        self._validate_refinement(ligand_ids=ligand_ids, metal_ids=metal_ids)
        self._validate_mcpb_refinement_geometry()
        if any(ligand.charge_method == "qmmesp_pyscf" for ligand in self.ligands):
            missing = [
                site.id
                for site in self.metals
                if site.model == "bonded_mcpb"
                and site.mcpb is not None
                and site.mcpb.provisional_nonbonded_parameter_set is None
            ]
            if missing:
                raise ValueError(
                    "QMMESP with bonded_mcpb metal sites requires an explicit "
                    "mcpb.provisional_nonbonded_parameter_set for provisional electrostatic "
                    f"embedding; missing for {missing}"
                )
        for ligand in self.ligands:
            if ligand.charge_method != "mcpb_resp_pyscf":
                continue
            key = (
                ligand.selector.chain,
                ligand.selector.resid,
                ligand.selector.icode,
            )
            matching_sites: list[MetalSiteConfig] = []
            for site in self.metals:
                if site.model != "bonded_mcpb" or site.mcpb is None:
                    continue
                coordinator_keys = {
                    (
                        bond.coordinator.chain,
                        bond.coordinator.resid,
                        bond.coordinator.icode,
                    )
                    for bond in site.mcpb.bonds
                }
                additional_keys = {
                    (item.chain, item.resid, item.icode)
                    for item in site.mcpb.additional_residues
                }
                if key in coordinator_keys or key in additional_keys:
                    matching_sites.append(site)
            if len(matching_sites) != 1:
                raise ValueError(
                    f"Ligand {ligand.id!r} uses mcpb_resp_pyscf but belongs to "
                    f"{len(matching_sites)} bonded MCPB sites; list it as a coordinator or "
                    "additional_residue in exactly one bonded site"
                )
            site = matching_sites[0]
            assert site.mcpb is not None
            if site.mcpb.workflow not in {"pyscf", "complete"}:
                raise ValueError(
                    f"Ligand {ligand.id!r} uses mcpb_resp_pyscf, so metal site "
                    f"{site.id!r} must use mcpb.workflow: pyscf or resume from "
                    "explicit PySCF-generated artifacts with mcpb.workflow: complete"
                )
            if (
                site.mcpb.workflow == "complete"
                and site.mcpb.software_version != "gau"
            ):
                raise ValueError(
                    f"Ligand {ligand.id!r} uses mcpb_resp_pyscf with "
                    "mcpb.workflow: complete, so software_version must be gau for "
                    "the PySCF-generated Gaussian-compatible artifacts"
                )
        return self

    def _validate_mcpb_refinement_geometry(self) -> None:
        for site in self.metals:
            if site.model != "bonded_mcpb" or site.mcpb is None:
                continue
            pyscf = site.mcpb.pyscf
            if pyscf is None or pyscf.geometry_source != "qmmm_refinement":
                continue
            if not self.refinement.enabled:
                raise ValueError(
                    f"Metal site {site.id!r} uses geometry_source: qmmm_refinement, "
                    "which requires refinement.enabled: true"
                )
            if site.id not in self.refinement.qm_components.metal_sites:
                raise ValueError(
                    f"Metal site {site.id!r} uses geometry_source: qmmm_refinement but "
                    "is not listed in refinement.qm_components.metal_sites"
                )

    def _validate_refinement(
        self,
        *,
        ligand_ids: list[str],
        metal_ids: list[str],
    ) -> None:
        refinement = self.refinement
        if not refinement.enabled:
            return
        if not self.structure.keep_crystal_waters:
            raise ValueError(
                "refinement requires structure.keep_crystal_waters: true so active-site "
                "waters are protonated and included"
            )
        if refinement.ash is None:
            raise ValueError("refinement.enabled: true requires refinement.ash settings")
        selected_ligands = refinement.qm_components.ligands
        selected_metals = refinement.qm_components.metal_sites
        if not selected_ligands and not selected_metals:
            raise ValueError(
                "refinement requires at least one explicit ligand or metal site in "
                "refinement.qm_components"
            )
        if len(selected_ligands) != len(set(selected_ligands)):
            raise ValueError("refinement.qm_components.ligands must be unique")
        if len(selected_metals) != len(set(selected_metals)):
            raise ValueError("refinement.qm_components.metal_sites must be unique")
        unknown_ligands = sorted(set(selected_ligands) - set(ligand_ids))
        if unknown_ligands:
            raise ValueError(
                f"refinement references unknown ligand ids: {unknown_ligands}"
            )
        unknown_metals = sorted(set(selected_metals) - set(metal_ids))
        if unknown_metals:
            raise ValueError(
                f"refinement references unknown metal site ids: {unknown_metals}"
            )
        coordinator_sites = refinement.qm_components.metal_coordinating_residues
        unselected_coordinators = sorted(set(coordinator_sites) - set(selected_metals))
        if unselected_coordinators:
            raise ValueError(
                "refinement metal_coordinating_residues contains unselected metal sites: "
                f"{unselected_coordinators}"
            )

        ligand_by_id = {ligand.id: ligand for ligand in self.ligands}
        site_by_id = {site.id: site for site in self.metals}
        component_multiplicities: list[tuple[str, int]] = []
        for ligand_id in selected_ligands:
            ligand = ligand_by_id[ligand_id]
            if "multiplicity" not in ligand.model_fields_set:
                raise ValueError(
                    f"Refinement ligand {ligand_id!r} must explicitly set multiplicity; "
                    "the normal ligand default is not accepted for a QM region"
                )
            component_multiplicities.append((f"ligand:{ligand_id}", ligand.multiplicity))
        for site_id in selected_metals:
            site = site_by_id[site_id]
            explicit_coordinators = coordinator_sites.get(site_id, [])
            if site.model == "nonbonded" and not explicit_coordinators:
                raise ValueError(
                    f"Refinement metal site {site_id!r} is nonbonded and must explicitly "
                    "list its complete coordinating residues in "
                    "refinement.qm_components.metal_coordinating_residues"
                )
            for index, ion in enumerate(site.ions, start=1):
                if ion.multiplicity is None:
                    raise ValueError(
                        f"Refinement metal ion {index} in site {site_id!r} must explicitly "
                        "set multiplicity"
                    )
                component_multiplicities.append(
                    (f"metal:{site_id}:{index}", ion.multiplicity)
                )
        open_shell = [item for item in component_multiplicities if item[1] > 1]
        if len(open_shell) > 1 and refinement.total_qm_multiplicity is None:
            labels = ", ".join(f"{name}={mult}" for name, mult in open_shell)
            raise ValueError(
                "Multiple open-shell QM components require an explicit "
                f"refinement.total_qm_multiplicity; found {labels}"
            )
