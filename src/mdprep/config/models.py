"""Pydantic models for mdprep YAML manifests."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mdprep.structure.classify import STANDARD_AMINO_ACIDS


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


class RemovedResidueConfig(StrictModel):
    """One residue the build leaves out, and why.

    Nothing is removed by inference: a residue goes only when it is named
    here, and the reason goes with it into the structure report. The builder
    writes these for what the user agreed to drop -- crystallisation
    additives, glycans, a covalently attached heterogen, a metal from the
    purification.
    """

    selector: ResidueSelector
    reason: str = Field(min_length=1)


class ModifiedResidueReplacement(StrictModel):
    """A modified residue of a chain, and the standard one it becomes."""

    selector: ResidueSelector
    parent: str

    @field_validator("parent")
    @classmethod
    def validate_parent(cls, value: str) -> str:
        parent = value.strip().upper()
        if parent not in STANDARD_AMINO_ACIDS:
            raise ValueError(
                f"{value!r} is not one of the twenty standard amino acids, so "
                "no force field template can build it in the chain"
            )
        return parent


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
    # What to do about whole residues the deposition never resolved. "fail"
    # requires every one to be approved in missing_residues before anything
    # proceeds. "leave_absent" declines to build the ones not listed and records
    # them, which is what an unresolved expression tag calls for: modelling a
    # disordered His-tag is worse than omitting it.
    unresolved_residues: Literal["fail", "leave_absent"] = "fail"
    # A modified residue inside a chain -- a selenomethionine, an oxidised
    # cysteine, a pyroglutamate -- cannot be built: no force field template
    # bonds it into the chain. Each one listed here is replaced by its
    # standard parent through PDBFixer, which keeps the atoms the two share
    # and builds the rest; where the modification was real chemistry, that
    # chemistry is gone, which is why each is named. Exactly these, and no
    # others: PDBFixer's own table is not consulted.
    replace_modified_residues: list[ModifiedResidueReplacement] = Field(
        default_factory=list
    )
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
            or self.replace_modified_residues
        )
        if self.backend == "none" and configured:
            raise ValueError(
                "structure.repair settings require backend: pdbfixer"
            )
        if self.backend == "pdbfixer" and not (
            self.add_missing_heavy_atoms
            or self.missing_residues
            or self.replace_modified_residues
        ):
            raise ValueError(
                "structure.repair backend: pdbfixer must explicitly request "
                "add_missing_heavy_atoms, replace_modified_residues and/or list "
                "missing_residues"
            )
        keys = [
            (item.selector.chain, item.selector.resname, item.selector.resid, item.selector.icode)
            for item in self.replace_modified_residues
        ]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "structure.repair.replace_modified_residues names a residue more than once"
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
    # Empty keeps every chain. Naming chains restricts the build to them, which
    # a bonded MCPB site requires because MCPB inputs are single-chain, and
    # which crystal structures containing several copies of a complex need.
    keep_chains: list[str] = Field(default_factory=list)
    # Waters bound to a configured metal site are part of that site and are
    # kept whatever this says; it governs the rest.
    keep_crystal_waters: bool = True
    # Free single-atom ions in the deposition -- a chloride in an anion
    # pocket, a sodium on the surface. tleap builds them from its own ion
    # library, so they need no parameters, only a decision to keep them. A
    # halide bound to a metal is part of that site, not one of these.
    keep_crystal_ions: bool = True
    altloc_policy: Literal["highest_occupancy", "first", "fail"] = "highest_occupancy"
    remove_unknown_heterogens: bool = False
    preserve_chain_ids: bool = True
    remove_input_hydrogens: bool = True
    # Residues left out on purpose, each with its reason. Applied before
    # anything else looks at the structure, so a removed residue is never
    # an unknown heterogen, a ligand or a donor.
    remove_residues: list[RemovedResidueConfig] = Field(default_factory=list)
    repair: StructureRepairConfig = Field(default_factory=StructureRepairConfig)

    @model_validator(mode="after")
    def validate_removed_residues(self) -> "StructureConfig":
        keys = [
            (item.selector.chain, item.selector.resname, item.selector.resid, item.selector.icode)
            for item in self.remove_residues
        ]
        duplicates = sorted({key for key in keys if keys.count(key) > 1}, key=str)
        if duplicates:
            raise ValueError(
                "structure.remove_residues names a residue more than once: "
                + ", ".join(f"{c}:{n}{r}{i or ''}" for c, n, r, i in duplicates)
            )
        return self


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
    # A single point by default. The heavy atoms come from the structure and
    # are meant to stay there, so an optimisation only relaxes the hydrogens
    # the tautomer differs by -- and it costs an optimisation per tautomer per
    # histidine to do it. "opt" is supported and constrained properly; it is
    # just not what most runs need.
    mode: Literal["sp", "opt"] = "sp"
    opt_level: Literal["loose", "normal", "tight"] = "loose"
    # Hartree/bohr^2 on each heavy atom when mode is "opt". xTB's $fix does
    # not hold atoms through an approximate-normal-coordinate optimisation --
    # measured drift of over 2 A on atoms it was told to fix -- so they are
    # restrained instead, and the comparison energy is taken from a separate
    # unrestrained single point so the restraint cannot reach it.
    constraint_force_constant: float = Field(default=1.0, gt=0)
    solvent: str | None = "water"
    # "qmmm" puts only the histidine side chain in the QM region and the
    # surroundings in as MM point charges. The side chain is neutral and closed
    # shell whichever tautomer it is, so there is no cluster charge to compute
    # and get wrong. "cluster" is the older whole-fragment treatment, kept for
    # comparison and for systems whose neighbours have no force-field charges.
    embedding: Literal["qmmm", "cluster"] = "qmmm"
    cutoff_angstrom: float = Field(default=5.0, gt=0)
    # Point charges cost almost nothing, so this reaches well past the distance
    # at which a tautomer preference stops changing.
    embedding_cutoff_angstrom: float = Field(default=12.0, gt=0)
    # MM charges nearer than this to a capping hydrogen are redistributed over
    # the rest of their residue instead of polarising the link atom.
    link_exclusion_angstrom: float = Field(default=1.8, ge=0)
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
    # The protonation the user intends, stated explicitly. A crystal structure
    # carries no ligand hydrogens; supplying a SMILES lets mdprep place them
    # without guessing a tautomer or a charge. Hydrogens the input already
    # has on the ligand are replaced by the SMILES's, unless
    # keep_input_hydrogens says otherwise.
    smiles: str | None = None
    # Whether the ligand keeps the hydrogens it has in the input rather than
    # losing them with structure.remove_input_hydrogens. Unset, it keeps them
    # when a user mol2 lists them or when there is no SMILES to place others
    # from. Kept hydrogens are checked against the SMILES's formula.
    keep_input_hydrogens: bool | None = None
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
    # The id of another ligand whose parameters this one is built with: one
    # residue of the same name, parameterised once, in another copy of the
    # protein. tleap builds every residue of a name from one template, so
    # two residues sharing a name share their parameters whether a manifest
    # says so or not; this says so, and says which one they come from. The
    # settings above must repeat that ligand's, so the file reads true.
    parameters_from: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_charge_inputs(self) -> "LigandConfig":
        if self.charge_method == "user_mol2" and not self.user_mol2:
            raise ValueError("charge_method: user_mol2 requires user_mol2")
        if self.keep_input_hydrogens is False:
            if self.user_mol2 is not None:
                raise ValueError(
                    "keep_input_hydrogens: false cannot be used with user_mol2: the mol2 "
                    "has to match the residue atom for atom, its hydrogens included"
                )
            if self.smiles is None:
                raise ValueError(
                    "keep_input_hydrogens: false requires smiles: nothing else would "
                    "place this ligand's hydrogens"
                )
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


class CoordinatingIonConfig(StrictModel):
    """A single-atom non-metal ion bound in a metal site.

    A coordinating halide belongs nowhere else. It is not a metal, so it does
    not go in ions[], where MCPB would treat it as a metal centre. And it
    cannot be a GAFF ligand, because antechamber's semi-empirical step refuses
    a single atom. Amber's Li/Merz sets parameterise F-, Cl-, Br- and I-
    directly, which is what the force-field relaxation uses.

    A polyatomic coordinating anion -- azide, for instance -- has no Amber ion
    parameters and does not need them: antechamber handles it. Configure it
    under ligands[] as a cofactor and name it as an mcpb coordinator, which is
    what alpha-ketoglutarate already does.

    Either way the ion is part of the metal site's QM region.
    """

    selector: AtomSelector
    element: str = Field(pattern=r"^[A-Z][a-z]?$", min_length=1, max_length=2)
    # Anions only: a coordinating cation is a metal and belongs in ions[].
    charge: int = Field(le=-1, ge=-1)
    parameter_set: Literal["12_6", "12_6_4", "hfe", "iod", "cm"] = "12_6"


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

    # g-xTB rather than GFN2, because GFN2 is parameterised for main-group
    # chemistry and describes transition-metal coordination badly: on the BesD
    # iron site it gives one Fe-O bond a negative force constant. g-xTB gets
    # all five right at the same order of cost. It needs its own build, which
    # MDPREP_GXTB_EXECUTABLE can point at once; see docs/metals.md.
    model: Literal["gfn2", "gxtb"] = "gxtb"
    executable: str | None = Field(default=None, min_length=1)
    release_tag: str | None = Field(default=None, min_length=1)
    expected_executable_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    accuracy: float = Field(default=0.001, gt=0)
    num_threads: int = Field(default=1, ge=1)
    max_scf_iterations: int | None = Field(default=None, ge=1)
    # Fermi smearing, for both the Hessian and the GFN2 ESP of the large
    # model. Left unset, xTB uses its own 300 K. An open-shell porphyrin --
    # a low-spin Fe(III) heme -- has d orbitals close enough together that
    # the SCF oscillates there without converging in any number of cycles;
    # at 1000 K it converges in seconds. The builder sets that for a site
    # whose donors include a heme, and the value is in every command record.
    electronic_temperature_kelvin: float | None = Field(default=None, gt=0)
    # Hotter electronic temperatures to converge the SCF from before the
    # Hessian. An open-shell heme's SCF can settle in more than one state of
    # the same spin, and which one a cold start reaches depends on as little
    # as the thread count: 2QFK's high-spin Fe(III) reached two, 27 kcal/mol
    # apart, whose Fe-N force constants differ by up to 1.7 times. With these
    # set, the SCF is converged once straight at electronic_temperature_kelvin
    # and once from each of these, cooled to it, each single-threaded so the
    # thread count cannot decide; the lowest energy wins and the Hessian
    # starts from its wavefunction. Every start is recorded.
    scf_annealing_kelvin: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _annealing_starts_hotter(self) -> "McpbXtbHessianConfig":
        target = self.electronic_temperature_kelvin or 300.0
        for kelvin in self.scf_annealing_kelvin:
            if not kelvin > target:
                raise ValueError(
                    "mcpb.pyscf.xtb.scf_annealing_kelvin must be hotter than the "
                    f"temperature the SCF is cooled to ({target:g} K); got {kelvin:g}."
                )
        return self


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
    # A tight-binding Hessian by default. The DFT alternative costs minutes to
    # hours on a site this size for a quantity the Seminario fit then reduces
    # to a handful of force constants.
    hessian_backend: Literal["pyscf", "xtb", "gxtb", "mace_polar1"] = "xtb"
    # The large model exists to be fitted to an ESP, and that fit sets the
    # charges the finished system carries. "pyscf" does it at DFT cost;
    # "xtb" does it with the GFN2 density in seconds, which is the difference
    # between a method a student can run and one they cannot. GFN2 charges are
    # not RESP charges in the usual sense, and every report says so.
    esp_backend: Literal["pyscf", "xtb"] = "pyscf"
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
            # Filled in rather than demanded: the backend is the default, so
            # requiring a block to restate the defaults would be noise.
            object.__setattr__(self, "xtb", McpbXtbHessianConfig())
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
    # Non-metal ions bound in this site, such as a coordinating halide.
    coordinating_ions: list[CoordinatingIonConfig] = Field(default_factory=list)
    nonbonded: NonbondedMetalConfig | None = None
    mcpb: McpbConfig | None = None
    # The id of the site in another copy of the protein whose parameters
    # this one reuses. A bonded copy is not fitted: it takes the reference
    # site's MCPB.py templates and bond parameters, residue for residue, so
    # its ions, coordinating ions, bonds and additional residues are listed in
    # the same order as the reference's, each the counterpart of the one in
    # the same place. Every other setting repeats the reference's.
    parameters_from: str | None = Field(default=None, min_length=1)

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


class OpenmmRefinementConfig(StrictModel):
    """Force-field hydrogen refinement settings.

    There is no active region here: a force field is cheap enough to relax
    every hydrogen, and freezing those outside an arbitrary radius would put a
    seam through the hydrogen-bond network for no saving.
    """

    platform: Literal["auto", "CUDA", "OpenCL", "CPU", "Reference"] = "auto"
    tolerance_kj_mol_per_nm: float = Field(default=1.0, gt=0)
    max_iterations: int = Field(default=0, ge=0)  # 0 means until converged
    nonbonded_cutoff_nanometer: float = Field(default=1.0, gt=0)


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
    # The refinement only ever moves hydrogens. A force field answers that
    # question in seconds where GFN2 QM/MM takes far longer, so it is the
    # default; the charges and parameters that reach the final build still
    # come from whatever QMMESP or MCPB calculation the manifest configures.
    backend: Literal["openmm", "ash"] = "openmm"
    qm_method: Literal["gfn2_xtb", "gxtb", "mace_polar1"] = "gfn2_xtb"
    embedding: Literal["electrostatic", "mechanical"] = "electrostatic"
    qm_components: RefinementQmComponentsConfig = Field(
        default_factory=RefinementQmComponentsConfig
    )
    active_region_cutoff_angstrom: Literal[4.0] = 4.0
    active_water_cutoff_angstrom: Literal[8.0] = 8.0
    include_contact_waters: Literal[True] = True
    # The default backend only moves hydrogens, so these default to match it.
    # Relaxing heavy atoms means backend: ash and setting both explicitly.
    movable_atoms: Literal[
        "all_active_region", "active_region_hydrogens"
    ] = "active_region_hydrogens"
    post_refinement_protonation: Literal["rerun", "reuse_initial"] = "reuse_initial"
    total_qm_multiplicity: int | None = Field(default=None, ge=1)
    ash: AshRefinementConfig | None = None
    openmm: OpenmmRefinementConfig = Field(
        default_factory=lambda: OpenmmRefinementConfig()
    )
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
    # None means draw a fresh one for this run. Replicates of one system are
    # meant to differ, and they differ through their starting velocities --
    # a fixed seed gives every replicate the same velocities and so the same
    # trajectory, which is the one thing a set of replicates must not be.
    # The drawn value is recorded in the MD report, so a particular replicate
    # can still be repeated by setting it here.
    random_seed: int | None = Field(default=None, ge=0)
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
        fitted = [
            site.id
            for site in self.metals
            if site.model == "bonded_mcpb" and site.parameters_from is None
        ]
        if len(fitted) > 1:
            raise ValueError(
                f"Bonded metal sites {fitted} would each need an MCPB fit of their own; "
                "mdprep fits one site per preparation, because MCPB.py gives every fit "
                "the same residue and atom-type names. A site in another copy of the "
                "protein can reuse that fit with parameters_from; otherwise model all "
                "but one nonbonded"
            )
        self._validate_shared_parameters()
        self._validate_removed_residues_are_not_built()
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

    def _validate_shared_parameters(self) -> None:
        """A block that takes another's parameters must be able to.

        The reference must exist and not itself borrow, every setting must
        repeat the reference's, and a bonded copy must list its residues in
        the order of the reference's, each the counterpart of the one in the
        same place -- residue numbers may differ, the chemistry may not.
        """

        ligands = {ligand.id: ligand for ligand in self.ligands}
        for ligand in self.ligands:
            if ligand.parameters_from is None:
                continue
            reference = ligands.get(ligand.parameters_from)
            label = f"Ligand {ligand.id!r} takes its parameters from {ligand.parameters_from!r}"
            if reference is None:
                raise ValueError(f"{label}, which is not a configured ligand")
            if reference.id == ligand.id or reference.parameters_from is not None:
                raise ValueError(
                    f"{label}; parameters_from must name a ligand that is parameterised "
                    "itself"
                )
            if reference.selector.resname != ligand.selector.resname:
                raise ValueError(
                    f"{label}, but they are {ligand.selector.resname} and "
                    f"{reference.selector.resname}; tleap shares a template by residue "
                    "name, so only residues of one name can share parameters"
                )
            mine = ligand.model_dump(exclude={"id", "selector", "parameters_from"})
            theirs = reference.model_dump(exclude={"id", "selector", "parameters_from"})
            differing = sorted(key for key in mine if mine[key] != theirs[key])
            if differing:
                raise ValueError(
                    f"{label} but differs from it in {', '.join(differing)}; repeat "
                    "the reference's settings, or parameterise it on its own"
                )

        sites = {site.id: site for site in self.metals}
        for site in self.metals:
            if site.parameters_from is None:
                continue
            reference = sites.get(site.parameters_from)
            label = f"Metal site {site.id!r} takes its parameters from {site.parameters_from!r}"
            if reference is None:
                raise ValueError(f"{label}, which is not a configured metal site")
            if reference.id == site.id or reference.parameters_from is not None:
                raise ValueError(
                    f"{label}; parameters_from must name a site that is parameterised "
                    "itself"
                )
            _require_same_site_chemistry(site, reference, label)

        # A cofactor whose charges come from a bonded site's fit can only
        # share them with the matching cofactor of a copy of that site.
        for ligand in self.ligands:
            if ligand.parameters_from is None or ligand.charge_method != "mcpb_resp_pyscf":
                continue
            own = _bonded_site_of(self.metals, ligand)
            source = _bonded_site_of(self.metals, ligands[ligand.parameters_from])
            if own is None or source is None or own.parameters_from != source.id:
                raise ValueError(
                    f"Ligand {ligand.id!r} takes its MCPB-fitted charges from "
                    f"{ligand.parameters_from!r}, so it has to coordinate a copy of "
                    f"that ligand's site, a metal site with parameters_from: "
                    f"{source.id if source is not None else '<its site>'}"
                )

    def _validate_removed_residues_are_not_built(self) -> None:
        """A residue cannot be both left out and built."""

        removed = {
            (item.selector.chain, item.selector.resname, item.selector.resid, item.selector.icode)
            for item in self.structure.remove_residues
        }
        if not removed:
            return
        used: list[tuple[str, tuple[str | None, str, int | None, str | None]]] = []
        for ligand in self.ligands:
            selector = ligand.selector
            used.append(
                (f"ligand {ligand.id!r}", (selector.chain, selector.resname, selector.resid, selector.icode))
            )
        for site in self.metals:
            atoms = [ion.selector for ion in site.ions] + [
                ion.selector for ion in site.coordinating_ions
            ]
            if site.mcpb is not None:
                atoms += [bond.coordinator for bond in site.mcpb.bonds]
                atoms += list(site.mcpb.additional_residues)
            for selector in atoms:
                used.append(
                    (f"metal site {site.id!r}", (selector.chain, selector.resname, selector.resid, selector.icode))
                )
        clashes = sorted(
            {
                f"{label} uses {chain}:{resname}{resid}{icode or ''}"
                for label, (chain, resname, resid, icode) in used
                if (chain, resname, resid, icode) in removed
            }
        )
        if clashes:
            raise ValueError(
                "structure.remove_residues leaves out residues the manifest also builds: "
                + "; ".join(clashes)
            )

    def _validate_mcpb_refinement_geometry(self) -> None:
        for site in self.metals:
            if site.model != "bonded_mcpb" or site.mcpb is None:
                continue
            # A copy reuses its reference's fit; its own geometry is not fitted.
            if site.parameters_from is not None:
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
        if refinement.backend == "ash" and refinement.ash is None:
            raise ValueError(
                "refinement.backend: ash requires refinement.ash settings"
            )
        if refinement.backend == "openmm":
            if refinement.movable_atoms != "active_region_hydrogens":
                raise ValueError(
                    "refinement.backend: openmm only moves hydrogens, so "
                    "movable_atoms must be active_region_hydrogens. Select "
                    "backend: ash to relax heavy atoms as well."
                )
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
        if open_shell and refinement.total_qm_multiplicity is None:
            # Taking the ion's own multiplicity as the region's was right only
            # while the ion was the single open shell and nothing coupled to
            # it, and nothing here can check either. One open-shell component
            # is enough to make the total a question for the chemist: the
            # region is one quantum system with one spin state, and a second
            # unpaired centre can align with the first or against it.
            labels = ", ".join(f"{name}={mult}" for name, mult in open_shell)
            raise ValueError(
                "An open-shell QM region requires an explicit "
                f"refinement.total_qm_multiplicity; found {labels}. State the "
                "multiplicity of the whole QM region, not of its parts"
            )


# Residue names one chemistry goes by: a histidine may be HIS in one copy and
# HID in another, a water HOH or WAT.
_SAME_RESIDUE = (
    frozenset({"HIS", "HID", "HIE", "HIP"}),
    frozenset({"ASP", "ASH"}),
    frozenset({"GLU", "GLH"}),
    frozenset({"CYS", "CYM", "CYX"}),
    frozenset({"LYS", "LYN"}),
    frozenset({"HOH", "WAT", "H2O", "TIP3", "OPC"}),
)


def _residue_family(resname: str) -> frozenset[str]:
    for family in _SAME_RESIDUE:
        if resname in family:
            return family
    return frozenset({resname})


def _require_same_site_chemistry(
    site: "MetalSiteConfig", reference: "MetalSiteConfig", label: str
) -> None:
    if site.model != reference.model:
        raise ValueError(f"{label} but is {site.model}, where it is {reference.model}")
    if len(site.ions) != len(reference.ions):
        raise ValueError(f"{label} but lists {len(site.ions)} ions to its {len(reference.ions)}")
    for own, theirs in zip(site.ions, reference.ions):
        if (own.element, own.charge, own.multiplicity) != (
            theirs.element,
            theirs.charge,
            theirs.multiplicity,
        ):
            raise ValueError(
                f"{label} but its ion {own.selector.resname}{own.selector.resid} differs "
                f"from {theirs.selector.resname}{theirs.selector.resid} in element, charge "
                "or multiplicity"
            )
    if len(site.coordinating_ions) != len(reference.coordinating_ions) or any(
        (own.element, own.charge, own.parameter_set)
        != (theirs.element, theirs.charge, theirs.parameter_set)
        for own, theirs in zip(site.coordinating_ions, reference.coordinating_ions)
    ):
        raise ValueError(
            f"{label} but its coordinating ions are not the reference's, in the same order"
        )
    if site.model == "nonbonded":
        if site.nonbonded != reference.nonbonded:
            raise ValueError(f"{label} but uses another ion parameter set")
        return
    assert site.mcpb is not None and reference.mcpb is not None
    mine = site.mcpb.model_dump(exclude={"bonds", "additional_residues"})
    theirs = reference.mcpb.model_dump(exclude={"bonds", "additional_residues"})
    differing = sorted(key for key in mine if mine[key] != theirs[key])
    if differing:
        raise ValueError(
            f"{label} but its mcpb settings differ in {', '.join(differing)}; a copy "
            "repeats the reference's settings and lists only its own residues"
        )
    if len(site.mcpb.bonds) != len(reference.mcpb.bonds):
        raise ValueError(
            f"{label} but has {len(site.mcpb.bonds)} bonds to the reference's "
            f"{len(reference.mcpb.bonds)}"
        )
    if len(site.mcpb.additional_residues) != len(reference.mcpb.additional_residues):
        raise ValueError(f"{label} but lists a different number of additional residues")
    own_ion_index = {ion.selector.model_dump_json(): index for index, ion in enumerate(site.ions)}
    reference_ion_index = {
        ion.selector.model_dump_json(): index for index, ion in enumerate(reference.ions)
    }
    pairing: dict[tuple[str, int, str | None], tuple[str, int, str | None]] = {}
    for number, (own, theirs) in enumerate(zip(site.mcpb.bonds, reference.mcpb.bonds), start=1):
        if own_ion_index.get(own.ion.model_dump_json()) != reference_ion_index.get(
            theirs.ion.model_dump_json()
        ):
            raise ValueError(
                f"{label} but its bond {number} is to a different ion of the site than "
                "the reference's bond in the same place"
            )
        if own.coordinator.atom_name != theirs.coordinator.atom_name or _residue_family(
            own.coordinator.resname
        ) != _residue_family(theirs.coordinator.resname):
            raise ValueError(
                f"{label} but its bond {number} is to {own.coordinator.resname} "
                f"{own.coordinator.atom_name} where the reference's is to "
                f"{theirs.coordinator.resname} {theirs.coordinator.atom_name}; list the "
                "bonds in the reference's order"
            )
        key = (own.coordinator.chain, own.coordinator.resid, own.coordinator.icode)
        source = (theirs.coordinator.chain, theirs.coordinator.resid, theirs.coordinator.icode)
        if pairing.setdefault(key, source) != source:
            raise ValueError(
                f"{label} but its bond {number} pairs residue {own.coordinator.resname}"
                f"{own.coordinator.resid} with two different residues of the reference"
            )
    if len(set(pairing.values())) != len(pairing):
        raise ValueError(
            f"{label} but pairs two of its residues with one residue of the reference"
        )
    for own, theirs in zip(site.mcpb.additional_residues, reference.mcpb.additional_residues):
        if _residue_family(own.resname) != _residue_family(theirs.resname):
            raise ValueError(
                f"{label} but its additional residue {own.resname}{own.resid} is not the "
                f"counterpart of {theirs.resname}{theirs.resid}"
            )


def _bonded_site_of(
    sites: list["MetalSiteConfig"], ligand: "LigandConfig"
) -> "MetalSiteConfig | None":
    key = (ligand.selector.chain, ligand.selector.resid, ligand.selector.icode)
    for site in sites:
        if site.model != "bonded_mcpb" or site.mcpb is None:
            continue
        residues = {
            (bond.coordinator.chain, bond.coordinator.resid, bond.coordinator.icode)
            for bond in site.mcpb.bonds
        } | {
            (item.chain, item.resid, item.icode) for item in site.mcpb.additional_residues
        }
        if key in residues:
            return site
    return None

