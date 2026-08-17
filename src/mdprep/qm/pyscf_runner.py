"""Lazy PySCF runner for ligand RESP/QMMESP charge derivation."""

from __future__ import annotations

import contextlib
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PySCFRunnerError(RuntimeError):
    """Raised when PySCF cannot run the requested QM calculation."""


@dataclass(frozen=True)
class PySCFResult:
    mol: object
    mf: object
    method: str
    basis: str
    dft_grid_level: int | None
    charge: int
    spin: int
    multiplicity: int
    electron_count: int
    energy_hartree: float
    converged: bool
    cycles: int | None
    scf_algorithm: str
    initial_guess: str
    initial_density_checkpoint_path: str | None
    initial_density_checkpoint_sha256: str | None
    initial_density_checkpoint_max_displacement_angstrom: float | None
    initial_density_checkpoint_rms_displacement_angstrom: float | None
    initial_density_checkpoint_projected: bool
    adiis_precondition_cycles: int | None
    adiis_precondition_dft_grid_level: int | None
    precondition_cycles_completed: int
    level_shift_mode: str
    level_shift_hartree: float
    damping_factor: float
    diis_space: int
    density_fitting: bool
    auxiliary_basis: str | None
    spin_square: float | None
    effective_multiplicity: float | None
    electrostatic_embedding_applied: bool
    mm_point_charge_count: int
    embedding_operator: str | None
    stdout: str
    stderr: str
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "basis": self.basis,
            "dft_grid_level": self.dft_grid_level,
            "charge": self.charge,
            "spin": self.spin,
            "multiplicity": self.multiplicity,
            "electron_count": self.electron_count,
            "energy_hartree": self.energy_hartree,
            "converged": self.converged,
            "cycles": self.cycles,
            "scf_algorithm": self.scf_algorithm,
            "initial_guess": self.initial_guess,
            "initial_density_checkpoint_path": self.initial_density_checkpoint_path,
            "initial_density_checkpoint_sha256": (
                self.initial_density_checkpoint_sha256
            ),
            "initial_density_checkpoint_max_displacement_angstrom": (
                self.initial_density_checkpoint_max_displacement_angstrom
            ),
            "initial_density_checkpoint_rms_displacement_angstrom": (
                self.initial_density_checkpoint_rms_displacement_angstrom
            ),
            "initial_density_checkpoint_projected": (
                self.initial_density_checkpoint_projected
            ),
            "adiis_precondition_cycles": self.adiis_precondition_cycles,
            "adiis_precondition_dft_grid_level": (
                self.adiis_precondition_dft_grid_level
            ),
            "precondition_cycles_completed": self.precondition_cycles_completed,
            "level_shift_mode": self.level_shift_mode,
            "level_shift_hartree": self.level_shift_hartree,
            "damping_factor": self.damping_factor,
            "diis_space": self.diis_space,
            "density_fitting": self.density_fitting,
            "auxiliary_basis": self.auxiliary_basis,
            "spin_square": self.spin_square,
            "effective_multiplicity": self.effective_multiplicity,
            "electrostatic_embedding_applied": self.electrostatic_embedding_applied,
            "mm_point_charge_count": self.mm_point_charge_count,
            "embedding_operator": self.embedding_operator,
            "warnings": self.warnings,
        }


def pyscf_available() -> bool:
    try:
        import pyscf  # noqa: F401
    except Exception:
        return False
    return True


def pyscf_version() -> str:
    try:
        import pyscf
    except Exception:
        return "not available"
    return getattr(pyscf, "__version__", "unknown")


def scf_class_name(method: str, spin: int) -> str:
    if method.upper() == "HF":
        return "UHF" if spin > 0 else "RHF"
    return "UKS" if spin > 0 else "RKS"


def run_pyscf_scf(
    *,
    elements: list[str],
    coordinates: np.ndarray,
    charge: int,
    spin: int,
    method: str,
    basis: str,
    max_cycle: int,
    conv_tol: float,
    mm_charges: np.ndarray | None = None,
    mm_coordinates: np.ndarray | None = None,
    work_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    initial_density_checkpoint_path: str | Path | None = None,
    expected_initial_density_checkpoint_sha256: str | None = None,
    initial_density_checkpoint_max_displacement_angstrom: float = 0.0,
    num_threads: int = 1,
    max_memory_mb: int = 4000,
    dft_grid_level: int = 3,
    scf_algorithm: str = "diis",
    initial_guess: str = "minao",
    adiis_precondition_cycles: int = 15,
    adiis_precondition_dft_grid_level: int | None = None,
    level_shift_mode: str = "static",
    level_shift_hartree: float = 0.0,
    damping_factor: float = 0.0,
    diis_space: int = 8,
    density_fitting: bool = False,
    auxiliary_basis: str | None = None,
) -> PySCFResult:
    try:
        from pyscf import dft, gto, lib, qmmm, scf
    except Exception as exc:
        raise PySCFRunnerError(
            "PySCF is required for gas_resp_pyscf, qmmesp_pyscf, and "
            "mcpb_resp_pyscf workflows."
        ) from exc

    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(elements), 3):
        raise PySCFRunnerError("QM coordinate array shape does not match element list.")
    if not elements or not np.all(np.isfinite(coords)):
        raise PySCFRunnerError("QM elements and finite coordinates are required.")
    if spin < 0:
        raise PySCFRunnerError("PySCF spin must be non-negative.")
    if max_cycle < 1 or not np.isfinite(conv_tol) or conv_tol <= 0:
        raise PySCFRunnerError("PySCF max_cycle and conv_tol must be positive.")
    if num_threads < 1:
        raise PySCFRunnerError("PySCF num_threads must be at least 1.")
    if max_memory_mb < 1:
        raise PySCFRunnerError("PySCF max_memory_mb must be positive.")
    if dft_grid_level < 0 or dft_grid_level > 9:
        raise PySCFRunnerError("PySCF dft_grid_level must be between 0 and 9.")
    if scf_algorithm not in {
        "diis",
        "newton",
        "adiis_then_diis",
        "adiis_then_newton",
    }:
        raise PySCFRunnerError(
            "PySCF scf_algorithm must be 'diis', 'newton', "
            "'adiis_then_diis', or 'adiis_then_newton'."
        )
    if initial_guess not in {"minao", "atom", "huckel"}:
        raise PySCFRunnerError(
            "PySCF initial_guess must be 'minao', 'atom', or 'huckel'."
        )
    if level_shift_mode not in {"static", "dynamic"}:
        raise PySCFRunnerError(
            "PySCF level_shift_mode must be 'static' or 'dynamic'."
        )
    if scf_algorithm != "diis" and level_shift_mode == "dynamic":
        raise PySCFRunnerError(
            "PySCF dynamic level shifting is available only with DIIS."
        )
    if adiis_precondition_cycles < 1:
        raise PySCFRunnerError(
            "PySCF adiis_precondition_cycles must be at least 1."
        )
    if adiis_precondition_dft_grid_level is not None:
        if scf_algorithm not in {"adiis_then_diis", "adiis_then_newton"}:
            raise PySCFRunnerError(
                "PySCF adiis_precondition_dft_grid_level requires "
                "scf_algorithm='adiis_then_diis' or 'adiis_then_newton'."
            )
        if not 0 <= adiis_precondition_dft_grid_level <= 9:
            raise PySCFRunnerError(
                "PySCF adiis_precondition_dft_grid_level must be between 0 and 9."
            )
    if not np.isfinite(level_shift_hartree) or level_shift_hartree < 0:
        raise PySCFRunnerError("PySCF level_shift_hartree must be non-negative.")
    if not np.isfinite(damping_factor) or not 0 <= damping_factor < 1:
        raise PySCFRunnerError("PySCF damping_factor must be in [0, 1).")
    if diis_space < 2:
        raise PySCFRunnerError("PySCF diis_space must be at least 2.")
    if auxiliary_basis is not None and not density_fitting:
        raise PySCFRunnerError(
            "PySCF auxiliary_basis requires density_fitting=True."
        )
    restart_values = (
        initial_density_checkpoint_path,
        expected_initial_density_checkpoint_sha256,
    )
    if any(value is not None for value in restart_values) and not all(
        value is not None for value in restart_values
    ):
        raise PySCFRunnerError(
            "PySCF initial density checkpoint path and expected SHA-256 must "
            "be provided together."
        )
    if initial_density_checkpoint_path is not None and scf_algorithm != "newton":
        raise PySCFRunnerError(
            "PySCF initial density checkpoints require scf_algorithm='newton'."
        )
    if (
        not np.isfinite(initial_density_checkpoint_max_displacement_angstrom)
        or initial_density_checkpoint_max_displacement_angstrom < 0
    ):
        raise PySCFRunnerError(
            "PySCF initial density checkpoint maximum displacement must be "
            "finite and non-negative."
        )
    mm_charge_values = None if mm_charges is None else np.asarray(mm_charges, dtype=float)
    mm_coordinate_values = None if mm_coordinates is None else np.asarray(mm_coordinates, dtype=float)
    if mm_charge_values is None and mm_coordinate_values is not None:
        raise PySCFRunnerError("MM charges are required when MM coordinates are provided.")
    if mm_charge_values is not None:
        if mm_coordinate_values is None:
            raise PySCFRunnerError("MM coordinates are required when MM charges are provided.")
        if mm_charge_values.ndim != 1 or mm_coordinate_values.shape != (len(mm_charge_values), 3):
            raise PySCFRunnerError("MM point-charge arrays must have shapes (n,) and (n, 3).")
        if not np.all(np.isfinite(mm_charge_values)) or not np.all(np.isfinite(mm_coordinate_values)):
            raise PySCFRunnerError("MM point-charge arrays contain non-finite values.")
    embedding_applied = mm_charge_values is not None and len(mm_charge_values) > 0
    atom_spec = [(element, tuple(coord)) for element, coord in zip(elements, coords, strict=True)]
    stdout = io.StringIO()
    stderr = io.StringIO()
    mol = gto.Mole()
    mol.atom = atom_spec
    mol.unit = "Angstrom"
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.basis = basis
    mol.max_memory = int(max_memory_mb)
    mol.verbose = 4
    mol.stdout = stdout
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
    restart_checkpoint = (
        Path(initial_density_checkpoint_path).resolve()
        if initial_density_checkpoint_path is not None
        else None
    )
    restart_checkpoint_sha256: str | None = None
    restart_max_displacement: float | None = None
    restart_rms_displacement: float | None = None
    restart_projected = False
    restart_mol = None
    if restart_checkpoint is not None:
        if not restart_checkpoint.is_file():
            raise PySCFRunnerError(
                f"PySCF initial density checkpoint does not exist: {restart_checkpoint}"
            )
        with restart_checkpoint.open("rb") as handle:
            restart_checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if restart_checkpoint_sha256.lower() != str(
            expected_initial_density_checkpoint_sha256
        ).lower():
            raise PySCFRunnerError(
                "PySCF initial density checkpoint SHA-256 mismatch: expected "
                f"{expected_initial_density_checkpoint_sha256}, found "
                f"{restart_checkpoint_sha256}."
            )
        try:
            restart_mol = lib.chkfile.load_mol(str(restart_checkpoint))
        except Exception as exc:
            raise PySCFRunnerError(
                f"PySCF initial density checkpoint molecule could not be read: {exc}"
            ) from exc
    previous_threads = int(lib.num_threads())
    lib.num_threads(int(num_threads))
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                mol.build()
            except Exception as exc:
                raise PySCFRunnerError(f"PySCF molecule build failed: {exc}") from exc
        if restart_mol is not None:
            identity_matches = (
                int(restart_mol.natm) == int(mol.natm)
                and int(restart_mol.charge) == int(mol.charge)
                and int(restart_mol.spin) == int(mol.spin)
                and int(restart_mol.nao_nr()) == int(mol.nao_nr())
                and np.array_equal(restart_mol.atom_charges(), mol.atom_charges())
            )
            if not identity_matches:
                raise PySCFRunnerError(
                    "PySCF initial density checkpoint molecule does not exactly match "
                    "the current atom order/elements, charge, spin, and basis dimensions."
                )
            displacement = np.linalg.norm(
                restart_mol.atom_coords(unit="Angstrom")
                - mol.atom_coords(unit="Angstrom"),
                axis=1,
            )
            restart_max_displacement = float(np.max(displacement, initial=0.0))
            restart_rms_displacement = float(
                np.sqrt(np.mean(np.square(displacement)))
            )
            if (
                restart_max_displacement
                > initial_density_checkpoint_max_displacement_angstrom + 1.0e-12
            ):
                raise PySCFRunnerError(
                    "PySCF initial density checkpoint maximum coordinate displacement "
                    f"{restart_max_displacement:.6f} A exceeds the explicit limit "
                    f"{initial_density_checkpoint_max_displacement_angstrom:.6f} A."
                )
            restart_projected = restart_max_displacement > 1.0e-10

        if method.upper() == "HF":
            mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
        else:
            try:
                mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
                mf.xc = method
                mf.grids.level = int(dft_grid_level)
            except Exception as exc:
                raise PySCFRunnerError(f"PySCF DFT setup failed for method {method!r}; use method: HF.") from exc
        if density_fitting:
            try:
                mf = mf.density_fit(auxbasis=auxiliary_basis)
            except Exception as exc:
                raise PySCFRunnerError(
                    f"PySCF density-fitting setup failed: {exc}"
                ) from exc
        mf.stdout = stdout
        mf.max_cycle = int(max_cycle)
        mf.conv_tol = float(conv_tol)
        mf.max_memory = int(max_memory_mb)
        mf.init_guess = initial_guess
        final_static_level_shift = (
            float(level_shift_hartree) if level_shift_mode == "static" else 0.0
        )
        mf.level_shift = (
            0.0
            if scf_algorithm in {"adiis_then_diis", "adiis_then_newton"}
            else final_static_level_shift
        )
        mf.damp = float(damping_factor)
        mf.diis_space = int(diis_space)
        if checkpoint is not None:
            mf.chkfile = str(checkpoint)
        if embedding_applied:
            try:
                mf = qmmm.mm_charge(
                    mf,
                    mm_coordinate_values,
                    mm_charge_values,
                    unit="Angstrom",
                )
                mf.stdout = stdout
                if checkpoint is not None:
                    mf.chkfile = str(checkpoint)
            except Exception as exc:
                raise PySCFRunnerError(f"PySCF point-charge embedding could not be applied: {exc}") from exc

        if level_shift_mode == "dynamic" and level_shift_hartree > 0:
            try:
                mf = scf.addons.dynamic_level_shift_(
                    mf,
                    factor=float(level_shift_hartree),
                )
                mf.stdout = stdout
                if checkpoint is not None:
                    mf.chkfile = str(checkpoint)
            except Exception as exc:
                raise PySCFRunnerError(
                    f"PySCF dynamic level-shift setup failed: {exc}"
                ) from exc

        if scf_algorithm == "newton":
            try:
                mf = mf.newton()
                mf.stdout = stdout
                mf.max_cycle = int(max_cycle)
                mf.conv_tol = float(conv_tol)
                mf.max_memory = int(max_memory_mb)
                if checkpoint is not None:
                    mf.chkfile = str(checkpoint)
            except Exception as exc:
                raise PySCFRunnerError(
                    f"PySCF second-order Newton SCF setup failed: {exc}"
                ) from exc

        restart_density = None
        if restart_checkpoint is not None:
            try:
                restart_density = mf.init_guess_by_chkfile(
                    str(restart_checkpoint),
                    project=restart_projected,
                )
            except Exception as exc:
                raise PySCFRunnerError(
                    f"PySCF initial density checkpoint could not be loaded: {exc}"
                ) from exc

        precondition_cycles_completed = 0
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                if scf_algorithm in {"adiis_then_diis", "adiis_then_newton"}:
                    if method.upper() != "HF" and adiis_precondition_dft_grid_level is not None:
                        mf.grids.level = int(adiis_precondition_dft_grid_level)
                    mf.DIIS = scf.ADIIS
                    mf.diis = True
                    mf.max_cycle = int(adiis_precondition_cycles)
                    mf.kernel()
                    precondition_cycles_completed = int(
                        getattr(mf, "cycles", adiis_precondition_cycles)
                    )
                    precondition_density = mf.make_rdm1()
                    mf.converged = False
                    mf.DIIS = scf.CDIIS
                    mf.diis = True
                    mf.level_shift = final_static_level_shift
                    mf.max_cycle = int(max_cycle)
                    if method.upper() != "HF" and adiis_precondition_dft_grid_level is not None:
                        mf.grids.level = int(dft_grid_level)
                        mf.grids.reset(mol)
                    if scf_algorithm == "adiis_then_newton":
                        mf = mf.newton()
                        mf.stdout = stdout
                        mf.max_cycle = int(max_cycle)
                        mf.conv_tol = float(conv_tol)
                        mf.max_memory = int(max_memory_mb)
                        if checkpoint is not None:
                            mf.chkfile = str(checkpoint)
                    energy = mf.kernel(dm0=precondition_density)
                else:
                    energy = mf.kernel(dm0=restart_density)
            except Exception as exc:
                raise PySCFRunnerError(f"PySCF SCF calculation failed: {exc}") from exc
    finally:
        lib.num_threads(previous_threads)
    converged = bool(getattr(mf, "converged", False))
    if not converged:
        raise PySCFRunnerError("PySCF SCF did not converge.")
    cycles_value = getattr(mf, "cycles", None)
    cycles = (
        precondition_cycles_completed + int(cycles_value)
        if cycles_value is not None
        else None
    )
    spin_square: float | None = None
    effective_multiplicity: float | None = None
    if spin > 0 and hasattr(mf, "spin_square"):
        try:
            spin_values = mf.spin_square()
            spin_square = float(spin_values[0])
            effective_multiplicity = float(spin_values[1])
        except Exception:
            spin_square = None
            effective_multiplicity = None
    return PySCFResult(
        mol=mol,
        mf=mf,
        method=method,
        basis=basis,
        dft_grid_level=None if method.upper() == "HF" else int(dft_grid_level),
        charge=int(charge),
        spin=int(spin),
        multiplicity=int(spin) + 1,
        electron_count=int(mol.nelectron),
        energy_hartree=float(energy),
        converged=converged,
        cycles=cycles,
        scf_algorithm=scf_algorithm,
        initial_guess=initial_guess,
        initial_density_checkpoint_path=(
            str(restart_checkpoint) if restart_checkpoint is not None else None
        ),
        initial_density_checkpoint_sha256=restart_checkpoint_sha256,
        initial_density_checkpoint_max_displacement_angstrom=(
            restart_max_displacement
        ),
        initial_density_checkpoint_rms_displacement_angstrom=(
            restart_rms_displacement
        ),
        initial_density_checkpoint_projected=restart_projected,
        adiis_precondition_cycles=(
            int(adiis_precondition_cycles)
            if scf_algorithm in {"adiis_then_diis", "adiis_then_newton"}
            else None
        ),
        adiis_precondition_dft_grid_level=(
            int(adiis_precondition_dft_grid_level)
            if adiis_precondition_dft_grid_level is not None
            else None
        ),
        precondition_cycles_completed=precondition_cycles_completed,
        level_shift_mode=level_shift_mode,
        level_shift_hartree=float(level_shift_hartree),
        damping_factor=float(damping_factor),
        diis_space=int(diis_space),
        density_fitting=bool(density_fitting),
        auxiliary_basis=auxiliary_basis,
        spin_square=spin_square,
        effective_multiplicity=effective_multiplicity,
        electrostatic_embedding_applied=embedding_applied,
        mm_point_charge_count=0 if mm_charge_values is None else len(mm_charge_values),
        embedding_operator="pyscf.qmmm.mm_charge" if embedding_applied else None,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        warnings=[],
    )


def run_pyscf_analytic_hessian(
    mf: object,
    *,
    num_threads: int = 1,
) -> np.ndarray:
    """Evaluate one fixed-geometry analytic Hessian with bounded PySCF threads."""

    if num_threads < 1:
        raise PySCFRunnerError("PySCF num_threads must be at least 1.")
    try:
        from pyscf import lib
    except Exception as exc:
        raise PySCFRunnerError(
            "PySCF is required for the MCPB analytic Hessian."
        ) from exc
    previous_threads = int(lib.num_threads())
    lib.num_threads(int(num_threads))
    try:
        try:
            return np.asarray(mf.Hessian().kernel(), dtype=float)
        except Exception as exc:
            raise PySCFRunnerError(f"PySCF analytic Hessian calculation failed: {exc}") from exc
    finally:
        lib.num_threads(previous_threads)
