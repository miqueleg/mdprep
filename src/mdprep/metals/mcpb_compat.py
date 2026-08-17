"""Numerically neutral compatibility adapter for MCPB.py Seminario parsing.

Some real, symmetric Cartesian Hessians yield complex-conjugate eigenpairs for
MCPB.py's non-symmetric 3x3 interatomic blocks.  Their projected force constant
can be exactly real but retain NumPy's complex scalar type, which MCPB.py cannot
round.  This adapter converts only negligible-imaginary scalar results to real
numbers and fails on any material imaginary component.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class McpbSeminarioCompatibilityError(ValueError):
    """Raised when a projected MCPB value is materially complex."""


def coerce_negligible_imaginary(
    value: Any,
    *,
    label: str,
    relative_tolerance: float = 1.0e-10,
) -> Any:
    """Return the real part only when the imaginary residual is negligible."""

    if not np.iscomplexobj(value):
        return value
    scalar = complex(value)
    limit = relative_tolerance * max(1.0, abs(scalar.real))
    if abs(scalar.imag) > limit:
        raise McpbSeminarioCompatibilityError(
            f"{label} is materially complex: {scalar!r}; allowed imaginary "
            f"magnitude is {limit:.3e}. The Hessian cannot be passed safely to MCPB.py."
        )
    return float(scalar.real)


def _coerce_result(value: Any, *, label: str, counter: list[int]) -> Any:
    if isinstance(value, tuple):
        return tuple(
            _coerce_result(item, label=f"{label}[{index}]", counter=counter)
            for index, item in enumerate(value)
        )
    converted = coerce_negligible_imaginary(value, label=label)
    if converted is not value:
        counter[0] += 1
    return converted


def run_mcpb_with_compatibility(
    mcpb_script: str | Path,
    mcpb_args: Sequence[str],
) -> None:
    """Run MCPB.py after installing the strict scalar-type compatibility shim."""

    script = Path(mcpb_script).resolve()
    if not script.is_file() or script.stat().st_size == 0:
        raise McpbSeminarioCompatibilityError(
            f"MCPB.py script is missing or empty: {script}"
        )
    try:
        from pymsmt.mcpb import gene_final_frcmod_file as seminario
    except Exception as exc:  # pragma: no cover - depends on external AmberTools
        raise McpbSeminarioCompatibilityError(
            "MCPB.py compatibility adapter could not import pymsmt from the active "
            "Python environment. Run mdprep from the AmberTools environment."
        ) from exc

    counter = [0]
    originals = {
        "get_bond_fc_with_sem": seminario.get_bond_fc_with_sem,
        "get_ang_fc": seminario.get_ang_fc,
        "get_mod_ang_fc": seminario.get_mod_ang_fc,
    }

    def wrap(name: str) -> Callable[..., Any]:
        original = originals[name]

        def compatible(*args: Any, **kwargs: Any) -> Any:
            return _coerce_result(
                original(*args, **kwargs),
                label=name,
                counter=counter,
            )

        return compatible

    for name in originals:
        setattr(seminario, name, wrap(name))
    previous_argv = sys.argv
    sys.argv = [str(script), *mcpb_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        for name, original in originals.items():
            setattr(seminario, name, original)
        print(
            "[mdprep] MCPB Seminario compatibility adapter converted "
            f"{counter[0]} negligible-imaginary scalar value(s) to real values."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcpb-script", required=True)
    parser.add_argument("mcpb_args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    mcpb_args = list(options.mcpb_args)
    if mcpb_args[:1] == ["--"]:
        mcpb_args = mcpb_args[1:]
    run_mcpb_with_compatibility(options.mcpb_script, mcpb_args)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through MCPB external tests
    raise SystemExit(main())
