#!/usr/bin/env python3
"""Compatibility entry point for mdprep's Roe--Brooks OpenMM workflow.

Installed users should normally run ``mdprep run-md MANIFEST``. This wrapper
keeps the protocol directly executable from a source checkout without carrying
a second, divergent implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mdprep.config.loader import load_manifest
from mdprep.md.roe_brooks import run_roe_brooks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--prmtop", type=Path)
    parser.add_argument("--inpcrd", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, resolve_paths=True)
    project_output = Path(manifest.project.output_dir)
    result = run_roe_brooks(
        prmtop_path=args.prmtop or project_output / "final" / "system.prmtop",
        inpcrd_path=args.inpcrd or project_output / "final" / "system.inpcrd",
        output_dir=(
            args.output_dir
            or project_output / "md" / "roe_brooks_2020"
        ),
        config=manifest.molecular_dynamics,
    )
    print(f"MD completed on {result.platform}: {result.final_pdb}")


if __name__ == "__main__":
    main()
