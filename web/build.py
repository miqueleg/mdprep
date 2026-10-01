#!/usr/bin/env python3
"""Stage the mdprep wheel the input builder loads under Pyodide.

The page installs the real package rather than reimplementing the decision
logic in JavaScript, so the wheel has to sit next to the static files. It is a
build artifact and is not committed; CI runs this before publishing the site,
and you run it once before serving the page locally.

    python web/build.py
    python -m http.server --directory web 8000
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


WEB = Path(__file__).resolve().parent
REPO = WEB.parent
# micropip parses the wheel filename, so it has to keep its PEP 427 form
# (name-version-pytag-abi-platform.whl). The page therefore cannot hard-code a
# name; build.py writes this little index next to the wheel instead.
INDEX_NAME = "wheel.json"


def build_wheel(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", staging, str(REPO)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        built = sorted(Path(staging).glob("mdprep-*.whl"))
        if not built:
            raise SystemExit("pip produced no mdprep wheel")
        target = destination / built[0].name
        shutil.copyfile(built[0], target)
        for stale in destination.glob("mdprep-*.whl"):
            if stale != target:
                stale.unlink()
        (destination / INDEX_NAME).write_text(
            json.dumps({"wheel": target.name, "version": version_of(target.name)}) + "\n",
            encoding="utf-8",
        )
        return target


def version_of(wheel_name: str) -> str:
    return wheel_name.split("-")[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=WEB / "vendor",
        help="Directory to write the wheel into (default: web/vendor).",
    )
    args = parser.parse_args()

    wheel = build_wheel(args.out)
    size_kb = wheel.stat().st_size / 1024
    print(f"wrote {wheel.relative_to(REPO)} ({size_kb:.0f} KB)")
    print("serve with: python -m http.server --directory web 8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
