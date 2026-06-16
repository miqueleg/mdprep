#!/usr/bin/env bash
set -euo pipefail

run_external=false
if [[ "${1:-}" == "--external" ]]; then
  run_external=true
fi

python -m pytest -q
python -m mdprep.cli --help >/dev/null
python -m mdprep.cli --version
python -m mdprep.cli config-check examples/*.yaml
python -m mdprep.cli selftest --quick

if [[ "${run_external}" == "true" ]]; then
  python -m pytest -q -m "external or ambertools or tleap or openmm or parmed or propka or xtb or pyscf or qmmesp"
fi
