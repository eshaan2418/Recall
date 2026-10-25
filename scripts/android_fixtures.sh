#!/usr/bin/env bash
# Regenerate the Kotlin parity-test fixtures from the Python reference implementation.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PYTHONPATH=$PWD "$PY" android/tools/make_parity_fixtures.py
