#!/bin/bash
# Clean-clone reproduction on fixtures/synthetic data (no licensed data, no GPU; target < 1 h, typically ~2 min).
# 1) lint + types + unit tests, 2) synthetic end-to-end pipeline, 3) figures from whatever results/ holds,
# 4) zero-unbacked-numbers check over the docs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONPATH=$PWD
$PY -m ruff check recall tests
$PY -m mypy
$PY -m pytest tests -m "not slow" -q
$PY -m recall.bench.quick
$PY -m recall.bench.figures --allow-missing
$PY -m recall.bench.check_numbers
echo "reproduce-quick OK"
