#!/bin/bash
# Run the zero-unbacked-numbers check against exactly what is committed (HEAD), as CI sees it.
# Catches docs that cite result files which exist locally but were never committed.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git archive HEAD | tar -x -C "$tmp"
PYTHONPATH="$tmp" .venv/bin/python -m recall.bench.check_numbers
