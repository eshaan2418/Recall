#!/bin/bash
# M2 → M3 → M4 → gate export → Android bundle → M5 (b,d,e) → M8 energy model → M9 stand-in. Logs in data/cache/logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1
L=data/cache/logs
step() { echo "$(date +%T) START $1"; shift; "$@"; echo "$(date +%T) EXIT $?"; }
step m2 .venv/bin/python -m recall.bench.run_m2 > $L/run_m2.log 2>&1
step m3 .venv/bin/python -m recall.bench.run_m3 > $L/run_m3.log 2>&1
step m4 .venv/bin/python -m recall.bench.run_m4 > $L/run_m4.log 2>&1
step gate-export .venv-export/bin/python -m recall.compress.export --models gate > $L/export_gate.log 2>&1
step bundle .venv/bin/python -m recall.compress.export_android > $L/bundle.log 2>&1
step m5bde .venv-export/bin/python -m recall.bench.run_m5 --parts bde > $L/run_m5_bde.log 2>&1
step energy .venv/bin/python -m recall.energy.model > $L/energy.log 2>&1
step m9 .venv/bin/python -m recall.bench.run_m9 --standin 3 > $L/run_m9.log 2>&1
echo ALLDONE
