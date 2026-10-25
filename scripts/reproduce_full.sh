#!/bin/bash
# Full reproduction from raw data (EPIC-KITCHENS-100 subset, ~45 GB download; many hours on one Apple-silicon laptop).
# Every step is resumable (cached features / models are reused if present).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
PYX=${PYX:-.venv-export/bin/python}   # LiteRT export environment
export PYTHONPATH=$PWD
./scripts/download_epic.sh
$PY -m recall.perception.extract                      # 2 fps feature cache (teacher HOI + MobileCLIP2-S2)
$PY -m recall.bench.make_queries
$PY -m recall.bench.select_embedder
$PY -m recall.bench.run_m1                            # heads, baselines B2/B3/B4, Recall all-frames
$PY -m recall.baselines.vlm                           # B1 large open VLM (slow; fixed test subsample)
$PY -m recall.bench.run_m2                            # learned trigger + Pareto (3 seeds)
$PY -m recall.bench.run_m3                            # memory/storage
$PY -m recall.bench.run_m4                            # calibration, abstention, grounded generation
$PY -m recall.compress.hoi_student --seed 0           # HOI student distillation
$PYX -m recall.compress.export                        # LiteRT exports fp32/int8dyn/int8/int4w
$PYX -m recall.bench.run_m5 --parts abcde             # compression table + end-to-end + desktop latency
$PY -m recall.privacy.eval                            # M7 redaction recall
$PY -m recall.energy.model                            # M8 estimated energy (on-device: recall.energy.harness)
$PY -m recall.bench.run_m9 --standin 3                # M9 tooling check (real: without --standin)
$PY -m recall.bench.figures
$PY -m recall.bench.check_numbers
echo "reproduce-full OK"
