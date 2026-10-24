#!/bin/bash
# Download the EPIC subset: one process per split, in parallel (resumable).
cd "$(dirname "$0")/.."
for s in train val test; do
  PYTHONUNBUFFERED=1 .venv/bin/python -m recall.data.loaders.epic download --split $s --workers 4 > data/cache/logs/download_$s.log 2>&1 &
done
wait
