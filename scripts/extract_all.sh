#!/bin/bash
# Extract features for every subset video as soon as its download completes (loops until all done).
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
while true; do
  .venv/bin/python -m recall.perception.extract --order ${ORDER:-fwd} 2>&1 | grep --line-buffered -v -i warn
  n=$(ls data/cache/feats/MobileCLIP2-S2/*.npz 2>/dev/null | wc -l)
  echo "$(date) extracted $n/59"
  [ "$n" -ge 59 ] && break
  sleep 60
done
