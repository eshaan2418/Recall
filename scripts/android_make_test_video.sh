#!/usr/bin/env bash
# Regenerate the synthetic e2e clip (android/app/src/androidTest/assets/e2e/keys_on_table.webm). Needs ffmpeg + Pillow.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
"$PY" android/tools/make_test_video.py
