#!/bin/bash
# Privacy + secret scan over the full git history AND the working tree (tracked + untracked-not-ignored files).
# Fails (exit 1) on:
#   1. credentials: gitleaks over history (all refs) and the working tree incl. untracked files (scripts/gitleaks.toml)
#   2. any image / video / audio / array / model-weight / archive / database blob, detected by extension AND by
#      magic bytes, in any commit on any ref or in untracked files, unless its path is on the explicit allow-list in
#      recall/privacy/scan_history.py (synthetic test fixtures under tests/fixtures/synthetic/, plots, launcher icons)
#   3. blobs > 2 MB; personal-data paths (data/personal/, data/raw/, data/cache/, study/data/, .env*)
#   4. a face (BlazeFace) in any allow-listed image. Needs .venv + models/litert/face_detector.tflite; without them
#      this step is skipped with a warning, or fails if SCAN_STRICT=1.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=scripts/gitleaks.toml
command -v gitleaks >/dev/null || { echo "gitleaks not installed (brew install gitleaks)"; exit 1; }

echo "== gitleaks (history, all refs) =="
gitleaks git . --config "$CFG" --log-opts="--all" --redact --no-banner

echo "== gitleaks (working tree: tracked + untracked-not-ignored files) =="
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git ls-files -z --cached --others --exclude-standard | xargs -0 tar cf - 2>/dev/null | tar xf - -C "$tmp"
gitleaks dir "$tmp" --config "$CFG" --redact --no-banner

echo "== media / binaries / large blobs / personal paths (history + untracked) =="
PY=.venv/bin/python
[ -x "$PY" ] || PY=python3
if [ -x .venv/bin/python ] && [ -f models/litert/face_detector.tflite ] && .venv/bin/python -c "import ai_edge_litert" 2>/dev/null; then
  PYTHONPATH=. "$PY" -m recall.privacy.scan_history --faces
else
  [ "${SCAN_STRICT:-0}" = 1 ] && { echo "SCAN_STRICT=1 but face check unavailable (.venv / ai_edge_litert / face model)"; exit 1; }
  echo "WARN: face check on allow-listed images skipped (run: python -m recall.privacy.assets face_detector)"
  PYTHONPATH=. "$PY" -m recall.privacy.scan_history
fi
echo "scan OK"
