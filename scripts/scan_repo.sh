#!/bin/bash
# Secret + large-file + personal-data scan over the working tree AND full git history.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== gitleaks (history) =="; gitleaks git . --redact --no-banner
echo "== gitleaks (working tree, tracked files) =="; git ls-files -z | xargs -0 tar cf - 2>/dev/null | (mkdir -p /tmp/recall_scan_$$ && tar xf - -C /tmp/recall_scan_$$) && gitleaks dir /tmp/recall_scan_$$ --redact --no-banner; rm -rf /tmp/recall_scan_$$
echo "== large blobs in history (>2 MB) =="
big=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1=="blob" && $3>2000000 {print $3, $4}')
if [ -n "$big" ]; then echo "$big"; exit 1; fi
echo "== forbidden paths in history =="
bad=$(git log --all --name-only --pretty=format: | sort -u | grep -E '^(data/personal/|data/raw/|study/data/)|\.(mp4|MP4|tar|jpg|jpeg|png|env)$' | grep -vE '^(data/personal/README.md|results/.*/figs/.*\.png|paper/figs/.*\.png|docs/.*\.png|android/app/src/main/res/.*\.png)$' || true)
if [ -n "$bad" ]; then echo "$bad"; exit 1; fi
echo "scan OK"
