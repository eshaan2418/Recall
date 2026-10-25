"""Zero-unbacked-numbers check (Definition of Done #4).

Extracts every number from the public-facing docs (README.md, RESUME_BULLETS.md, FINAL_REVIEW.md, DEMO.md,
paper/*.tex) and checks each against the set of values present in ``results/**/*.json``:

* decimals / percentages must equal some result value at the printed precision (percentages also match
  fractions ×100);
* integers ≥ 10 must equal some result value (or its ×100 / rounding), unless whitelisted;
* spec targets and stated assumptions (5 %/h, 1 GB/week, 2 s, 2 fps, 16 h/day, years, section numbers …) are
  listed with a reason in ``configs/number_whitelist.txt``.

Lines inside ``<!-- nocheck -->`` … ``<!-- /nocheck -->`` (or LaTeX ``% nocheck`` … ``% /nocheck``) are skipped —
used only for the bibliography and verbatim command lines.

Exit code 1 if any number is unbacked; prints file:line for each.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

from recall.paths import REPO, RESULTS

DOCS = ["README.md", "RESUME_BULLETS.md", "FINAL_REVIEW.md", "DEMO.md"]
NUM = re.compile(r"(?<![\w.\-/])(\d+(?:,\d{3})*(?:\.\d+)?)(\s?%|×|x\b)?(?![\w/])")
SKIP_CTX = re.compile(
    r"(results/|configs/|\.py|\.json|\.md|\.sh|\.kt|https?://|`[^`]*$|P\d\d_|M\d\b|Fig\.|Table|§|\\ref|\\cite|\\label)"
)


def result_values() -> list[float]:
    vals: list[float] = []

    def walk(x: object) -> None:
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)) and math.isfinite(x):
            vals.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for p in RESULTS.rglob("*.json"):
        try:
            walk(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    return vals


def whitelist() -> set[str]:
    p = REPO / "configs" / "number_whitelist.txt"
    if not p.exists():
        return set()
    return {ln.split("#")[0].strip() for ln in p.read_text().splitlines() if ln.split("#")[0].strip()}


def backed(tok: str, vals: list[float], cache: dict) -> bool:
    """True if some result value v, or v×100 (fractions printed as %), v/1e3, v/1e6, v/1e9 (kB/MB/GB),
    rounds to the printed number at its printed precision."""
    if tok in cache:
        return cache[tok]
    s = tok.replace(",", "")
    x = float(s)
    dec = len(s.split(".")[1]) if "." in s else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    ok = any(abs(t - x) <= tol for v in vals for t in (v, v * 100, 1 - v, v / 1e3, v / 1e6, v / 1e9))
    cache[tok] = ok
    return ok


def scan_file(path: Path, vals: list[float], wl: set[str], cache: dict) -> list[str]:
    bad = []
    skipping = fenced = False
    for i, raw in enumerate(path.read_text().splitlines(), 1):
        if "nocheck" in raw:
            skipping = "/nocheck" not in raw
            continue
        if raw.strip().startswith("```"):  # code blocks hold commands / illustrative examples, not claims
            fenced = not fenced
            continue
        if skipping or fenced or raw.strip().startswith(("$ ", "make ", "python ")):
            continue
        line = re.sub(r"<[^>]*>", " ", raw)  # HTML tags (image widths, alt text) are layout, not claims
        for m in NUM.finditer(line):
            tok, unit = m.group(1), (m.group(2) or "").strip()
            pre = line[max(0, m.start() - 25) : m.start()]
            if SKIP_CTX.search(pre[-12:]) or tok in wl or f"{tok}{unit}" in wl:
                continue
            s = tok.replace(",", "")
            is_int = "." not in s
            if is_int and not unit and float(s) < 10:
                continue
            if is_int and 1900 <= float(s) <= 2100:
                continue
            if not backed(tok, vals, cache):
                bad.append(f"{path.relative_to(REPO)}:{i}: {tok}{unit}  …{line.strip()[:110]}")
    return bad


def main() -> int:
    vals = result_values()
    wl = whitelist()
    files = [REPO / d for d in DOCS if (REPO / d).exists()] + sorted((REPO / "paper").glob("*.tex"))
    cache: dict = {}
    bad = [b for f in files for b in scan_file(f, vals, wl, cache)]
    for b in bad:
        print("UNBACKED", b)
    print(f"checked {len(files)} files against {len(vals)} result values: {len(bad)} unbacked")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
