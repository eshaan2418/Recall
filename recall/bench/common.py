"""Result provenance + statistics helpers shared by every experiment runner.

Every result file written through :func:`write_result` carries the git commit,
hardware, library versions, seed, config, timestamp and evidence stage, so any
number quoted in docs can be traced back (see ``bench/check_numbers.py``).
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from recall.paths import REPO, RESULTS

STAGES = {"desktop-proxy", "emulator", "on-device", "estimated", "offline-eval"}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def hardware() -> dict[str, str]:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        info["cpu"] = platform.processor()
    return info


def versions() -> dict[str, str]:
    out = {}
    for mod in ("numpy", "torch", "transformers", "open_clip", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def result_header(stage: str, seed: int | None, config: dict[str, Any]) -> dict[str, Any]:
    assert stage in STAGES, stage
    return {
        "evidence_stage": stage,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "hardware": hardware(),
        "versions": versions(),
        "seed": seed,
        "config": config,
        "timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_result(rel: str, payload: dict[str, Any]) -> Path:
    assert "evidence_stage" in payload, "results must carry provenance (use result_header)"
    p = RESULTS / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, default=_default))
    return p


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n: int = 2000,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Point estimate + 95% percentile bootstrap CI. If ``groups`` is given, resample whole groups
    (e.g. videos) — a cluster bootstrap that respects within-video correlation."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(stat(values))
    if groups is None:
        idx = rng.integers(0, len(values), size=(n, len(values)))
        boots = np.array([stat(values[i]) for i in idx])
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        members = [np.flatnonzero(groups == g) for g in ug]
        boots = np.empty(n)
        for b in range(n):
            pick = rng.integers(0, len(ug), size=len(ug))
            boots[b] = stat(values[np.concatenate([members[i] for i in pick])])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, groups: np.ndarray | None = None, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """CI of mean(a) - mean(b) for paired per-query outcomes."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return bootstrap_ci(d, np.mean, n, seed, groups)
