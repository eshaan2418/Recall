"""Confidence calibration + abstention (Contribution 3).

The calibrator maps the answer's evidence features (detector score, margin, location confidence,
number of supporting observations, age of the memory, conflicting sightings, …) to P(answer correct).
Fit on **val** records only; evaluated on test. Metrics: ECE (15 equal-mass bins), Brier, AUROC,
reliability diagram, risk–coverage curve, selective accuracy at fixed coverage.

Abstained-by-retrieval answers (no candidate) get confidence 0 and count as "I don't know".
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from recall.qa.answer import FEATURES

KINDS = ("where", "temporal", "state")


def feat_matrix(recs: list[dict]) -> np.ndarray:
    rows = []
    for r in recs:
        f = r.get("feats") or {}
        v = [float(f.get(k, 0.0)) for k in FEATURES]
        v += [float(f.get("fallback", 0.0)), float(f.get("p_on", 0.5)), abs(float(f.get("p_on", 0.5)) - 0.5)]
        v += [float(r["kind"] == k) for k in KINDS]
        rows.append(v)
    return np.array(rows, np.float32).reshape(len(recs), len(FEATURES) + 3 + len(KINDS))


def labels(recs: list[dict]) -> np.ndarray:
    """1 if the answer given is correct. For absent-object queries an *answer* is always wrong."""
    y = []
    for r in recs:
        if r["kind"] == "absent":
            y.append(0.0)
        else:
            y.append(float(bool(r.get("correct"))))
    return np.array(y)


@dataclass
class Calibrator:
    method: str  # logistic | isotonic | raw
    lr: LogisticRegression | None = None
    iso: IsotonicRegression | None = None

    def predict(self, recs: list[dict]) -> np.ndarray:
        answered = np.array([bool(r.get("answered")) for r in recs])
        if self.method == "raw":
            p = np.array([float((r.get("feats") or {}).get("score", 0.0)) for r in recs])
        else:
            X = feat_matrix(recs)
            assert self.lr is not None
            p = self.lr.predict_proba(X)[:, 1]
            if self.method == "isotonic":
                assert self.iso is not None
                p = self.iso.predict(p)
        return np.where(answered, p, 0.0)


def fit(recs_val: list[dict], method: str = "logistic", C: float = 1.0) -> Calibrator:
    if method == "raw":
        return Calibrator("raw")
    ans = [r for r in recs_val if r.get("answered")]
    X, y = feat_matrix(ans), labels(ans)
    lr = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    iso = None
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(lr.predict_proba(X)[:, 1], y)
    return Calibrator(method, lr, iso)


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in itertools.pairwise(edges):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            out.append(
                {
                    "lo": float(lo),
                    "hi": float(hi),
                    "n": int(m.sum()),
                    "conf": float(p[m].mean()),
                    "acc": float(y[m].mean()),
                }
            )
    return out


def risk_coverage(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the top-c fraction by confidence; selective accuracy at each coverage c."""
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    cov = np.arange(1, len(p) + 1) / len(p)
    acc = np.cumsum(ys) / np.arange(1, len(p) + 1)
    return cov, acc


def aurc(p: np.ndarray, y: np.ndarray) -> float:
    cov, acc = risk_coverage(p, y)
    return float(np.mean(1 - acc))


def summary(p: np.ndarray, y: np.ndarray, coverages=(0.25, 0.5, 0.75, 1.0)) -> dict:
    cov, acc = risk_coverage(p, y)
    sel = {f"sel_acc@{c}": float(acc[max(0, int(np.ceil(c * len(p))) - 1)]) for c in coverages}
    try:
        auroc = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    except ValueError:
        auroc = float("nan")
    return {
        "ece": ece(p, y),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": auroc,
        "aurc": aurc(p, y),
        **sel,
        "n": len(p),
        "base_acc": float(y.mean()),
    }


def threshold_for_precision(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Smallest confidence threshold whose answered set has accuracy ≥ target (fit on val)."""
    order = np.argsort(-p)
    acc = np.cumsum(y[order]) / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(acc >= target)
    if len(ok) == 0:
        return 1.01
    return float(p[order][ok[-1]])
