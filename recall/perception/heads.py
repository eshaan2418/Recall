"""Object / location classification heads on top of frozen embeddings.

``p(class | emb) ∝ p_zeroshot^α · p_probe^(1-α)`` — a zero-shot text-bank
classifier blended with a linear probe trained on the *train* participants
(α and the probe's C are chosen on *val*). Classes unseen in training keep
their zero-shot score, so the vocabulary stays open.

Two families: ``region`` heads (object crops / surroundings, used by Recall)
and ``frame`` heads (whole frames, used by the frame-sampling baseline B2 so it
gets the same supervision budget).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from recall.data.queries import LOCATION_KEYS, PUT_VERBS, VERB_TAKE
from recall.paths import CACHE, REPO
from recall.perception.embed import zero_shot_probs
from recall.perception.extract import REGION_KINDS


@dataclass
class BlendHead:
    text_bank: np.ndarray  # [C, D] over the full vocabulary
    scale: float = 100.0
    alpha: float = 1.0  # 1 = pure zero-shot
    probe: LogisticRegression | None = None
    probe_classes: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    def log_probs(self, emb: np.ndarray) -> np.ndarray:
        emb = np.atleast_2d(np.asarray(emb, np.float32))
        lz = np.log(zero_shot_probs(emb, self.text_bank, self.scale) + 1e-9)
        if self.probe is None or self.alpha >= 1.0:
            return lz
        lp = np.full_like(lz, np.log(1e-4))
        lp[:, self.probe_classes] = np.log(self.probe.predict_proba(emb) + 1e-9)
        z = self.alpha * lz + (1 - self.alpha) * lp
        z -= z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def probs(self, emb: np.ndarray) -> np.ndarray:
        return np.exp(self.log_probs(emb))


def _windows(narr, vid: str, verbs: set[int]):
    d = narr[(narr.video_id == vid) & (narr.verb_class.isin(verbs))]
    return d


def region_training_data(narr, feats_by_vid: dict, loc_ids: list[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Collect (embedding, label) pairs from cached features + narrations."""
    Xo, yo, Xl, yl, Xfo, yfo, Xfl, yfl = [], [], [], [], [], [], [], []
    loc_set = set(loc_ids)
    for vid, f in feats_by_vid.items():
        t = f["t"]
        rt, rk, emb = f["reg_t"], f["reg_kind"], f["reg_emb"]
        k_obj, k_ctx, k_pctx = (REGION_KINDS.index(k) for k in ("obj", "ctx", "pctx"))
        d = _windows(narr, vid, PUT_VERBS | {VERB_TAKE})
        for r in d.itertuples():
            fr = np.flatnonzero((t >= r.start_s) & (t <= r.stop_s))
            if len(fr):
                m = np.isin(rt, fr) & (rk == k_obj)
                Xo.append(emb[m])
                yo += [int(r.noun_class)] * int(m.sum())
                Xfo.append(f["frame_emb"][fr])
                yfo += [int(r.noun_class)] * len(fr)
            if r.verb_class in PUT_VERBS:
                locs = [c for c in r.all_noun_classes[1:] if c in loc_set]
                if not locs:
                    continue
                li = loc_ids.index(locs[0])
                fr2 = np.flatnonzero((t >= r.stop_s - 0.5) & (t <= r.stop_s + 1.5))
                m = (np.isin(rt, fr2) & (rk == k_pctx)) | (np.isin(rt, fr) & (rk == k_ctx))
                Xl.append(emb[m])
                yl += [li] * int(m.sum())
                Xfl.append(f["frame_emb"][fr2])
                yfl += [li] * len(fr2)

    def cat(X: list, y: list) -> tuple[np.ndarray, np.ndarray]:
        D = next((x.shape[1] for x in X if len(x)), 1)
        return (np.concatenate(X).astype(np.float32) if X else np.zeros((0, D), np.float32)), np.array(y, int)

    return {"obj": cat(Xo, yo), "loc": cat(Xl, yl), "frame_obj": cat(Xfo, yfo), "frame_loc": cat(Xfl, yfl)}


def fit_probe(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0) -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def tune_head(
    text_bank: np.ndarray,
    scale: float,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    seed: int = 0,
) -> tuple[BlendHead, dict]:
    """Grid over C and α on val accuracy."""
    zs = BlendHead(text_bank, scale)
    best: tuple[float, float, LogisticRegression | None, float | None] = (
        float((zs.probs(Xva).argmax(1) == yva).mean()) if len(yva) else 0.0,
        1.0,
        None,
        None,
    )
    grid: list[dict] = []
    log: dict = {"zero_shot_val_acc": best[0], "grid": grid}
    if len(np.unique(ytr)) < 2:
        return zs, log
    for C in (0.1, 1.0, 10.0):
        probe = fit_probe(Xtr, ytr, C, seed)
        for a in (0.0, 0.25, 0.5, 0.75):
            h = BlendHead(text_bank, scale, a, probe, probe.classes_)
            acc = float((h.probs(Xva).argmax(1) == yva).mean()) if len(yva) else 0.0
            grid.append({"C": C, "alpha": a, "val_acc": acc})
            if acc > best[0]:
                best = (acc, a, probe, C)
    acc, a_best, best_probe, C_best = best
    log.update({"best_val_acc": acc, "alpha": a_best, "C": C_best})
    if best_probe is None:
        return zs, log
    return BlendHead(text_bank, scale, a_best, best_probe, best_probe.classes_), log


@dataclass
class Heads:
    obj: BlendHead
    loc: BlendHead
    frame_obj: BlendHead
    frame_loc: BlendHead
    loc_ids: list[int]
    meta: dict

    def save(self, name: str) -> None:
        p = CACHE / "heads" / f"{name}.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pickle.dumps(self))
        (REPO / "results" / "m1").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load(name: str) -> Heads:
        # Only ever loads files this repo wrote into data/cache (never downloaded pickles).
        return pickle.loads((CACHE / "heads" / f"{name}.pkl").read_bytes())


def location_keys(loc_ids: list[int], id2key: dict[int, str]) -> list[str]:
    return [id2key[i] for i in loc_ids]


def describe(h: Heads) -> str:
    return json.dumps(h.meta, indent=1)


__all__ = ["LOCATION_KEYS", "BlendHead", "Heads", "region_training_data", "tune_head"]
