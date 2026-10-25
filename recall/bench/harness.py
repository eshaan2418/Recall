"""Offline evaluation harness: policy → events → memory view → answers → per-query records.

A *system* = (trigger policy, memory kind, answer config, heads). The harness
is shared by M1 (baselines), M2 (trigger Pareto), M3 (memory variants) and M4
(calibration), so every comparison uses identical queries, scoring and CIs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from recall.bench.common import bootstrap_ci
from recall.data.queries import Query, load_queries
from recall.memory.schema import Event
from recall.paths import CACHE, REPO
from recall.perception.events import ExtractorConfig, RegionIndex, extract_events
from recall.perception.extract import load_feats
from recall.perception.heads import Heads
from recall.qa.answer import Answer, AnswerConfig, MemoryView, answer
from recall.qa.parse import ParsedQuery

PolicyFn = Callable[[str, dict], np.ndarray]

WHERE_PRE_S, WHERE_POST_S = 5.0, 10.0  # temporal tolerance around the GT put window
TEMPORAL_TOL_S = 10.0


def subset(split: str | None = None) -> list[dict]:
    s = json.loads((REPO / "configs" / "epic_subset.json").read_text())
    return [v for v in s if split is None or v["split"] == split]


def available(split: str, model: str) -> list[str]:
    return [v["video_id"] for v in subset(split) if (CACHE / "feats" / model / f"{v['video_id']}.npz").exists()]


def queries_for(split: str) -> list[Query]:
    return load_queries(REPO / "configs" / "queries" / f"{split}.jsonl")


@dataclass
class System:
    name: str
    policy: PolicyFn
    memory: str = "events"  # events | frames
    cfg: AnswerConfig = field(default_factory=AnswerConfig)
    cfg_by_kind: dict[str, AnswerConfig] = field(default_factory=dict)  # overrides per query family
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    state_head: Callable | None = None
    frame_stride: int = 1  # for memory=frames: keep 1 of every `frame_stride` processed frames


def frame_events(feats: dict, mask: np.ndarray, vid: str) -> list[Event]:
    out = []
    for i in np.flatnonzero(mask):
        e = feats["frame_emb"][i].astype(np.float32)
        e = e / (np.linalg.norm(e) + 1e-8)
        out.append(
            Event(
                vid, float(feats["t"][i]), "frame", -1, int(i), e, e, e, n_obs=1, hand_score=1.0, release_observed=True
            )
        )
    return out


def build_view(
    sys: System, vid: str, feats: dict, heads: Heads, regions: RegionIndex | None = None
) -> tuple[MemoryView, dict]:
    mask = sys.policy(vid, feats)
    if sys.memory == "frames":
        if sys.frame_stride > 1:
            idx = np.flatnonzero(mask)[:: sys.frame_stride]
            mask = np.zeros_like(mask)
            mask[idx] = True
        evs = frame_events(feats, mask, vid)
        view = MemoryView.build(evs, heads.frame_obj, heads.frame_loc)
    else:
        evs = extract_events(feats, mask, vid, sys.extractor, regions)
        view = MemoryView.build(evs, heads.obj, heads.loc)
    dur_h = len(feats["t"]) / 2.0 / 3600
    cost = {
        "heavy_frames": int(mask.sum()),
        "frames": len(mask),
        "hours": dur_h,
        "n_events": len(evs),
        "heavy_per_hour": mask.sum() / dur_h,
    }
    return view, cost


def score(q: Query, a: Answer, loc_ids: list[int]) -> dict:
    r: dict = {
        "qid": q.qid,
        "video_id": q.video_id,
        "kind": q.kind,
        "answered": not a.abstain,
        "t_ans": a.t,
        "feats": a.feats,
    }
    if q.kind == "absent":
        r["correct"] = bool(a.abstain)
        return r
    if a.abstain:
        r.update(correct=False, loc_correct=False if q.gt_loc_class is not None else None, time_ok=False, t_err=None)
        return r
    assert a.t is not None and q.gt_t0 is not None and q.gt_t1 is not None
    terr = 0.0 if q.gt_t0 <= a.t <= q.gt_t1 else min(abs(a.t - q.gt_t0), abs(a.t - q.gt_t1))
    r["t_err"] = terr
    if q.kind == "where":
        r["time_ok"] = bool(q.gt_t0 - WHERE_PRE_S <= a.t <= q.gt_t1 + WHERE_POST_S)
        if q.gt_loc_class is not None:
            pred = loc_ids[a.loc_idx] if a.loc_idx is not None else None
            r["loc_correct"] = bool(pred == q.gt_loc_class)
            r["pred_loc"] = pred
            r["correct"] = r["loc_correct"]
        else:
            r["loc_correct"] = None
            r["correct"] = r["time_ok"]
    elif q.kind == "temporal":
        r["time_ok"] = bool(terr <= TEMPORAL_TOL_S)
        r["time_ok30"] = bool(terr <= 30.0)
        r["correct"] = r["time_ok"]
    elif q.kind == "state":
        r["time_ok"] = bool(terr <= TEMPORAL_TOL_S)
        r["correct"] = bool(a.state == q.gt_state)
        r["pred_state"] = a.state
    return r


def run(
    sys: System,
    split: str,
    heads: Heads,
    model: str,
    queries: list[Query] | None = None,
    feats_cache: dict | None = None,
) -> tuple[list[dict], dict]:
    qs = queries if queries is not None else queries_for(split)
    vids = available(split, model)
    by_vid: dict[str, list[Query]] = {}
    for q in qs:
        by_vid.setdefault(q.video_id, []).append(q)
    recs, costs = [], []
    for vid in vids:
        if vid not in by_vid:
            continue
        if feats_cache is not None and vid in feats_cache:
            feats, regions = feats_cache[vid]
        else:
            feats = load_feats(vid, (model, ""))
            regions = RegionIndex(feats)
        view, cost = build_view(sys, vid, feats, heads, regions)
        costs.append(cost)
        for q in by_vid[vid]:
            kind = "where" if q.kind == "absent" else q.kind
            pq = ParsedQuery(kind, q.obj_key, _verb_name(q.verb_class))
            cfg = sys.cfg_by_kind.get(kind, sys.cfg)
            if sys.memory == "frames" and cfg.kinds_override is None:
                cfg = replace(cfg, kinds_override=("frame",))
            a = answer(view, pq, q.obj_class, q.t_query, cfg, sys.state_head)
            rec = score(q, a, heads.loc_ids)
            rec["system"] = sys.name
            recs.append(rec)
    H = sum(c["hours"] for c in costs)
    summary_cost = {
        "hours": H,
        "heavy_frames": sum(c["heavy_frames"] for c in costs),
        "heavy_per_hour": sum(c["heavy_frames"] for c in costs) / max(H, 1e-9),
        "events_per_hour": sum(c["n_events"] for c in costs) / max(H, 1e-9),
        "frac_processed": sum(c["heavy_frames"] for c in costs) / max(1, sum(c["frames"] for c in costs)),
    }
    return recs, summary_cost


def _verb_name(vc: int | None) -> str | None:
    return {
        0: "take",
        1: "put",
        2: "wash",
        3: "open",
        4: "close",
        5: "put",
        6: "turn-on",
        8: "turn-off",
        13: "put",
    }.get(vc if vc is not None else -1)


def metrics(recs: list[dict], seed: int = 0) -> dict:
    """Accuracy per query family with 95% cluster-bootstrap CIs (resampling videos)."""
    out: dict = {}

    def ci(vals: list, groups: list) -> dict:
        if not vals:
            return {"n": 0}
        m, lo, hi = bootstrap_ci(np.array(vals, float), groups=np.array(groups), seed=seed)
        return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}

    def sel(pred: Callable[[dict], bool], key: str) -> dict:
        rr = [r for r in recs if pred(r) and r.get(key) is not None]
        return ci([float(r[key]) for r in rr], [r["video_id"] for r in rr])

    out["where_loc_acc"] = sel(lambda r: r["kind"] == "where" and r.get("loc_correct") is not None, "loc_correct")
    out["where_time_acc"] = sel(lambda r: r["kind"] == "where", "time_ok")
    out["where_acc"] = sel(lambda r: r["kind"] == "where", "correct")
    out["temporal_acc"] = sel(lambda r: r["kind"] == "temporal", "time_ok")
    out["temporal_acc30"] = sel(lambda r: r["kind"] == "temporal", "time_ok30")
    out["state_acc"] = sel(lambda r: r["kind"] == "state", "correct")
    out["absent_abstain"] = sel(lambda r: r["kind"] == "absent", "correct")
    out["absent_hallucination"] = {"n": 0}
    if out["absent_abstain"].get("n"):
        out["absent_hallucination"] = {
            "mean": 1 - out["absent_abstain"]["mean"],
            "lo": 1 - out["absent_abstain"]["hi"],
            "hi": 1 - out["absent_abstain"]["lo"],
            "n": out["absent_abstain"]["n"],
        }
    terrs = [r["t_err"] for r in recs if r["kind"] == "where" and r.get("t_err") is not None]
    out["where_median_t_err_s"] = float(np.median(terrs)) if terrs else None
    out["answer_rate"] = float(np.mean([r["answered"] for r in recs if r["kind"] != "absent"])) if recs else None
    return out
