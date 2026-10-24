"""M1 — offline reference pipeline + baselines on the EPIC-derived query sets.

1. Train object/location heads (region + frame) on train participants, tune on val.
2. Train on/off state heads (event + frame) on train.
3. For every system, tune the answer config per query family on val, then evaluate on test.

Systems: Recall(all frames) · B4 oracle trigger · B2 frame-sample+CLIP at N ∈ {1,2,5,10,30} s ·
B3 manual tagging (q ∈ {0.1,0.25,0.5,1.0}). B1 (large VLM) lives in ``recall/baselines/vlm.py``.

Outputs: results/m1/{heads.json, systems_test.json, records/<system>.jsonl}
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, available, build_view, metrics, queries_for, run
from recall.data.loaders.epic import load_narrations
from recall.data.queries import Query, interaction_windows
from recall.paths import EPIC_ANN, RESULTS
from recall.perception.embed import load_embedder
from recall.perception.events import RegionIndex
from recall.perception.extract import load_feats
from recall.perception.heads import Heads, region_training_data, tune_head
from recall.perception.vocab import location_bank, location_ids, object_bank
from recall.qa.answer import AnswerConfig
from recall.qa.state import fit_state_head, state_training_pairs
from recall.trigger.policies import oracle, uniform

MODEL = "MobileCLIP2-S2"
B2_INTERVALS_S = [1, 2, 5, 10, 30]
TAUS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
MODES = [("latest", 0.0), ("recency", 120.0), ("recency", 600.0)]


def load_split(split: str) -> dict:
    out = {}
    for vid in available(split, MODEL):
        f = load_feats(vid, (MODEL, ""))
        out[vid] = (f, RegionIndex(f))
    return out


def all_frames(vid: str, f: dict) -> np.ndarray:
    return np.ones(len(f["t"]), bool)


def make_oracle(narr: pd.DataFrame):
    def pol(vid: str, f: dict) -> np.ndarray:
        return oracle(f["t"], interaction_windows(narr, vid), pad_s=1.0)

    return pol


def make_uniform(k: int):
    def pol(vid: str, f: dict) -> np.ndarray:
        return uniform(len(f["t"]), k)

    return pol


def train_heads(train: dict, val: dict, narr: pd.DataFrame, seed: int = 0) -> Heads:
    emb = load_embedder(MODEL, "dfndr2b")
    ob, lb = object_bank(emb), location_bank(emb)
    lids = location_ids()
    tr = region_training_data(narr, {k: v[0] for k, v in train.items()}, lids)
    va = region_training_data(narr, {k: v[0] for k, v in val.items()}, lids)
    meta: dict = {"n_train": {k: len(v[1]) for k, v in tr.items()}, "n_val": {k: len(v[1]) for k, v in va.items()}}
    hs = {}
    for name, bank in (("obj", ob), ("loc", lb), ("frame_obj", ob), ("frame_loc", lb)):
        h, log = tune_head(bank, emb.logit_scale, *tr[name], *va[name], seed=seed)
        hs[name] = h
        meta[name] = {k: v for k, v in log.items() if k != "grid"}
        print(name, meta[name])
    heads = Heads(hs["obj"], hs["loc"], hs["frame_obj"], hs["frame_loc"], lids, meta)
    heads.save(f"{MODEL}_seed{seed}")
    return heads


def train_state_heads(train: dict, narr: pd.DataFrame, heads: Heads) -> tuple:
    nt = pd.read_csv(EPIC_ANN / "EPIC_100_noun_classes.csv")
    key2id = dict(zip(nt.key, nt.id, strict=True))
    ev_views, fr_views = {}, {}
    for vid, (f, reg) in train.items():
        ev_views[vid], _ = build_view(System("all", all_frames), vid, f, heads, reg)
        fr_views[vid], _ = build_view(System("b2", make_uniform(2), memory="frames"), vid, f, heads, reg)
    Xe, ye, ce = state_training_pairs(narr, ev_views, key2id)
    Xf, yf, cf = state_training_pairs(narr, fr_views, key2id)
    return fit_state_head(Xe, ye, ce), fit_state_head(Xf, yf, cf), {"n_event_pairs": len(ye), "n_frame_pairs": len(yf)}


def tune_system(sys: System, val: dict, heads: Heads, qs_val: list[Query]) -> tuple[System, dict]:
    """Choose (mode, tau) per family on val. where+absent are tuned jointly (mean of the two accuracies)."""
    best_cfg: dict[str, AnswerConfig] = {}
    log: dict = {}
    fams = {"where": ("where", "absent"), "temporal": ("temporal",), "state": ("state",)}
    # one pass per config over val, then pick per family
    grid = [
        replace(sys.cfg, mode=m, recency_s=r or sys.cfg.recency_s, tau=t)
        for (m, r), t in itertools.product(MODES, TAUS)
    ]
    scores: dict[str, list[float]] = {f: [] for f in fams}
    for cfg in grid:
        s2 = replace(sys, cfg=cfg, cfg_by_kind={})
        recs, _ = run(s2, "val", heads, MODEL, qs_val, val)
        m = metrics(recs)
        scores["where"].append(
            np.nanmean([m["where_acc"].get("mean", np.nan), m["absent_abstain"].get("mean", np.nan)])
        )
        scores["temporal"].append(m["temporal_acc"].get("mean", np.nan))
        scores["state"].append(m["state_acc"].get("mean", np.nan))
    for fam, kinds in fams.items():
        j = int(np.nanargmax(scores[fam]))
        for k in kinds:
            best_cfg[k] = grid[j]
        log[fam] = {
            "mode": grid[j].mode,
            "recency_s": grid[j].recency_s,
            "tau": grid[j].tau,
            "val_score": scores[fam][j],
        }
    return replace(sys, cfg_by_kind=best_cfg), log


def manual_tagging(narr: pd.DataFrame, qs: list[Query], q_tag: float, seed: int) -> list[dict]:
    """B3: the user tags each put-down with probability q (perfect labels when tagged)."""
    rng = np.random.default_rng(seed)
    recs = []
    by_vid: dict[str, list[Query]] = {}
    for q in qs:
        by_vid.setdefault(q.video_id, []).append(q)
    for vid, vq in by_vid.items():
        d = narr[(narr.video_id == vid) & narr.verb_class.isin([1, 5, 13])]
        tagged = d[rng.random(len(d)) < q_tag]
        for q in vq:
            r: dict = {"qid": q.qid, "video_id": vid, "kind": q.kind, "system": f"B3_q{q_tag}"}
            if q.kind == "absent":
                r.update(correct=True, answered=False)
            elif q.kind != "where":
                r.update(correct=False, answered=False, time_ok=False, t_err=None)  # tags carry no state/verb info
            else:
                hit = tagged[(tagged.noun_class == q.obj_class) & (tagged.stop_s < q.t_query)]
                if len(hit) == 0:
                    r.update(
                        correct=False,
                        answered=False,
                        time_ok=False,
                        t_err=None,
                        loc_correct=False if q.gt_loc_class is not None else None,
                    )
                else:
                    h = hit.iloc[-1]
                    ok_t = bool(abs(h.start_s - q.gt_t0) < 1e-6)
                    r.update(answered=True, time_ok=ok_t, t_err=0.0 if ok_t else abs(h.start_s - q.gt_t0))
                    if q.gt_loc_class is not None:
                        r["loc_correct"] = ok_t  # same event → the user typed the right place
                        r["correct"] = ok_t
                    else:
                        r["loc_correct"] = None
                        r["correct"] = ok_t
            recs.append(r)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="skip B2 intervals except 2 s and 10 s")
    a = ap.parse_args()
    narr = load_narrations()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    print("videos", len(train), len(val), len(test))
    heads = train_heads(train, val, narr, a.seed)
    st_ev, st_fr, st_log = train_state_heads(train, narr, heads)
    qs_val = [q for q in queries_for("val") if q.video_id in val]
    qs_test = [q for q in queries_for("test") if q.video_id in test]

    systems = [
        System("Recall-all-frames", all_frames, state_head=st_ev),
        System("B4-oracle-trigger", make_oracle(narr), state_head=st_ev),
    ]
    for n in [2, 10] if a.quick else B2_INTERVALS_S:
        systems.append(System(f"B2-frame-CLIP-{n}s", make_uniform(int(2 * n)), memory="frames", state_head=st_fr))

    out: dict = {"heads": heads.meta, "state_heads": st_log, "systems": {}}
    (RESULTS / "m1" / "records").mkdir(parents=True, exist_ok=True)
    for sys in systems:
        tuned, tlog = tune_system(sys, val, heads, qs_val)
        recs, cost = run(tuned, "test", heads, MODEL, qs_test, test)
        m = metrics(recs, seed=a.seed)
        out["systems"][sys.name] = {"val_tuning": tlog, "cost": cost, "metrics": m}
        print(sys.name, json.dumps({k: (v.get("mean") if isinstance(v, dict) else v) for k, v in m.items()}))
        with open(RESULTS / "m1" / "records" / f"{sys.name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=float) + "\n")
    for qt in (0.1, 0.25, 0.5, 1.0):
        recs = manual_tagging(narr, qs_test, qt, a.seed)
        m = metrics(recs, seed=a.seed)
        H = sum(len(v[0]["t"]) for v in test.values()) / 7200
        d = narr[narr.video_id.isin(list(test)) & narr.verb_class.isin([1, 5, 13])]
        n_tags = float(qt * len(d))
        out["systems"][f"B3-manual-q{qt}"] = {"cost": {"expected_tags_per_hour": n_tags / H, "hours": H}, "metrics": m}
        print(f"B3 q={qt}", m["where_acc"].get("mean"))
    out["n_queries_test"] = {k: sum(q.kind == k for q in qs_test) for k in ("where", "temporal", "state", "absent")}
    out["n_videos"] = {"train": len(train), "val": len(val), "test": len(test)}
    write_result(
        "m1/systems_test.json",
        {
            **result_header("offline-eval", a.seed, {"model": MODEL, "quick": a.quick, "taus": TAUS, "modes": MODES}),
            **out,
        },
    )
    print("wrote results/m1/systems_test.json")


if __name__ == "__main__":
    main()
