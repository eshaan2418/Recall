"""`make reproduce-quick` core: the whole M1–M4 pipeline on SYNTHETIC streams (no licensed data, ~1 min).

It exercises trigger policies → events → memory store → QA → calibration → metrics/CIs end to end, so a clean
clone (and CI) can verify the code path. Numbers in results/quick/ are about synthetic data and are never quoted
as findings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, build_view, metrics, score
from recall.data.synthetic import SynthWorld, make_video, where_queries
from recall.memory.store import MemoryStore, StoreConfig
from recall.perception.events import extract_events
from recall.qa import calibrate as cal
from recall.qa.answer import AnswerConfig, answer
from recall.qa.parse import ParsedQuery
from recall.trigger.policies import motion_score, threshold_for_budget, threshold_policy, uniform


def run_policy(policy, world: SynthWorld, vids: range) -> tuple[list[dict], float]:
    heads = world.heads()
    recs, frac = [], []
    for k in vids:
        vid = f"S{k}"
        feats, narr = make_video(world, vid, seed=k)
        sys = System("x", policy, cfg=AnswerConfig(tau=0.2))
        view, cost = build_view(sys, vid, feats, heads)
        frac.append(cost["heavy_frames"] / cost["frames"])
        for q in where_queries(narr, vid, 600.0):
            a = answer(view, ParsedQuery("where", q.obj_key), q.obj_class, q.t_query, sys.cfg)
            r = score(q, a, heads.loc_ids)
            recs.append(r)
    return recs, float(np.mean(frac))


def main() -> None:
    world = SynthWorld(noise=0.35)
    pareto = []
    for k in (1, 2, 4, 8, 16):
        recs, frac = run_policy(lambda vid, f, k=k: uniform(len(f["t"]), k), world, range(6))
        pareto.append({"policy": f"uniform-k{k}", "frac": frac, "where_loc_acc": metrics(recs)["where_loc_acc"]})
    for b in (0.1, 0.3):

        def pol(vid: str, f: dict, b: float = b) -> np.ndarray:
            s = motion_score(f["thumb"])
            return threshold_policy(s, threshold_for_budget(s, b, 4), 4)

        recs, frac = run_policy(pol, world, range(6))
        pareto.append({"policy": f"motion@{b}", "frac": frac, "where_loc_acc": metrics(recs)["where_loc_acc"]})
    # calibration on synthetic val/test videos
    hard = SynthWorld(noise=0.8, seed=1)  # harder world so that errors exist to calibrate against
    rv, _ = run_policy(lambda vid, f: uniform(len(f["t"]), 6), hard, range(6, 14))
    rt, _ = run_policy(lambda vid, f: uniform(len(f["t"]), 6), hard, range(14, 22))
    c = cal.fit(rv, "logistic")
    calib = cal.summary(c.predict(rt), cal.labels(rt))
    # storage: real SQLite store of one synthetic video's events
    feats, _ = make_video(world, "S0", seed=0)
    evs = extract_events(feats, np.ones(len(feats["t"]), bool), "S0")
    with tempfile.TemporaryDirectory() as d:
        st = MemoryStore(Path(d) / "m.db", StoreConfig(store_thumbs=False))
        st.add_all(evs)
        size = st.size_bytes()
        st.db.close()
    write_result(
        "quick/synthetic.json",
        {
            **result_header("offline-eval", 0, {"world": vars(world)}),
            "WARNING": "synthetic streams — pipeline check only, not a finding",
            "pareto": pareto,
            "calibration": calib,
            "store_bytes_one_video": size,
            "n_events_one_video": len(evs),
        },
    )
    print("quick OK:", [(p["policy"], round(p["where_loc_acc"]["mean"], 3)) for p in pareto], calib)


if __name__ == "__main__":
    main()
