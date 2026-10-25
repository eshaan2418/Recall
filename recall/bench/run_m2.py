"""M2 — learned memory trigger vs uniform / motion / skin-heuristic / teacher-contact / oracle triggers.

Every policy is evaluated with the *same* Recall event pipeline and answer config (tuned on val for the
all-frames system in M1), at matched heavy-model budgets. Thresholds for budgeted policies are calibrated
on **val** scores, then applied unchanged to test; the realised test budget is what we report.

Outputs: results/m2/{gate_training.json, pareto_test.json, figs/pareto_*.png}
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, metrics, queries_for, run
from recall.bench.run_m1 import MODEL, load_split, make_oracle, train_state_heads
from recall.data.loaders.epic import load_narrations
from recall.data.loaders.hoi import PORTABLE, STATIONARY
from recall.data.queries import interaction_windows
from recall.paths import CACHE, RESULTS
from recall.perception.heads import Heads
from recall.qa.answer import AnswerConfig
from recall.trigger.gate import (
    GateData,
    frame_inputs,
    imu_features,
    labels,
    save_gate,
    score_video,
    skin_score,
    train_gate,
)
from recall.trigger.policies import burst, motion_score, threshold_for_budget, threshold_policy, uniform

BUDGETS = [0.03, 0.06, 0.1, 0.2, 0.3, 0.5]
UNIFORM_K = [1, 2, 3, 4, 6, 10, 20, 40]
BURST = 4


def gate_data(split: dict, narr) -> list[GateData]:
    out = []
    for vid, (f, _) in split.items():
        out.append(
            GateData(
                frame_inputs(f["thumb"]), imu_features(vid, f["t"]), labels(f["t"], interaction_windows(narr, vid)), vid
            )
        )
    return out


def teacher_contact(f: dict) -> np.ndarray:
    st = f["hands"][..., 2]
    pres = f["hands"][..., 0] > 0
    return ((pres & np.isin(st, (PORTABLE, STATIONARY))).any(1)).astype(np.float32)


def tuned_cfgs() -> dict[str, AnswerConfig]:
    m1 = json.loads((RESULTS / "m1" / "systems_test.json").read_text())
    tl = m1["systems"]["Recall-all-frames"]["val_tuning"]
    fam = {"where": "where", "absent": "where", "temporal": "temporal", "state": "state"}
    return {
        k: AnswerConfig(
            mode=tl[v]["mode"], recency_s=tl[v]["recency_s"], tau=tl[v]["tau"], score=tl[v].get("score", "prob")
        )
        for k, v in fam.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    narr = load_narrations()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    heads = Heads.load(f"{MODEL}_seed0")
    st_ev, _, _ = train_state_heads(train, narr, heads)
    cfgs = tuned_cfgs()
    qs_test = [q for q in queries_for("test") if q.video_id in test]
    gtr, gva, gte = gate_data(train, narr), gate_data(val, narr), gate_data(test, narr)

    # ---------- train gates ----------
    gate_log: dict = {}
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}  # variant -> split -> vid -> scores
    for use_imu in (True, False):
        for seed in a.seeds:
            name = f"learned{'+imu' if use_imu else ''}_s{seed}"
            m, log = train_gate(gtr, gva, seed, epochs=a.epochs, use_imu=use_imu)
            save_gate(m, CACHE / "gates" / f"{name}.pt")
            gate_log[name] = log
            scores[name] = {
                "val": {d.vid: score_video(m, d, use_imu) for d in gva},
                "test": {d.vid: score_video(m, d, use_imu) for d in gte},
            }
            from sklearn.metrics import average_precision_score

            yt = np.concatenate([d.y for d in gte])
            st = np.concatenate([scores[name]["test"][d.vid] for d in gte])
            log["test_ap"] = float(average_precision_score(yt, st))
            print(name, "val AP", log["val_ap"], "test AP", log["test_ap"])
    write_result(
        "m2/gate_training.json",
        {
            **result_header("offline-eval", None, {"epochs": a.epochs, "seeds": a.seeds}),
            "gates": gate_log,
            "test_pos_rate": float(np.mean(np.concatenate([d.y for d in gte]))),
        },
    )

    # ---------- cheap heuristic scores ----------
    for name, fn in (
        ("motion", lambda f: motion_score(f["thumb"])),
        ("skin", lambda f: skin_score(f["thumb"])),
        ("teacher-contact", teacher_contact),
    ):
        scores[name] = {
            "val": {v: fn(f) for v, (f, _) in val.items()},
            "test": {v: fn(f) for v, (f, _) in test.items()},
        }

    rows = []

    def evaluate(policy_name: str, pol, budget: float | None, seed: int | None) -> None:
        sys = System(policy_name, pol, cfg=cfgs["where"], cfg_by_kind=cfgs, state_head=st_ev)
        recs, cost = run(sys, "test", heads, MODEL, qs_test, test)
        m = metrics(recs)
        rows.append({"policy": policy_name, "target_budget": budget, "seed": seed, **cost, "metrics": m})
        print(
            f"{policy_name:22s} budget={budget} seed={seed} frac={cost['frac_processed']:.3f} "
            f"where={m['where_acc'].get('mean'):.3f} temporal={m['temporal_acc'].get('mean'):.3f}"
        )

    evaluate("oracle", make_oracle(narr), None, None)
    for k in UNIFORM_K:
        evaluate(f"uniform-k{k}", lambda vid, f, k=k: uniform(len(f["t"]), k), 1 / k, None)
    for k in (4, 8, 16, 32):  # uniform bursts: same burst length as triggered policies
        evaluate(f"uniform-burst-k{k}", lambda vid, f, k=k: burst(uniform(len(f["t"]), k), BURST), None, None)
    for name in list(scores):
        for b in BUDGETS:
            val_s = np.concatenate(list(scores[name]["val"].values()))
            thr = threshold_for_budget(val_s, b, BURST)
            seed = int(name.split("_s")[-1]) if "_s" in name else None
            evaluate(
                name.split("_s")[0] if seed is not None else name,
                lambda vid, f, name=name, thr=thr: threshold_policy(scores[name]["test"][vid], thr, BURST),
                b,
                seed,
            )
    write_result(
        "m2/pareto_test.json",
        {
            **result_header("offline-eval", None, {"budgets": BUDGETS, "burst": BURST, "uniform_k": UNIFORM_K}),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
