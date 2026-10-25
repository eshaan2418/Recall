"""M4 — calibrated confidence, abstention, grounded answer text.

For each system (Recall all-frames, Recall learned-trigger, B2 frame+CLIP 2 s) we produce val and test
records with the M1-tuned answer configs, fit calibrators on val only, and report on test:
ECE / Brier / AUROC / AURC, reliability diagrams, risk–coverage curves, selective accuracy, and the
hallucination rate on never-seen-object queries before/after confidence-based abstention.

Confidence variants: always-confident (p=1 when answering; how a plain VLM/retriever behaves),
raw retrieval score, logistic calibrator, isotonic-on-logistic.

Also: grounded rendering of every test answer + optional LLM rephrasing (accepted only if verified).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import numpy as np

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, metrics, queries_for, run
from recall.bench.run_m1 import MODEL, all_frames, load_split, make_uniform, train_state_heads
from recall.bench.run_m2 import BURST, gate_data, threshold_for_budget, tuned_cfgs
from recall.data.loaders.epic import load_narrations
from recall.paths import CACHE, RESULTS
from recall.perception.heads import Heads
from recall.perception.vocab import location_names
from recall.qa import calibrate as cal
from recall.qa.answer import Answer, AnswerConfig
from recall.qa.generate import grounded_rewrite, render
from recall.trigger.gate import TinyGate, score_video
from recall.trigger.policies import threshold_policy

TARGET_PRECISION = 0.7


def evaluate_conf(name: str, p: np.ndarray, y: np.ndarray, recs: list[dict]) -> dict:
    out = cal.summary(p, y)
    out["reliability"] = cal.reliability(p, y)
    cov, acc = cal.risk_coverage(p, y)
    k = np.linspace(0, len(p) - 1, 50).astype(int)
    out["risk_coverage"] = {"coverage": cov[k].tolist(), "sel_acc": acc[k].tolist()}
    for kind in ("where", "temporal", "state"):
        m = np.array([r["kind"] == kind for r in recs])
        if m.sum():
            out[f"ece_{kind}"] = cal.ece(p[m], y[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--gate", default="learned+imu_s0")
    ap.add_argument("--llm", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    ap.add_argument("--n-llm", type=int, default=100)
    a = ap.parse_args()
    narr = load_narrations()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    heads = Heads.load(f"{MODEL}_seed0")
    st_ev, st_fr, _ = train_state_heads(train, narr, heads)
    cfgs = tuned_cfgs()
    m1 = json.loads((RESULTS / "m1" / "systems_test.json").read_text())
    tl = m1["systems"]["B2-frame-CLIP-2s"]["val_tuning"]
    fam = {"where": "where", "absent": "where", "temporal": "temporal", "state": "state"}
    cfgs_b2 = {
        k: AnswerConfig(
            mode=tl[v]["mode"], recency_s=tl[v]["recency_s"], tau=tl[v]["tau"], score=tl[v].get("score", "prob")
        )
        for k, v in fam.items()
    }

    import torch

    use_imu = "+imu" in a.gate
    gate = TinyGate(use_imu=use_imu)
    gate.load_state_dict(torch.load(CACHE / "gates" / f"{a.gate}.pt"))
    gsc = {d.vid: score_video(gate, d, use_imu) for d in gate_data(val, narr) + gate_data(test, narr)}
    thr = threshold_for_budget(np.concatenate([gsc[v] for v in val]), a.budget, BURST)

    systems = [
        System("Recall-all-frames", all_frames, cfg=cfgs["where"], cfg_by_kind=cfgs, state_head=st_ev),
        System(
            f"Recall-learned@{a.budget}",
            lambda vid, f: threshold_policy(gsc[vid], thr, BURST),
            cfg=cfgs["where"],
            cfg_by_kind=cfgs,
            state_head=st_ev,
        ),
        System(
            "B2-frame-CLIP-2s",
            make_uniform(4),
            memory="frames",
            cfg=cfgs_b2["where"],
            cfg_by_kind=cfgs_b2,
            state_head=st_fr,
        ),
    ]
    qv = [q for q in queries_for("val") if q.video_id in val]
    qt = [q for q in queries_for("test") if q.video_id in test]
    out: dict = {"systems": {}}
    for sys in systems:
        rv, _ = run(sys, "val", heads, MODEL, qv, val)
        rt, _ = run(sys, "test", heads, MODEL, qt, test)
        yv, yt = cal.labels(rv), cal.labels(rt)
        res: dict = {"test_metrics_no_abstention": metrics(rt)}
        confs = {}
        confs["always-confident"] = np.array([1.0 if r.get("answered") else 0.0 for r in rt])
        confs["raw-score"] = cal.fit(rv, "raw").predict(rt)
        for meth in ("logistic", "isotonic"):
            c = cal.fit(rv, meth)
            confs[meth] = c.predict(rt)
            pv = c.predict(rv)
            th = cal.threshold_for_precision(pv, yv, TARGET_PRECISION)
            pt = confs[meth]
            answered = pt >= th
            absent = np.array([r["kind"] == "absent" for r in rt])
            res[f"{meth}_selective"] = {
                "threshold_from_val": th,
                "target_precision": TARGET_PRECISION,
                "test_coverage": float(answered[~absent].mean()),
                "test_selective_acc": float(yt[answered & ~absent].mean()) if (answered & ~absent).any() else None,
                "absent_hallucination_rate": float(answered[absent].mean()) if absent.any() else None,
                "n_absent": int(absent.sum()),
            }
        absent = np.array([r["kind"] == "absent" for r in rt])
        res["absent_hallucination_no_abstention"] = float(
            np.mean([r.get("answered") for r, ab in zip(rt, absent) if ab])
        )
        res["confidence"] = {k: evaluate_conf(k, v, yt, rt) for k, v in confs.items()}
        out["systems"][sys.name] = res
        print(
            sys.name,
            {k: round(v["ece"], 4) for k, v in res["confidence"].items()},
            {k: res[k] for k in res if k.endswith("_selective")},
        )
        (RESULTS / "m4").mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "m4" / f"records_{sys.name}.jsonl", "w") as fh:
            for r, p in zip(rt, confs["isotonic"], strict=True):
                fh.write(json.dumps({**r, "conf_isotonic": float(p)}, default=float) + "\n")

    # ---------- grounded rendering + optional LLM rephrase on Recall learned-trigger answers ----------
    sysname = f"Recall-learned@{a.budget}"
    rt = [json.loads(line) for line in (RESULTS / "m4" / f"records_{sysname}.jsonl").read_text().splitlines()]
    locs = location_names()
    llm: Callable[[str], str] | None = None
    try:
        from mlx_lm import generate as lgen
        from mlx_lm import load as lload

        loaded = lload(a.llm)
        lm, tok = loaded[0], loaded[1]

        def _llm(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            return lgen(lm, tok, prompt=p, max_tokens=60, verbose=False)

        llm = _llm
    except Exception as e:  # pragma: no cover
        print("LLM unavailable:", e)
    th = out["systems"][sysname]["isotonic_selective"]["threshold_from_val"]
    gen_rows: list[dict] = []
    hallucinated = 0
    for r in rt[: a.n_llm * 4]:
        kind = "where" if r["kind"] == "absent" else r["kind"]
        ans = Answer(kind, not r["answered"], None, r.get("t_ans"), None, r.get("pred_state"))
        loc = None
        if kind == "where" and r["answered"]:
            pl = r.get("pred_loc")
            loc = locs[heads.loc_ids.index(pl)] if pl in heads.loc_ids else "kitchen counter"
        obj = r["qid"]
        base = render(ans, "object", loc, r["conf_isotonic"], th)
        rr = grounded_rewrite(base, loc, locs, llm) if len(gen_rows) < a.n_llm else base
        mentioned = [l for l in locs if l in rr.text.lower() and (loc is None or l not in loc)]
        hallucinated += int(bool(mentioned))
        gen_rows.append(
            {
                "qid": obj,
                "kind": r["kind"],
                "template": base.text,
                "final": rr.text,
                "used_llm": rr.used_llm,
                "abstained": rr.abstained,
            }
        )
    n_llm_tried = sum(1 for g in gen_rows[: a.n_llm] if not g["abstained"])
    out["generation"] = {
        "n_rendered": len(gen_rows),
        "llm": a.llm if llm else None,
        "llm_rewrites_accepted": sum(g["used_llm"] for g in gen_rows),
        "llm_rewrites_attempted": n_llm_tried,
        "answers_mentioning_a_location_not_in_evidence": hallucinated,
        "absent_queries_rendered_as_abstain": sum(1 for g in gen_rows if g["kind"] == "absent" and g["abstained"]),
        "absent_queries_rendered": sum(1 for g in gen_rows if g["kind"] == "absent"),
        "examples": gen_rows[:12],
    }
    print(
        out["generation"]["llm_rewrites_accepted"],
        "/",
        n_llm_tried,
        "LLM rewrites accepted; hallucinated:",
        hallucinated,
    )
    write_result(
        "m4/calibration_test.json",
        {
            **result_header(
                "offline-eval", 0, {"budget": a.budget, "gate": a.gate, "target_precision": TARGET_PRECISION}
            ),
            **out,
        },
    )


if __name__ == "__main__":
    main()
