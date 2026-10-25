"""Export small JSON fixtures from the Python reference implementation for the Kotlin parity tests.

Run from the repo root:  ./scripts/android_fixtures.sh   (== .venv/bin/python android/tools/make_parity_fixtures.py)
Writes android/app/src/test/resources/parity/*.json. Uses only the Python modules the Kotlin code ports
(parse, trigger policies, BlendHead, answerer, calibrator, event extractor, renderer); no data files needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from recall.memory.schema import Event
from recall.perception.events import extract_events
from recall.perception.extract import REGION_KINDS
from recall.perception.heads import BlendHead
from recall.qa import calibrate as cal
from recall.qa.answer import AnswerConfig, MemoryView, answer_state, answer_temporal, answer_where
from recall.qa.generate import clock, render
from recall.qa.parse import parse
from recall.qa.state import StateHead
from recall.trigger.policies import burst

OUT = Path(__file__).resolve().parents[1] / "app" / "src" / "test" / "resources" / "parity"
rng = np.random.default_rng(1234)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def fl(x) -> list:
    return np.asarray(x, np.float64).round(9).tolist()


def probe_export(clf: LogisticRegression) -> dict:
    """sklearn LR → softmax rows. Binary LR (coef [1,D]) becomes two rows [0, w] so softmax == sigmoid."""
    W, b = clf.coef_, clf.intercept_
    if W.shape[0] == 1:
        W = np.vstack([np.zeros_like(W), W])
        b = np.array([0.0, b[0]])
    return {"classes": clf.classes_.astype(int).tolist(), "w": fl(W), "b": fl(b)}


def head_export(h: BlendHead) -> dict:
    d = {"bank": fl(h.text_bank), "scale": h.scale, "alpha": h.alpha}
    if h.probe is not None:
        d["probe"] = probe_export(h.probe)
    return d


def write(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=None, separators=(",", ":")))
    print("wrote", OUT / f"{name}.json")


# ------------------------------------------------------------------ parse
QUESTIONS = [
    "Where are my keys?", "where is the phone", "Where's my wallet?", "wheres the remote", "Where did I leave my glasses?",
    "where did I put the scissors", "Where did I place the mug?", "where did i last see my passport", "Where did I park?",
    "Have you seen my charger?", "Find my headphones", "find the  TV   remote!", "keys", "the bottle", "my red pen.",
    "When did I last take my pills?", "when did i take the medication last", "When did I pick up the knife?",
    "When did I put down the cup?", "when did I open the fridge", "When did I close the door?",
    "when did I turn off the oven", "When did I switch on the kettle?", "when did i wash the pan",
    "When did I rinse the cup?", "when did i grab my bag", "when did I use the hob last", "When did I drink coffee?",
    "Did I turn off the stove?", "did i switch off the tap", "Have I turned off the oven?", "Is the tap still on?",
    "is the hob off", "Is the kettle on?", "  WHERE ARE THE SPOONS??? ",
]


def fixture_parse() -> None:
    write("parse", [{"q": q, **parse(q).__dict__} for q in QUESTIONS])


# ------------------------------------------------------------------ burst
def fixture_burst() -> None:
    cases = []
    for length, refr in [(1, 0), (4, 0), (4, 2), (3, 5), (6, 1)]:
        fire = rng.random(60) < 0.15
        cases.append({"fire": fire.tolist(), "length": length, "refractory": refr,
                      "out": burst(fire, length, refr).tolist()})
    write("burst", cases)


# ------------------------------------------------------------------ heads
def make_head(C: int, D: int, alpha: float, probe_classes: list[int] | None, scale: float = 50.0) -> BlendHead:
    bank = unit(rng.normal(size=(C, D))).astype(np.float32)
    if probe_classes is None:
        return BlendHead(bank, scale, alpha)
    X = unit(rng.normal(size=(40 * len(probe_classes), D))).astype(np.float32)
    y = rng.choice(probe_classes, size=len(X))
    for k in probe_classes:  # make the classes somewhat separable
        X[y == k] += 0.6 * bank[k]
    X = unit(X).astype(np.float32)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return BlendHead(bank, scale, alpha, clf, clf.classes_)


def fixture_heads() -> None:
    D = 16
    cases = []
    for alpha, pc in [(1.0, None), (0.5, [0, 2, 3, 7]), (0.25, [1, 5]), (0.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]:
        h = make_head(12, D, alpha, pc)
        E = unit(rng.normal(size=(10, D))).astype(np.float32)
        cases.append({"head": head_export(h), "emb": fl(E), "logp": fl(h.log_probs(E))})
    write("heads", cases)


# ------------------------------------------------------------------ answers
def fixture_answers() -> None:
    D = 16
    obj = make_head(12, D, 0.5, [0, 2, 3, 7], scale=30.0)
    loc = make_head(5, D, 0.75, [0, 1, 2, 3, 4], scale=30.0)
    N = 40
    t = np.sort(rng.uniform(0, 3000, N)).round(3)
    kinds = rng.choice(["put", "take", "touch"], N, p=[0.4, 0.35, 0.25])
    events = []
    for i in range(N):
        o = unit(rng.normal(size=D) + 1.2 * obj.text_bank[rng.integers(0, 6)])
        l = unit(rng.normal(size=D) + 1.2 * loc.text_bank[rng.integers(0, 5)])
        s = unit(rng.normal(size=D))
        events.append(Event("s", float(t[i]), str(kinds[i]), int(rng.integers(0, 2)), i, o.astype(np.float32),
                            l.astype(np.float32), s.astype(np.float32), n_obs=int(rng.integers(1, 9)),
                            hand_score=float(rng.uniform(0.5, 1)), release_observed=bool(rng.random() < 0.5)))
    view = MemoryView.build(events, obj, loc)
    Xs = unit(rng.normal(size=(60, 2 * D)))
    ys = rng.integers(0, 2, 60)
    st = StateHead(LogisticRegression(max_iter=2000).fit(Xs, ys), {3: 0.8}, 0.5)
    queries = []
    for cfg in [AnswerConfig(), AnswerConfig(mode="best", tau=0.1), AnswerConfig(mode="recency", recency_s=300.0),
                AnswerConfig(mode="latest", score="ratio", tau=0.3), AnswerConfig(tau=0.9, fallback_take=False),
                AnswerConfig(mode="recency", recency_s=300.0, tau=0.02, loc_mode="marginal")]:
        for cls in range(12):
            for tq in (800.0, 2000.0, 3100.0):
                cd = {"mode": cfg.mode, "tau": cfg.tau, "recency_s": cfg.recency_s, "score": cfg.score,
                      "fallback_take": cfg.fallback_take, "loc_mode": cfg.loc_mode}
                for kind, verb in [("where", None), ("temporal", "take"), ("temporal", "put"), ("temporal", "open"),
                                   ("state", None)]:
                    if kind == "where":
                        a = answer_where(view, cls, tq, cfg)
                    elif kind == "temporal":
                        a = answer_temporal(view, cls, verb, tq, cfg)
                    else:
                        a = answer_state(view, cls, tq, cfg, st)
                    queries.append({"kind": kind, "verb": verb, "cls": cls, "tq": tq, "cfg": cd,
                                    "abstain": a.abstain, "event_idx": a.event_idx, "t": a.t, "loc_idx": a.loc_idx,
                                    "state": a.state, "feats": {k: float(v) for k, v in a.feats.items()} if not a.abstain else {}})
    write("answers", {
        "obj": head_export(obj), "loc": head_export(loc),
        "state_probe": {"coef": fl(st.clf.coef_[0]), "intercept": float(st.clf.intercept_[0])},
        "events": [{"t": e.t, "kind": e.kind, "side": e.side, "obj": fl(e.obj_emb), "loc": fl(e.loc_emb),
                    "scene": fl(e.scene_emb), "n_obs": e.n_obs, "hand_score": e.hand_score,
                    "release": e.release_observed} for e in events],
        "queries": queries,
    })


# ------------------------------------------------------------------ calibration
def fake_recs(n: int) -> list[dict]:
    recs = []
    for _ in range(n):
        kind = str(rng.choice(["where", "temporal", "state"]))
        f = {"score": rng.uniform(0, 1), "is_argmax": float(rng.random() < 0.6), "score_margin": rng.uniform(-0.5, 1),
             "loc_p": rng.uniform(0.2, 1), "loc_margin": rng.uniform(0, 0.8), "log_n_obs": np.log1p(rng.integers(1, 9)),
             "log_age_s": np.log1p(rng.uniform(0, 4000)), "n_conflicts": float(rng.integers(0, 5)),
             "second_score": rng.uniform(0, 0.8), "release_obs": float(rng.random() < 0.5),
             "hand_score": rng.uniform(0.5, 1), "n_candidates": float(rng.integers(1, 8))}
        if kind == "where":
            f["fallback"] = float(rng.random() < 0.2)
        if kind == "state":
            f["p_on"] = rng.uniform(0, 1)
        z = 2.5 * f["score"] + f["is_argmax"] - 0.4 * f["n_conflicts"] + rng.normal(0, 0.7)
        answered = bool(rng.random() < 0.9)
        recs.append({"kind": kind, "answered": answered, "correct": bool(z > 1.2), "feats": {k: float(v) for k, v in f.items()}})
    return recs


def calib_export(c: cal.Calibrator, threshold: float) -> dict:
    names = [*cal.FEATURES, "fallback", "p_on", "p_on_margin", *[f"kind={k}" for k in cal.KINDS]]
    d = {"features": names, "coef": fl(c.lr.coef_[0]), "intercept": float(c.lr.intercept_[0]),
         "iso_x": [], "iso_y": [], "threshold": threshold}
    if c.iso is not None:
        d["iso_x"], d["iso_y"] = fl(c.iso.X_thresholds_), fl(c.iso.y_thresholds_)
    return d


def fixture_calibration() -> None:
    val, test = fake_recs(400), fake_recs(80)
    out = []
    for meth in ("logistic", "isotonic"):
        c = cal.fit(val, meth)
        th = cal.threshold_for_precision(c.predict(val), cal.labels(val), 0.7)
        out.append({"calibrator": calib_export(c, th), "recs": test, "p": fl(c.predict(test))})
    write("calibration", out)


# ------------------------------------------------------------------ events
def fixture_events() -> None:
    cases = []
    for seed in range(4):
        r = np.random.default_rng(100 + seed)
        T, D = 120, 8
        t = (np.arange(T) * 0.5).astype(np.float32)
        protos = unit(r.normal(size=(4, D)))
        hands = np.zeros((T, 2, 7), np.float32)
        reg_t, reg_kind, reg_side, reg_box, reg_emb = [], [], [], [], []
        for side in (0, 1):
            i = 0
            while i < T:  # alternating bouts: none / portable(obj k) / stationary
                L = int(r.integers(2, 12))
                mode = r.choice(["none", "portable", "stationary", "flicker"], p=[0.35, 0.4, 0.15, 0.1])
                k = int(r.integers(0, 4))
                for j in range(i, min(T, i + L)):
                    if mode == "none" or (mode == "flicker" and r.random() < 0.5):
                        continue
                    st = 3 if mode in ("portable", "flicker") else 4
                    hands[j, side] = [1, r.uniform(0.5, 1), st, 0.1, 0.2, 0.4, 0.6]
                    b = np.sort(r.uniform(0, 1, 4).reshape(2, 2), axis=0).T.reshape(-1)[[0, 2, 1, 3]]
                    reg_t.append(j); reg_kind.append(REGION_KINDS.index("obj")); reg_side.append(side)
                    reg_box.append(b); reg_emb.append(unit(protos[k] + 0.3 * r.normal(size=D)))
                i += L
        for side in (0, 1):  # pctx crops for every frame whose previous frame had an object
            prev = {j for j, s in zip(reg_t, reg_side, strict=True) if s == side}
            for j in range(1, T):
                if j - 1 in prev:
                    reg_t.append(j); reg_kind.append(REGION_KINDS.index("pctx")); reg_side.append(side)
                    reg_box.append(np.array([0, 0, 1, 1])); reg_emb.append(unit(r.normal(size=D)))
        feats = {"t": t, "hands": hands, "frame_emb": unit(r.normal(size=(T, D))).astype(np.float16),
                 "reg_t": np.array(reg_t), "reg_kind": np.array(reg_kind), "reg_side": np.array(reg_side),
                 "reg_box": np.array(reg_box, np.float32).reshape(-1, 4), "reg_emb": np.array(reg_emb, np.float16)}
        processed = r.random(T) < [0.95, 0.7, 0.4, 0.25][seed]
        evs = extract_events(feats, processed, "s")
        cases.append({
            "t": fl(t), "hands": fl(hands), "frame_emb": fl(feats["frame_emb"]), "processed": processed.tolist(),
            "regions": [{"t": int(a), "kind": REGION_KINDS[int(b)], "side": int(c), "box": fl(d), "emb": fl(e)}
                        for a, b, c, d, e in zip(feats["reg_t"], feats["reg_kind"], feats["reg_side"],
                                                 feats["reg_box"], feats["reg_emb"], strict=True)],
            "events": [{"t": e.t, "kind": e.kind, "side": e.side, "obj": fl(e.obj_emb), "loc": fl(e.loc_emb),
                        "scene": fl(e.scene_emb), "n_obs": e.n_obs, "hand_score": e.hand_score,
                        "release": e.release_observed, "box": fl(e.box)} for e in evs],
        })
    write("events", cases)


# ------------------------------------------------------------------ render
def fixture_render() -> None:
    from recall.qa.answer import Answer

    rows = []
    for kind, ab, t, st, conf, loc in [
        ("where", False, 3725.0, None, 0.82, "kitchen counter"), ("where", False, 50000.4, None, 0.345, "drawer"),
        ("where", False, 100.0, None, 0.125, "shelf"), ("where", True, None, None, None, None),
        ("temporal", False, 43200.0, None, 0.675, None), ("state", False, 86399.0, "off", 0.905, None),
        ("state", False, 0.0, "on", 0.6, None), ("where", False, 7200.0, None, 0.2, "table"),
    ]:
        a = Answer(kind, ab, None, t, None, st)
        r = render(a, "keys", loc, conf, 0.15)
        rows.append({"kind": kind, "abstain": ab, "t": t, "state": st, "conf": conf, "loc": loc, "obj": "keys",
                     "abstain_below": 0.15, "text": r.text, "abstained": r.abstained})
    clocks = [{"t": x, "clock": clock(x)} for x in (0.0, 59.9, 3600.0, 43199.0, 43200.0, 46800.5, 86399.0, 90000.0)]
    write("render", {"rows": rows, "clocks": clocks})


if __name__ == "__main__":
    fixture_parse()
    fixture_burst()
    fixture_heads()
    fixture_answers()
    fixture_calibration()
    fixture_events()
    fixture_render()
