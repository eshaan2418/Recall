"""Write the Android model bundle (asset contract in android/README.md) into models/litert/.

    python -m recall.compress.export_android [--precision int8dyn] [--budget 0.2]

Files: heads.bin, vocab.json, calibrator.json, answer_config.json, and canonical model names
(embedder.tflite / gate.tflite / hoi_student.tflite copied from the chosen precision). Everything is derived from
train/val artefacts only: heads (M1), val-tuned answer configs (M1), isotonic calibrator fitted on val (M4 recipe),
the gate threshold for the operating budget calibrated on val scores (M2 recipe).
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct

import numpy as np

from recall.compress.export import OUT
from recall.perception.heads import BlendHead, Heads

MODEL = "MobileCLIP2-S2"


def head_bytes(h: BlendHead) -> bytes:
    bank = np.asarray(h.text_bank, np.float32)
    C, D = bank.shape
    out = struct.pack("<iiff", C, D, float(h.scale), float(h.alpha)) + bank.tobytes()
    if h.probe is None or h.alpha >= 1.0:
        return out + struct.pack("<i", 0)
    W, b = h.probe.coef_.astype(np.float32), h.probe.intercept_.astype(np.float32)
    cls = np.asarray(h.probe_classes, np.int32)
    if W.shape[0] == 1:  # binary sklearn LR → two softmax rows so softmax == sigmoid
        W = np.vstack([np.zeros_like(W), W])
        b = np.array([0.0, b[0]], np.float32)
    K = W.shape[0]
    assert len(cls) == K, (len(cls), K)
    return out + struct.pack("<i", K) + cls.tobytes() + W.tobytes() + b.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="int8dyn")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--gate", default="learned+imu_s0")
    a = ap.parse_args()
    import torch

    from recall.bench.harness import System, queries_for, run
    from recall.bench.run_m1 import load_split, train_state_heads
    from recall.bench.run_m2 import BURST, gate_data, tuned_cfgs
    from recall.data.loaders.epic import load_narrations
    from recall.paths import CACHE
    from recall.perception.vocab import location_ids, location_names, noun_names
    from recall.qa import calibrate as cal
    from recall.qa.resolve import _synonyms
    from recall.trigger.gate import TinyGate, score_video
    from recall.trigger.policies import threshold_for_budget, threshold_policy

    heads = Heads.load(f"{MODEL}_seed0")
    (OUT / "heads.bin").write_bytes(head_bytes(heads.obj) + head_bytes(heads.loc))
    (OUT / "vocab.json").write_text(
        json.dumps(
            {"nouns": noun_names(), "synonyms": _synonyms(), "loc_ids": location_ids(), "loc_names": location_names()}
        )
    )
    narr = load_narrations()
    train, val = load_split("train"), load_split("val")
    st_ev, _, _ = train_state_heads(train, narr, heads)
    del train
    cfgs = tuned_cfgs()
    use_imu = "+imu" in a.gate
    gate = TinyGate(use_imu=use_imu)
    gate.load_state_dict(torch.load(CACHE / "gates" / f"{a.gate}.pt", map_location="cpu"))
    gsc = {d.vid: score_video(gate, d, use_imu) for d in gate_data(val, narr)}
    thr = threshold_for_budget(np.concatenate(list(gsc.values())), a.budget, BURST)
    sys = System(
        "bundle",
        lambda vid, f: threshold_policy(gsc[vid], thr, BURST),
        cfg=cfgs["where"],
        cfg_by_kind=cfgs,
        state_head=st_ev,
    )
    rv, _ = run(sys, "val", heads, MODEL, [q for q in queries_for("val") if q.video_id in val], val)
    c = cal.fit(rv, "isotonic")
    th = cal.threshold_for_precision(c.predict(rv), cal.labels(rv), 0.7)
    assert c.lr is not None and c.iso is not None
    names = [*cal.FEATURES, "fallback", "p_on", "p_on_margin", *[f"kind={k}" for k in cal.KINDS]]
    (OUT / "calibrator.json").write_text(
        json.dumps(
            {
                "features": names,
                "coef": c.lr.coef_[0].tolist(),
                "intercept": float(c.lr.intercept_[0]),
                "iso_x": c.iso.X_thresholds_.tolist(),
                "iso_y": c.iso.y_thresholds_.tolist(),
                "threshold": float(th),
            }
        )
    )
    sp: dict = {"state_prior": {str(k): v for k, v in st_ev.prior_on.items()}, "global_state_prior": st_ev.global_prior}
    if st_ev.clf is not None:
        sp["state_probe"] = {"coef": st_ev.clf.coef_[0].tolist(), "intercept": float(st_ev.clf.intercept_[0])}
    p_gate = float(thr)  # score_video returns sigmoid probabilities; the app compares sigmoid(logit) to this
    conf = {
        k: {"mode": v.mode, "tau": v.tau, "recency_s": v.recency_s, "score": v.score, "fallback_take": v.fallback_take,
            "loc_mode": v.loc_mode}
        for k, v in cfgs.items()
        if k in ("where", "temporal", "state")
    }
    (OUT / "answer_config.json").write_text(
        json.dumps({"configs": conf, "gate_threshold": p_gate, "operating_budget": a.budget, **sp})
    )
    for name in ("embedder", "gate", "hoi_student"):
        src = OUT / f"{name}_{a.precision}.tflite"
        if src.exists():
            shutil.copy(src, OUT / f"{name}.tflite")
    print("bundle written to", OUT, "gate threshold p =", p_gate, "calibrator threshold", th)


if __name__ == "__main__":
    main()
