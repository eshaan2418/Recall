"""M1 addendum — B0 location prior + paired comparisons on B1's fixed test subsample.

* B0-prior: always answer the most frequent "where" location of the *train+val* narrations (no perception at all);
  abstains never, so its hallucination rate on never-seen objects is 100 %.
* Paired comparison on the exact queries B1 (large VLM) answered: Recall-all-frames, B2-1s, B0 and B1, with
  paired cluster-bootstrap CIs of the differences.

Output: results/m1/prior_and_paired.json
"""

from __future__ import annotations

import collections
import json

import numpy as np

from recall.bench.common import bootstrap_ci, paired_bootstrap_diff, result_header, write_result
from recall.bench.harness import queries_for
from recall.paths import RESULTS


def _load(name: str) -> dict[str, dict]:
    return {
        r["qid"]: r for r in map(json.loads, (RESULTS / "m1" / "records" / f"{name}.jsonl").read_text().splitlines())
    }


def main() -> None:
    trval = [q for s in ("train", "val") for q in queries_for(s) if q.kind == "where" and q.gt_loc_class is not None]
    prior_cls, prior_n = collections.Counter(q.gt_loc_class for q in trval).most_common(1)[0]
    prior_key = next(q.gt_loc_key for q in trval if q.gt_loc_class == prior_cls)
    test = queries_for("test")
    b0 = {}
    for q in test:
        if q.kind == "where" and q.gt_loc_class is not None:
            b0[q.qid] = {"qid": q.qid, "video_id": q.video_id, "kind": q.kind, "correct": q.gt_loc_class == prior_cls}
        elif q.kind == "absent":
            b0[q.qid] = {"qid": q.qid, "video_id": q.video_id, "kind": q.kind, "correct": False}
    loc_b0 = [r for r in b0.values() if r["kind"] == "where"]
    m, lo, hi = bootstrap_ci(
        np.array([r["correct"] for r in loc_b0], float), groups=np.array([r["video_id"] for r in loc_b0])
    )
    out: dict = {
        "B0_prior": {
            "location": prior_key,
            "train_val_share": prior_n / len(trval),
            "test_where_loc_acc": {"mean": m, "lo": lo, "hi": hi, "n": len(loc_b0)},
            "absent_hallucination": 1.0,
        },
    }
    systems = {
        "Recall-all-frames": _load("Recall-all-frames"),
        "B2-frame-CLIP-1s": _load("B2-frame-CLIP-1s"),
        "B1-VLM-K16": _load("B1-VLM-K16"),
        "B0-prior": b0,
    }
    b1 = json.loads((RESULTS / "m1" / "b1_vlm_K16.json").read_text())
    qset = set(b1["qids"])
    paired: dict = {}
    for kind, key in (("where", "loc_correct"), ("absent", "correct")):
        ids = sorted(
            q for q in qset if all(q in s for s in systems.values()) and systems["B1-VLM-K16"][q]["kind"] == kind
        )
        if kind == "where":
            ids = [q for q in ids if systems["B1-VLM-K16"][q].get("loc_correct") is not None]
        groups = np.array([systems["B1-VLM-K16"][q]["video_id"] for q in ids])
        vals = {n: np.array([float(bool(s[q].get(key, s[q].get("correct")))) for q in ids]) for n, s in systems.items()}
        res: dict = {n: dict(zip(("mean", "lo", "hi"), bootstrap_ci(v, groups=groups), strict=True)) for n, v in vals.items()}
        res["n"] = len(ids)
        for n in ("B1-VLM-K16", "B2-frame-CLIP-1s", "B0-prior"):
            d, dlo, dhi = paired_bootstrap_diff(vals["Recall-all-frames"], vals[n], groups=groups)
            res[f"Recall_minus_{n}"] = {"delta": d, "lo": dlo, "hi": dhi}
        paired[kind] = res
    out["paired_on_B1_subsample"] = paired
    write_result("m1/prior_and_paired.json", {**result_header("offline-eval", 0, {"prior_from": "train+val"}), **out})
    print(json.dumps(out, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
