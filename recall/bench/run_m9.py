"""M9 — personal-data evaluation: Recall vs baselines on annotated first-person recordings.

    python -m recall.bench.run_m9                 # real data: data/personal/raw/*.mp4 + annotations
    python -m recall.bench.run_m9 --standin 3     # no personal data: EPIC test clips re-encoded to MP4 and
                                                  # auto-annotated from narrations, run through the *same*
                                                  # personal code path (tooling check; labelled "stand-in")

Personal videos have no released hand–object teacher, so the pipeline is the deployable one: HOI *student* +
MobileCLIP2-S2 embedder + the image-only learned gate (no IMU). Systems: Recall (learned trigger @ budget),
Recall all-frames, B2 frame+CLIP 2 s, B3 manual tagging (q=0.5), B4 oracle trigger (annotated moments).

Correctness: ``evidence_ok`` = the returned evidence moment lies within [put − 5 s, put + 10 s] (the user sees
the thumbnail) — vocabulary-free, so it works for free-text places. ``loc_correct`` additionally maps the free-text
place to the 41-class location vocabulary (only when it resolves) and checks the predicted class.

Outputs: results/m9/personal.json (or results/m9/standin.json); failure gallery JPEGs (redacted) go to
data/personal/cache/gallery/ — NEVER into results/ or git.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import torch

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, metrics, run, subset
from recall.compress import hoi_student as hs
from recall.data.loaders.personal import PERSONAL, VideoFileReader, build_queries, load_annotations
from recall.data.queries import LOCATION_KEYS, PUT_VERBS, VERB_TAKE, Query, readable
from recall.paths import CACHE
from recall.perception.embed import load_embedder
from recall.perception.events import RegionIndex
from recall.perception.extract import extract_with
from recall.perception.heads import Heads
from recall.qa.resolve import resolve
from recall.trigger.gate import GateData, TinyGate, frame_inputs, score_video
from recall.trigger.policies import threshold_for_budget, threshold_policy

MODEL = "MobileCLIP2-S2"
BURST = 4


# ---------------------------------------------------------------- stand-in data


def make_standin(n: int) -> list[str]:
    """Re-encode n EPIC test videos (2 fps-sufficient 10 fps MP4) + narration-derived annotations."""
    from recall.data.loaders.epic import FrameReader, load_narrations

    narr = load_narrations()
    raw, ann = PERSONAL / "raw", PERSONAL / "annotations"
    raw.mkdir(parents=True, exist_ok=True)
    ann.mkdir(parents=True, exist_ok=True)
    seen: dict[str, dict] = {}
    for v in subset("test"):
        seen.setdefault(v["video_id"].split("_")[0], v)
    names = []
    for v in list(seen.values())[:n]:
        name = f"standin_{v['video_id']}"
        mp4 = raw / f"{name}.mp4"
        if not mp4.exists():
            rd = FrameReader(v["video_id"])
            step = v["fps"] / 10
            p = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "456x256",
                    "-r",
                    "10",
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4),
                ],
                stdin=subprocess.PIPE,
            )
            assert p.stdin is not None
            for k in range(int(len(rd) / step)):
                p.stdin.write(rd.frame(int(k * step) + 1).tobytes())
            p.stdin.close()
            p.wait()
            rd.close()
        d = narr[narr.video_id == v["video_id"]]
        locs = set(LOCATION_KEYS)
        from recall.perception.vocab import noun_table

        id2key = dict(zip(noun_table().id, noun_table().key, strict=True))
        lines = []
        for r in d.itertuples():
            if r.verb_class in PUT_VERBS:
                place = next((id2key[c] for c in r.all_noun_classes[1:] if id2key.get(c) in locs), None)
                lines.append(
                    {
                        "video": name,
                        "t": float(r.stop_s),
                        "kind": "put",
                        "object": readable(id2key[r.noun_class]),
                        "place": readable(place) if place else None,
                    }
                )
            elif r.verb_class == VERB_TAKE:
                lines.append(
                    {
                        "video": name,
                        "t": float(r.start_s),
                        "kind": "take",
                        "object": readable(id2key[r.noun_class]),
                        "place": None,
                    }
                )
        (ann / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        names.append(name)
    return names


# ---------------------------------------------------------------- features


def personal_feats(name: str, student: hs.HOIStudent) -> dict:
    out = PERSONAL / "cache" / "feats" / f"{name}.npz"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        rd = VideoFileReader(PERSONAL / "raw" / f"{name}.mp4")
        emb = load_embedder(MODEL, "dfndr2b")
        dev = emb.dev
        st = student.to(dev).eval()
        from PIL import Image

        def hoi_fn(i: int, fidx: int, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x = np.asarray(Image.fromarray(img).resize(hs.IN_HW[::-1], Image.Resampling.BILINEAR))[None]
            H, O = hs.decode(hs.predict(st, x, dev))
            return H[0], O[0]

        extract_with(name, rd.fps, rd.duration, out, emb.images, emb.dim, hoi_fn, {"personal": True}, reader=rd)
    f = dict(np.load(out))
    return f


# ---------------------------------------------------------------- failure analysis


def failure_type(r: dict, view_t: np.ndarray, q: Query) -> str | None:
    if r.get("correct"):
        return None
    if q.kind == "absent":
        return "hallucinated answer for never-seen object"
    assert q.gt_t0 is not None and q.gt_t1 is not None
    near = np.abs(view_t - (q.gt_t0 + q.gt_t1) / 2) <= 10.0
    if not near.any():
        return "interaction not captured (trigger / hand-object miss, occlusion, pocket/bag)"
    if not r.get("answered"):
        return "captured but abstained (low object score)"
    if r.get("time_ok"):
        return "right moment, wrong place label"
    return "picked another event (similar object / distractor)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", type=int, default=0, help="use N EPIC test clips as stand-in personal data")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from recall.bench.run_m1 import make_uniform
    from recall.bench.run_m2 import tuned_cfgs

    names = (
        make_standin(a.standin)
        if a.standin
        else sorted(p.stem for p in (PERSONAL / "raw").glob("*.mp4") if not p.stem.startswith("standin_"))
    )
    names = [n for n in names if (PERSONAL / "annotations" / f"{n}.jsonl").exists()]
    if not names:
        raise SystemExit("no annotated personal recordings (see HUMAN_TODO.md); use --standin N for a tooling check")
    heads = Heads.load(f"{MODEL}_seed0")
    cfgs = tuned_cfgs()
    student = hs.load(CACHE / "hoi_student" / "student_s0.pt")
    emb = load_embedder(MODEL, "dfndr2b")
    obj_bank = np.load(CACHE / "text" / f"{MODEL}_obj_300.npy")
    from recall.perception.vocab import noun_table

    k2i = dict(zip(noun_table().key, noun_table().id, strict=True))
    loc_key_by_id = {k2i[k]: k for k in LOCATION_KEYS if k in k2i}

    # gate: image-only variant (personal recordings have no IMU stream)
    gate = TinyGate(use_imu=False)
    gate.load_state_dict(torch.load(CACHE / "gates" / f"learned_s{a.seed}.pt", map_location="cpu"))
    from recall.bench.run_m1 import load_split
    from recall.bench.run_m2 import gate_data
    from recall.data.loaders.epic import load_narrations

    val = load_split("val")
    thr = threshold_for_budget(
        np.concatenate([score_video(gate, d, False) for d in gate_data(val, load_narrations())]), a.budget, BURST
    )
    del val

    feats_cache, qs_all, anns_by = {}, [], {}
    for n in names:
        f = personal_feats(n, student)
        feats_cache[n] = (f, RegionIndex(f))
        anns = load_annotations(PERSONAL / "annotations" / f"{n}.jsonl")
        anns_by[n] = anns
        objs = sorted({x.object for x in anns})
        oc = {o: c for o in objs if (c := resolve(o, emb, obj_bank)[0]) is not None}
        places = sorted({x.place for x in anns if x.place})
        lc: dict[str, int] = {}
        for p in places:
            c, _ = resolve(p)
            if c is None:
                c, _ = resolve(p.split()[-1])
            if c is not None and c in loc_key_by_id:
                lc[p] = c
        annotated = set(oc.values())
        absent = [
            readable(k)
            for k in ("scissors", "battery", "camera", "remote", "wallet", "glove", "candle")
            if k in k2i and k2i[k] not in annotated
        ][:5]
        for o in absent:
            oc[o] = k2i.get(o, -1)
        qs = build_queries(anns, len(f["t"]) / 2.0, oc, lc, absent)
        qs_all += [q for q in qs if q.obj_class >= 0]

    gsc = {
        n: score_video(
            gate,
            GateData(frame_inputs(f["thumb"]), np.zeros((len(f["t"]), 7), np.float32), np.zeros(len(f["t"])), n),
            False,
        )
        for n, (f, _) in feats_cache.items()
    }

    def oracle_pol(vid: str, f: dict) -> np.ndarray:
        m = np.zeros(len(f["t"]), bool)
        for x in anns_by[vid]:
            m |= np.abs(f["t"] - x.t) <= 2.0
        return m

    systems = [
        System(
            f"Recall-learned@{a.budget}",
            lambda vid, f: threshold_policy(gsc[vid], thr, BURST),
            cfg=cfgs["where"],
            cfg_by_kind=cfgs,
        ),
        System("Recall-all-frames", lambda vid, f: np.ones(len(f["t"]), bool), cfg=cfgs["where"], cfg_by_kind=cfgs),
        System("B4-oracle-trigger", oracle_pol, cfg=cfgs["where"], cfg_by_kind=cfgs),
        System("B2-frame-CLIP-2s", make_uniform(4), memory="frames", cfg=cfgs["where"], cfg_by_kind=cfgs),
    ]
    out: dict = {
        "videos": names,
        "standin": bool(a.standin),
        "n_queries": {k: sum(q.kind == k for q in qs_all) for k in ("where", "absent")},
        "hours": sum(len(f["t"]) for f, _ in feats_cache.values()) / 7200,
        "systems": {},
    }
    qmap = {q.qid: q for q in qs_all}
    for sys in systems:
        recs, cost = run(sys, "custom", heads, MODEL, qs_all, feats_cache, videos=names)
        from recall.bench.harness import build_view

        fails: dict[str, int] = {}
        for vid in names:
            view, _ = build_view(sys, vid, feats_cache[vid][0], heads, feats_cache[vid][1])
            for r in recs:
                if r["video_id"] == vid:
                    r["evidence_ok"] = r.get("time_ok", r.get("correct"))
                    ft = failure_type(r, view.t, qmap[r["qid"]])
                    if ft:
                        fails[ft] = fails.get(ft, 0) + 1
        m = metrics(recs)
        ev = [r for r in recs if r["kind"] == "where"]
        out["systems"][sys.name] = {
            "metrics": m,
            "cost": cost,
            "failure_types": fails,
            "evidence_ok": float(np.mean([bool(r["time_ok"]) for r in ev])) if ev else None,
        }
        print(
            sys.name,
            m["where_time_acc"].get("mean"),
            m["where_loc_acc"].get("mean"),
            m["absent_abstain"].get("mean"),
            fails,
        )
    # B3 manual tagging from annotations (q = 0.5)
    rng = np.random.default_rng(a.seed)
    b3 = []
    for q in qs_all:
        tags = [
            x
            for x in anns_by[q.video_id]
            if x.kind == "put" and x.t < q.t_query and rng.random() < 0.5 and q.obj_key == x.object
        ]
        ok = q.kind == "absent" or (bool(tags) and q.gt_t0 is not None and abs(tags[-1].t - (q.gt_t0 + 3.0)) < 1e-6)
        b3.append(
            {
                "qid": q.qid,
                "video_id": q.video_id,
                "kind": q.kind,
                "correct": ok,
                "time_ok": ok,
                "answered": bool(tags) if q.kind != "absent" else False,
                "loc_correct": ok if q.gt_loc_class is not None else None,
            }
        )
    out["systems"]["B3-manual-q0.5"] = {"metrics": metrics(b3)}
    write_result(
        f"m9/{'standin' if a.standin else 'personal'}.json",
        {
            **result_header("offline-eval", a.seed, {"budget": a.budget, "standin": a.standin}),
            "WARNING": "stand-in: EPIC kitchen clips through the personal pipeline, NOT personal data"
            if a.standin
            else None,
            **out,
        },
    )


if __name__ == "__main__":
    main()
