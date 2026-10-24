"""M3 — compact episodic memory: accuracy vs storage, retention, query latency.

Memory variants (all at the same trigger operating point):
  * events-fp16      : event rows + fp16 object & location embeddings + 1 redacted-size JPEG thumbnail
  * events-int8      : same, int8 embeddings
  * events-labels    : event rows with top-5 object/location labels only (no embeddings, no thumbnail)
  * frames-emb       : −event graph ablation — one fp16 frame embedding per processed frame (no events)
  * keyframes-2s     : B2-style store, 1 JPEG keyframe (456×256) + embedding every 2 s

Storage is measured by writing real SQLite files (thumbnails JPEG-encoded from the actual frames) for the
test videos. Weekly projection assumes 16 recorded hours/day × 7 days (stated assumption, not a measurement).
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import replace

import numpy as np
from PIL import Image

from recall.bench.common import result_header, write_result
from recall.bench.harness import System, frame_events, metrics, queries_for, run, score, subset
from recall.bench.run_m1 import MODEL, all_frames, load_split, train_state_heads
from recall.bench.run_m2 import BURST, gate_data, threshold_for_budget, tuned_cfgs
from recall.data.loaders.epic import FrameReader, load_narrations
from recall.memory.schema import Event
from recall.memory.store import MemoryStore, StoreConfig
from recall.paths import CACHE
from recall.perception.events import extract_events
from recall.perception.heads import Heads
from recall.qa.answer import MemoryView, answer
from recall.qa.parse import ParsedQuery
from recall.trigger.gate import TinyGate, score_video
from recall.trigger.policies import threshold_policy, uniform

HOURS_PER_DAY = 16
THUMB_WH = (128, 72)


def thumb_bytes(img: np.ndarray, box: tuple | None, wh: tuple[int, int] = THUMB_WH, q: int = 70) -> bytes:
    im = Image.fromarray(img)
    if box is not None:
        W, H = im.size
        l, t, r, b = box
        cx, cy = (l + r) / 2 * W, (t + b) / 2 * H
        w = max((r - l) * W * 3, W * 0.35)
        h = w * wh[1] / wh[0]
        im = im.crop(
            (int(max(0, cx - w / 2)), int(max(0, cy - h / 2)), int(min(W, cx + w / 2)), int(min(H, cy + h / 2)))
        )
    im = im.resize(wh)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


def fill_topk(events: list[Event], heads: Heads, k: int = 5) -> None:
    if not events:
        return
    O = np.stack([e.obj_emb for e in events])
    L = np.stack([e.loc_emb for e in events])
    po, pl = heads.obj.probs(O), heads.loc.probs(L)
    for e, a, b in zip(events, po, pl, strict=True):
        io_, il = np.argsort(-a)[:k], np.argsort(-b)[:k]
        e.obj_top = [(int(i), float(a[i])) for i in io_]
        e.loc_top = [(int(i), float(b[i])) for i in il]


def labels_only_view(events: list[Event], n_obj: int, n_loc: int) -> MemoryView:
    """MemoryView whose class probabilities come only from stored top-k labels (rest → tiny mass)."""
    v = MemoryView.build([], _DummyHead(n_obj), _DummyHead(n_loc))  # type: ignore[arg-type]
    if not events:
        return v
    N = len(events)
    ol = np.full((N, n_obj), np.log(1e-6))
    ll = np.full((N, n_loc), np.log(1e-6))
    for i, e in enumerate(events):
        for c, p in e.obj_top:
            ol[i, c] = np.log(max(p, 1e-6))
        for c, p in e.loc_top:
            ll[i, c] = np.log(max(p, 1e-6))
    z = np.zeros((N, 1), np.float32)
    return MemoryView(np.array([e.t for e in events]), np.array([e.kind for e in events]),
                      np.array([e.n_obs for e in events], float), np.array([e.hand_score for e in events], float),
                      np.array([e.release_observed for e in events], float), ol, ll, z, z, z, events)  # fmt: skip


class _DummyHead:
    def __init__(self, n: int):
        self.text_bank = np.zeros((n, 1), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.2, help="learned-trigger operating point (fraction of frames)")
    ap.add_argument("--gate", default="learned+imu_s0")
    a = ap.parse_args()
    narr = load_narrations()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    heads = Heads.load(f"{MODEL}_seed0")
    st_ev, _, _ = train_state_heads(train, narr, heads)
    cfgs = tuned_cfgs()
    qs = [q for q in queries_for("test") if q.video_id in test]

    import torch

    use_imu = "+imu" in a.gate
    gate = TinyGate(use_imu=use_imu)
    gate.load_state_dict(torch.load(CACHE / "gates" / f"{a.gate}.pt"))
    gva = gate_data(val, narr)
    thr = threshold_for_budget(np.concatenate([score_video(gate, d, use_imu) for d in gva]), a.budget, BURST)
    gte = {d.vid: score_video(gate, d, use_imu) for d in gate_data(test, narr)}
    policies = {
        "all-frames": all_frames,
        f"learned@{a.budget}": lambda vid, f: threshold_policy(gte[vid], thr, BURST),
    }

    fps_of = {v["video_id"]: float(v["fps"]) for v in subset("test")}
    out_dir = CACHE / "m3_stores"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    hours = sum(len(f["t"]) for f, _ in test.values()) / 7200
    for pname, pol in policies.items():
        base = System(pname, pol, cfg=cfgs["where"], cfg_by_kind=cfgs, state_head=st_ev)
        # --- per-variant storage, measured with real SQLite files ---
        sizes = {k: 0 for k in ("events-fp16", "events-int8", "events-labels", "frames-emb", "keyframes-2s")}
        n_events = n_merged = 0
        recs_labels = []
        lat_ms = []
        for vid, (f, reg) in test.items():
            mask = pol(vid, f)
            evs = extract_events(f, mask, vid, base.extractor, reg)
            fill_topk(evs, heads)
            rd = FrameReader(vid)
            fps = fps_of[vid]
            for e in evs:
                e.thumb = thumb_bytes(rd.frame(round(e.t * fps) + 1), e.box)
            for name, cfg in (
                ("events-fp16", StoreConfig()),
                ("events-int8", StoreConfig(emb_dtype="int8")),
                ("events-labels", StoreConfig(store_obj_emb=False, store_loc_emb=False, store_thumbs=False)),
            ):
                p = out_dir / f"{pname}_{name}_{vid}.db"
                p.unlink(missing_ok=True)
                st = MemoryStore(p, cfg)
                st.add_all(evs)
                if name == "events-fp16":
                    n_events += st.n_events()
                    n_merged += len(evs) - st.n_events()
                    # query latency: SQL fetch + heads + answer, per test query of this video
                    for q in [q for q in qs if q.video_id == vid]:
                        t0 = time.perf_counter()
                        rows_ = st.events(before_t=q.t_query)
                        ev2 = [
                            Event(
                                vid,
                                r["t"],
                                r["kind"],
                                r["side"],
                                0,
                                r["obj_emb"],
                                r["loc_emb"],
                                r["obj_emb"],
                                n_obs=r["n_obs"],
                                hand_score=r["hand_score"],
                                release_observed=bool(r["release_obs"]),
                            )
                            for r in rows_
                        ]
                        view = MemoryView.build(ev2, heads.obj, heads.loc)
                        kind = "where" if q.kind == "absent" else q.kind
                        answer(view, ParsedQuery(kind, q.obj_key), q.obj_class, q.t_query, cfgs[q.kind], None)
                        lat_ms.append((time.perf_counter() - t0) * 1e3)
                sizes[name] += st.size_bytes()
                st.db.close()
            # labels-only accuracy
            v_lab = labels_only_view(evs, heads.obj.text_bank.shape[0], heads.loc.text_bank.shape[0])
            for q in [q for q in qs if q.video_id == vid]:
                kind = "where" if q.kind == "absent" else q.kind
                c = cfgs[q.kind]
                ans = answer(v_lab, ParsedQuery(kind, q.obj_key, None), q.obj_class, q.t_query, replace(c), None)
                recs_labels.append(score(q, ans, heads.loc_ids))
            # frames-emb (−event graph) storage: one fp16 embedding row per processed frame
            fe = frame_events(f, mask, vid)
            st = MemoryStore(out_dir / f"{pname}_frames_{vid}.db", StoreConfig(store_loc_emb=False, store_thumbs=False))
            st.add_all(fe)
            sizes["frames-emb"] += st.size_bytes()
            st.db.close()
            # keyframes every 2 s with full-res JPEG + embedding
            km = uniform(len(f["t"]), 4)
            kb = (
                sum(
                    len(thumb_bytes(rd.frame(round(f["t"][i] * fps) + 1), None, (456, 256), 75)) + 1024
                    for i in np.flatnonzero(km)[::10]
                )
                * 10
            )
            sizes["keyframes-2s"] += kb
            rd.close()
        recs_full, cost = run(base, "test", heads, MODEL, qs, test)
        recs_frames, _ = run(replace(base, memory="frames"), "test", heads, MODEL, qs, test)
        acc = {
            "events-fp16": metrics(recs_full),
            "events-labels": metrics(recs_labels),
            "frames-emb": metrics(recs_frames),
        }
        for name, b in sizes.items():
            rows.append(
                {
                    "policy": pname,
                    "variant": name,
                    "bytes_total": b,
                    "bytes_per_hour": b / hours,
                    "projected_GB_per_week": b / hours * HOURS_PER_DAY * 7 / 1e9,
                    "metrics": acc.get(name),
                    "heavy_per_hour": cost["heavy_per_hour"],
                }
            )
        rows.append(
            {
                "policy": pname,
                "variant": "_summary",
                "events_per_hour": n_events / hours,
                "merged_fraction": n_merged / max(1, n_events + n_merged),
                "query_latency_ms_p50": float(np.percentile(lat_ms, 50)),
                "query_latency_ms_p95": float(np.percentile(lat_ms, 95)),
                "n_latency": len(lat_ms),
            }
        )
        for r in rows[-6:]:
            print(
                {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "metrics"},
                (r.get("metrics") or {}).get("where_acc", {}).get("mean"),
            )
    write_result(
        "m3/memory_storage.json",
        {
            **result_header(
                "desktop-proxy",
                0,
                {"budget": a.budget, "gate": a.gate, "hours_per_day_assumption": HOURS_PER_DAY, "thumb_wh": THUMB_WH},
            ),
            "test_hours": hours,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
