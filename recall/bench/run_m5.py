"""M5 — compression & LiteRT port: accuracy cost of every step teacher → student → INT8 → INT4.

Run in the export env (needs litert/quantizer):  ``PYTHONPATH=. .venv-export/bin/python -m recall.bench.run_m5``

Parts (all on the *test* split; nothing here is tuned):

A. Embedder: zero-shot object / location top-1 on fixed test crops for the ViT-L-14 teacher, MobileCLIP2-S2
   (PyTorch) and its LiteRT exports; head-based top-1 with the M1 heads; cosine to PyTorch fp32.
B. Gate: test AP (interaction frames) of PyTorch vs LiteRT precisions.
C. HOI student vs the Faster R-CNN teacher (agreement on held-out test frames) for PyTorch / LiteRT precisions.
D. End-to-end (``--e2e``): re-run the whole perception stack with student HOI + LiteRT embedder on a fixed test
   subset (one video per test participant) and answer the same queries → accuracy deltas with paired CIs.
E. Desktop-proxy latency of each .tflite (CPU/XNNPACK, batch 1, 1 and 4 threads).

Outputs: results/m5/compression.json (+ records for D).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from recall.bench.common import bootstrap_ci, paired_bootstrap_diff, result_header, write_result
from recall.bench.harness import System, metrics, queries_for, run, subset
from recall.compress import hoi_student as hs
from recall.compress.export import OUT, TFLiteRunner, tflite_image_fn
from recall.data.loaders.epic import load_narrations
from recall.paths import CACHE, RESULTS
from recall.perception.embed import zero_shot_probs
from recall.perception.events import RegionIndex
from recall.perception.extract import extract_with, load_feats
from recall.perception.heads import Heads
from recall.perception.vocab import location_ids

MODEL = "MobileCLIP2-S2"
PRECS = ("fp32", "int8dyn", "int8", "int4w")


def e2e_videos() -> list[str]:
    """Fixed subset: the first listed test video of each test participant."""
    seen: dict[str, str] = {}
    for v in subset("test"):
        seen.setdefault(v["video_id"].split("_")[0], v["video_id"])
    return list(seen.values())


def ci(x: np.ndarray) -> dict:
    m, lo, hi = bootstrap_ci(np.asarray(x, float))
    return {"mean": m, "lo": lo, "hi": hi, "n": len(x)}


# ---------------------------------------------------------------- A. embedder


def part_a(vids: list[str], heads: Heads) -> dict:
    from recall.bench.select_embedder import collect
    from recall.perception.embed import load_embedder

    obj_imgs, obj_lab, loc_imgs, loc_lab = collect(vids, max_per_video=60)
    lid = location_ids()
    yo, yl = np.array(obj_lab), np.array([lid.index(c) for c in loc_lab])
    out: dict = {"n_obj": len(yo), "n_loc": len(yl), "videos": vids, "rows": {}}
    text = CACHE / "text"
    ref = None
    variants: list[tuple[str, object]] = [("ViT-L-14-teacher/torch", None), ("MobileCLIP2-S2/torch-fp32", None)]
    variants += [(f"MobileCLIP2-S2/litert-{p}", OUT / f"embedder_{p}.tflite") for p in PRECS]
    for name, path in variants:
        t0 = time.time()
        if path is None:
            arch, pre = ("ViT-L-14", "dfn2b") if "ViT-L" in name else ("MobileCLIP2-S2", "dfndr2b")
            emb = load_embedder(arch, pre)
            if emb.dev.type == "mps" or "torch-fp32" in name:
                emb.model.float().to("cpu")
                emb.dev = torch.device("cpu")
            oe, le = emb.images(obj_imgs, batch=32), emb.images(loc_imgs, batch=32)
            tb_o, tb_l = np.load(text / f"{arch}_obj_300.npy"), np.load(text / f"{arch}_loc_41.npy")
            load_embedder.cache_clear()
        else:
            if not Path(str(path)).exists():
                out["rows"][name] = {"missing": True}
                continue
            fn = tflite_image_fn(Path(str(path)))
            oe, le = fn(obj_imgs), fn(loc_imgs)
            tb_o, tb_l = np.load(text / f"{MODEL}_obj_300.npy"), np.load(text / f"{MODEL}_loc_41.npy")
        r: dict = {
            "obj_top1_zeroshot": ci(zero_shot_probs(oe, tb_o).argmax(1) == yo),
            "loc_top1_zeroshot": ci(zero_shot_probs(le, tb_l).argmax(1) == yl),
            "s_per_img_cpu": (time.time() - t0) / (len(yo) + len(yl)),
        }
        if "MobileCLIP2-S2" in name:
            r["obj_top1_head"] = ci(heads.obj.probs(oe).argmax(1) == yo)
            r["loc_top1_head"] = ci(np.array(heads.loc_ids)[heads.loc.probs(le).argmax(1)] == np.array(loc_lab))
            if ref is None:
                ref = (oe, le)
            else:
                r["cosine_to_torch_fp32"] = float(np.mean(np.sum(np.concatenate([oe, le]) * np.concatenate(ref), 1)))
        out["rows"][name] = r
        print("A", name, {k: (v["mean"] if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
    return out


# ---------------------------------------------------------------- B. gate


def part_b(test: dict, narr) -> dict:
    from sklearn.metrics import average_precision_score

    from recall.bench.run_m2 import gate_data
    from recall.trigger.gate import CTX, TinyGate, score_video

    gt = gate_data(test, narr)
    y = np.concatenate([d.y for d in gt])
    g = TinyGate(use_imu=True)
    g.load_state_dict(torch.load(CACHE / "gates" / "learned+imu_s0.pt", map_location="cpu"))
    g.eval()
    out: dict = {
        "torch-fp32": {
            "test_ap": float(average_precision_score(y, np.concatenate([score_video(g, d, True) for d in gt])))
        }
    }
    # LiteRT: evaluate on every 4th step (batch-1 streaming is slow on desktop); torch AP on the same steps too
    stride = 4
    sel = [(k, i) for k, d in enumerate(gt) for i in range(CTX - 1, len(d.y), stride)]
    ys = np.array([gt[k].y[i] for k, i in sel])
    sc_t = {k: score_video(g, d, True) for k, d in enumerate(gt)}
    out["torch-fp32"]["test_ap_strided"] = float(average_precision_score(ys, [sc_t[k][i] for k, i in sel]))
    for p in PRECS:
        f = OUT / f"gate_{p}.tflite"
        if not f.exists():
            continue
        run_ = TFLiteRunner(f, threads=1)
        s = []
        for k, i in sel:
            d = gt[k]
            x = d.x[i - CTX + 1 : i + 1].astype(np.float32)[None]
            imu = d.imu[i - CTX + 1 : i + 1].astype(np.float32)[None]
            s.append(float(run_(x, imu).ravel()[0]))
        out[f"litert-{p}"] = {
            "test_ap_strided": float(average_precision_score(ys, s)),
            "max_abs_logit_err": float(np.max(np.abs(np.array(s) - np.array([_logit(sc_t[k][i]) for k, i in sel])))),
        }
        print("B", p, out[f"litert-{p}"], flush=True)
    out["n_steps_strided"] = len(sel)
    return out


def _logit(p: float) -> float:
    p = min(max(p, 1e-7), 1 - 1e-7)
    return float(np.log(p / (1 - p)))


# ---------------------------------------------------------------- C. HOI student


def part_c() -> dict:
    te = np.load(hs.build_split("test", 0.25))
    x, y = te["x"], te["y"]
    model = hs.load(CACHE / "hoi_student" / "student_s0.pt")
    out: dict = {"n_frames": len(x)}
    out["torch-fp32"] = hs.agreement(hs.predict(model, x, torch.device("cpu"), 64), y)
    print("C torch", out["torch-fp32"], flush=True)
    for p in PRECS:
        f = OUT / f"hoi_student_{p}.tflite"
        if not f.exists():
            continue
        r = TFLiteRunner(f)
        pred = np.stack([r((xi.transpose(2, 0, 1)[None] / 255.0).astype(np.float32))[0] for xi in x])
        out[f"litert-{p}"] = hs.agreement(pred, y)
        print("C", p, out[f"litert-{p}"], flush=True)
    return out


# ---------------------------------------------------------------- D. end-to-end


def student_hoi_fn(path: Path):
    r = TFLiteRunner(path)
    from PIL import Image

    def fn(i: int, fidx: int, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(Image.fromarray(img).resize(hs.IN_HW[::-1], Image.Resampling.BILINEAR))
        pred = r((x.transpose(2, 0, 1)[None] / 255.0).astype(np.float32))
        H, O = hs.decode(pred)
        return H[0], O[0]

    return fn


E2E = {  # name -> (hoi student precision or None for teacher, embedder precision or None for cached torch)
    "teacher-HOI+torch-embedder": (None, None),
    "student-HOI+fp32": ("fp32", "fp32"),
    "student-HOI+int8dyn": ("int8dyn", "int8dyn"),
    "student-HOI+int8dyn+emb-int4w": ("int8dyn", "int4w"),
}


def part_d(narr, heads: Heads) -> dict:
    from recall.bench.run_m1 import all_frames, load_split, train_state_heads
    from recall.bench.run_m2 import tuned_cfgs

    vids = [v for v in e2e_videos() if (CACHE / "feats" / MODEL / f"{v}.npz").exists()]
    meta = {v["video_id"]: v for v in subset("test")}
    train = load_split("train")
    st_ev, _, _ = train_state_heads(train, narr, heads)
    del train
    cfgs = tuned_cfgs()
    qs = [q for q in queries_for("test") if q.video_id in vids]
    recs_by: dict[str, list[dict]] = {}
    out: dict = {"videos": vids, "n_queries": len(qs), "variants": {}}
    for name, (hp, ep) in E2E.items():
        feats_dir = CACHE / "feats" / MODEL if hp is None else CACHE / "feats" / f"m5_{name}"
        cache: dict = {}
        t0 = time.time()
        for v in vids:
            p = feats_dir / f"{v}.npz"
            if not p.exists():
                assert hp is not None and ep is not None
                feats_dir.mkdir(parents=True, exist_ok=True)
                extract_with(
                    v,
                    meta[v]["fps"],
                    meta[v]["duration"],
                    p,
                    tflite_image_fn(OUT / f"embedder_{ep}.tflite"),
                    512,
                    student_hoi_fn(OUT / f"hoi_student_{hp}.tflite"),
                    {"m5": name},
                )
                print("D extracted", name, v, round(time.time() - t0), "s", flush=True)
            f = dict(np.load(p))
            cache[v] = (f, RegionIndex(f))
        sys = System(name, all_frames, cfg=cfgs["where"], cfg_by_kind=cfgs, state_head=st_ev)
        recs, cost = _run_on(sys, heads, qs, cache)
        recs_by[name] = recs
        out["variants"][name] = {"metrics": metrics(recs), "cost": cost}
        print("D", name, {k: v.get("mean") for k, v in out["variants"][name]["metrics"].items() if isinstance(v, dict)})
        (RESULTS / "m5" / "records").mkdir(parents=True, exist_ok=True)
        with open(RESULTS / "m5" / "records" / f"{name}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, default=float) + "\n")
    base = {r["qid"]: r for r in recs_by["teacher-HOI+torch-embedder"]}
    for name, recs in recs_by.items():
        if name == "teacher-HOI+torch-embedder":
            continue
        diffs = {}
        for kind in ("where", "temporal", "state", "absent"):
            rr = [r for r in recs if r["kind"] == kind and r["qid"] in base]
            if rr:
                a = np.array([float(r["correct"]) for r in rr])
                b = np.array([float(base[r["qid"]]["correct"]) for r in rr])
                m, lo, hi = paired_bootstrap_diff(a, b, groups=np.array([r["video_id"] for r in rr]))
                diffs[kind] = {"delta": m, "lo": lo, "hi": hi, "n": len(rr)}
        out["variants"][name]["delta_vs_teacher"] = diffs
    return out


def _run_on(sys: System, heads: Heads, qs: list, cache: dict) -> tuple[list[dict], dict]:
    return run(sys, "test", heads, MODEL, qs, cache, videos=list(cache))


# ---------------------------------------------------------------- E. latency


def part_e(threads: tuple[int, ...] = (1, 4), n: int = 30) -> dict:
    out: dict = {}
    for f in sorted(OUT.glob("*_*.tflite")):
        r = TFLiteRunner(f, threads=threads[0])
        shapes = [tuple(r.runner.get_input_details()[k]["shape"]) for k in r.inputs]
        xs = [np.random.default_rng(0).random(s, dtype=np.float32) for s in shapes]
        row: dict[str, float] = {"bytes": f.stat().st_size}
        for th in threads:
            r = TFLiteRunner(f, threads=th)
            for _ in range(3):
                r(*xs)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                r(*xs)
                ts.append((time.perf_counter() - t0) * 1e3)
            row[f"cpu{th}_ms_p50"] = float(np.median(ts))
            row[f"cpu{th}_ms_p95"] = float(np.percentile(ts, 95))
        out[f.stem] = row
        print("E", f.stem, row, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="abce")
    ap.add_argument("--e2e", action="store_true")
    a = ap.parse_args()
    narr = load_narrations()
    heads = Heads.load(f"{MODEL}_seed0")
    prev = RESULTS / "m5" / "compression.json"
    res: dict = json.loads(prev.read_text()) if prev.exists() else {}
    res = {k: v for k, v in res.items() if k in ("A_embedder", "B_gate", "C_hoi_student", "D_end_to_end", "E_latency")}
    if "a" in a.parts:
        res["A_embedder"] = part_a(e2e_videos(), heads)
    if "b" in a.parts:
        test = {
            v: (load_feats(v, (MODEL, "")), None)
            for v in [x["video_id"] for x in subset("test")]
            if (CACHE / "feats" / MODEL / f"{v}.npz").exists()
        }
        res["B_gate"] = part_b(test, narr)
    if "c" in a.parts:
        res["C_hoi_student"] = part_c()
        log = CACHE / "hoi_student" / "student_s0.json"
        if log.exists():
            res["C_hoi_student"]["training"] = json.loads(log.read_text())
    if a.e2e or "d" in a.parts:
        res["D_end_to_end"] = part_d(narr, heads)
    if "e" in a.parts:
        res["E_latency"] = part_e()
    mf = OUT / "MANIFEST.json"
    res["manifest"] = json.loads(mf.read_text()) if mf.exists() else None
    write_result(
        "m5/compression.json",
        {
            **result_header("desktop-proxy", 0, {"parts": a.parts, "e2e": a.e2e, "precisions": PRECS}),
            "note": "Latency = desktop CPU (Apple M5 Pro, XNNPACK), shared/loaded machine; NOT phone latency.",
            **res,
        },
    )


if __name__ == "__main__":
    main()
