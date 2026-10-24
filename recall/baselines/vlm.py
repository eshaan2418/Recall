"""B1 — large VLM given uniformly sampled raw frames + the question (the "send video to a big model" design).

No cloud key exists, so B1 is a large *open* VLM run locally (Qwen3-VL-8B-Instruct, 4-bit MLX). The model
gets K frames sampled uniformly from [0, t_query] (456×256, timestamps in the prompt) and must answer from
a closed location list (same vocabulary as Recall) or say unknown. Cloud mode (frontier API) is supported
only behind ``--allow-cloud`` and only for public EPIC frames; it is not used for reported numbers.

Evaluated on a fixed, seeded test subsample (cost: ~K images per query). Output records use the same
scoring as every other system (``bench.harness.score``).
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
from PIL import Image

from recall.bench.common import result_header, write_result
from recall.bench.harness import metrics, queries_for, score, subset
from recall.data.loaders.epic import FrameReader
from recall.data.queries import Query, readable
from recall.paths import RESULTS
from recall.perception.vocab import location_ids, noun_table
from recall.qa.answer import Answer

MODEL_ID = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def _fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def sample_frames(vid: str, fps: float, t_q: float, k: int) -> tuple[list[Image.Image], np.ndarray]:
    ts = np.linspace(0, max(0.0, t_q - 0.5), k)
    rd = FrameReader(vid)
    ims = [Image.fromarray(rd.frame(round(t * fps) + 1)) for t in ts]
    rd.close()
    return ims, ts


def build_prompt(q: Query, ts: np.ndarray, loc_names: list[str]) -> str:
    frames = ", ".join(f"frame {i} @ {_fmt_t(t)}" for i, t in enumerate(ts))
    head = (
        "You are a memory assistant. The images are frames from my first-person (head-mounted) video, in "
        f"chronological order: {frames}. Only use what is visible in the frames.\n"
    )
    if q.kind in ("where", "absent"):
        return head + (
            f"Question: Where is the {readable(q.obj_key)} now (where did I last put it)?\n"
            f"Choose exactly one location from this list: {', '.join(loc_names)}. If the object never appears, "
            'answer "unknown".\nReply with JSON only: {"location": "<location or unknown>", "frame": <index of the '
            "frame that shows it last, or -1>}"
        )
    if q.kind == "temporal":
        return head + (
            f"Question: {q.text}\nReply with JSON only: "
            '{"frame": <index of the frame closest to when it happened, or -1 if not visible>}'
        )
    return head + (
        f"Question: {q.text} Is the {readable(q.obj_key)} on or off now?\nReply with JSON only: "
        '{"state": "on" | "off" | "unknown", "frame": <index of the most relevant frame, or -1>}'
    )


def parse_reply(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def to_answer(q: Query, rep: dict, ts: np.ndarray, loc_names: list[str]) -> Answer:
    kind = "where" if q.kind == "absent" else q.kind
    fr = rep.get("frame", -1)
    fr = int(fr) if isinstance(fr, int | float) else -1
    t = float(ts[fr]) if 0 <= fr < len(ts) else None
    if kind == "where":
        loc = str(rep.get("location", "unknown")).strip().lower()
        if loc not in loc_names:
            return Answer(kind, True)
        return Answer(kind, False, None, t if t is not None else float(ts[-1]), loc_names.index(loc))
    if kind == "temporal":
        return Answer(kind, t is None, None, t)
    st = str(rep.get("state", "unknown")).lower()
    if st not in ("on", "off"):
        return Answer(kind, True)
    return Answer(kind, False, None, t if t is not None else float(ts[-1]), state=st)


def pick_subsample(qs: list[Query], n_per_kind: dict[str, int], seed: int = 0) -> list[Query]:
    rng = np.random.default_rng(seed)
    out = []
    for k, n in n_per_kind.items():
        pool = [q for q in qs if q.kind == k]
        if k == "where":  # prioritise located where-queries (location accuracy is the headline metric)
            pool = [q for q in pool if q.gt_loc_class is not None]
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        out += [pool[i] for i in sorted(idx)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-where", type=int, default=120)
    ap.add_argument("--n-other", type=int, default=40)
    ap.add_argument("--allow-cloud", action="store_true", help="not implemented for reported results")
    ap.add_argument("--max-tokens", type=int, default=60)
    a = ap.parse_args()
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor = load(MODEL_ID)
    nt = noun_table()
    loc_names = [readable(nt.key[i]) for i in location_ids()]
    info = {v["video_id"]: v for v in subset("test")}
    qs = pick_subsample(
        queries_for("test"), {"where": a.n_where, "absent": a.n_other, "temporal": a.n_other, "state": a.n_other}
    )
    out_path = RESULTS / "m1" / "records" / f"B1-VLM-K{a.k}.jsonl"
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            done[r["qid"]] = r
    recs = list(done.values())
    lat = []
    with open(out_path, "a") as fh:
        for q in qs:
            if q.qid in done:
                continue
            ims, ts = sample_frames(q.video_id, info[q.video_id]["fps"], q.t_query, a.k)
            prompt = apply_chat_template(processor, model.config, build_prompt(q, ts, loc_names), num_images=len(ims))
            t0 = time.time()
            res = generate(model, processor, prompt, ims, max_tokens=a.max_tokens, temperature=0.0, verbose=False)  # type: ignore[arg-type]
            lat.append(time.time() - t0)
            txt = res.text if hasattr(res, "text") else str(res)
            ans = to_answer(q, parse_reply(txt), ts, loc_names)
            r = score(q, ans, location_ids())
            r.update(system=f"B1-VLM-K{a.k}", raw=txt[:300], latency_s=lat[-1])
            fh.write(json.dumps(r, default=float) + "\n")
            fh.flush()
            recs.append(r)
            print(q.qid, q.kind, r.get("correct"), f"{lat[-1]:.1f}s", txt[:80].replace("\n", " "))
    lat_all = [r.get("latency_s", np.nan) for r in recs]
    write_result(
        f"m1/b1_vlm_K{a.k}.json",
        {
            **result_header(
                "desktop-proxy", 0, {"model": MODEL_ID, "k": a.k, "n_where": a.n_where, "n_other": a.n_other}
            ),
            "metrics": metrics(recs),
            "n": len(recs),
            "median_latency_s": float(np.nanmedian(lat_all)),
            "qids": [q.qid for q in qs],
            "note": "Large open VLM run locally (no cloud key); not a frontier cloud model. Latency under shared-machine load.",
        },
    )


if __name__ == "__main__":
    main()
