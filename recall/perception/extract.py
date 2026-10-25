"""Offline feature cache: everything the heavy and cheap models would output, per 2 fps frame.

Heavy-perception outputs are cached for *every* frame so that memory-trigger
policies can be simulated cheaply: a policy only decides which frames' heavy
outputs it is allowed to use (and is charged for them).

Per video ``data/cache/feats/<model>/<video_id>.npz``:

* ``t``           [T]       timestamps (s) at ``RATE`` fps
* ``thumb``       [T,36,64,3] uint8 low-res frames (input of the always-on gate)
* ``hands``       [T,2,7]   per side (L,R): present, score, state, l, t, r, b
* ``hobj``        [T,2,5]   contacted-object box per side: present, l, t, r, b
* ``frame_emb``   [T,D]     full-frame embedding (float16)
* ``reg_*``       regions: crops around contacted objects (``obj``) at the
                  current frame, and the surroundings of the previous step's
                  object box (``pctx``, used to see where it was released).
                  ``ctx``/``pobj`` kinds exist in the schema but are not
                  extracted (compute budget; see DECISIONS 2026-10-24).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from tqdm import tqdm

from recall.data.loaders.epic import FrameReader, tar_path
from recall.data.loaders.hoi import PORTABLE, STATIONARY, FrameHOI, load_hoi
from recall.paths import CACHE, EPIC_RAW, REPO
from recall.perception.embed import DEFAULT_MODEL, load_embedder

RATE = 2.0
THUMB_HW = (36, 64)
REGION_KINDS = ["obj", "ctx", "pobj", "pctx"]


def expand_box(b: np.ndarray | tuple, f: float, min_size: float = 0.06) -> tuple[float, float, float, float]:
    l, t, r, bt = (float(x) for x in b)
    cx, cy = (l + r) / 2, (t + bt) / 2
    w, h = max((r - l) * f, min_size), max((bt - t) * f, min_size)
    return max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)


def crop(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    H, W = img.shape[:2]
    l, t, r, b = box
    x0, y0 = int(l * W), int(t * H)
    x1, y1 = max(x0 + 2, int(np.ceil(r * W))), max(y0 + 2, int(np.ceil(b * H)))
    return img[y0:y1, x0:x1]


def hands_array(fr: FrameHOI, hand_thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Best hand per side → [2,7]; contacted object per side → [2,5]."""
    H = np.zeros((2, 7), np.float32)
    O = np.zeros((2, 5), np.float32)
    for side in (0, 1):
        hs = [h for h in fr.hands if h.side == side and h.score >= hand_thresh]
        if not hs:
            continue
        h = max(hs, key=lambda x: x.score)
        H[side] = [1, h.score, h.state, *h.box]
        o = fr.contacted_object(h)
        if o is not None:
            O[side] = [1, *o.box]
    return H, O


def extract_video(
    video_id: str, fps: float, duration: float, model: tuple[str, str] = DEFAULT_MODEL, out_dir: Path | None = None
) -> Path:
    out_dir = out_dir or CACHE / "feats" / model[0]
    out = out_dir / f"{video_id}.npz"
    if out.exists():
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # another worker owns this video
        os.close(fd)
    except FileExistsError:
        return out
    try:
        return _extract(video_id, fps, duration, model, out)
    finally:
        lock.unlink(missing_ok=True)


def _extract(video_id: str, fps: float, duration: float, model: tuple[str, str], out: Path) -> Path:
    emb = load_embedder(*model)
    hoi = load_hoi(EPIC_RAW / "hoi" / f"{video_id}.pkl")

    def teacher(i: int, fidx: int, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return hands_array(hoi[min(fidx - 1, len(hoi) - 1)])

    return extract_with(video_id, fps, duration, out, emb.images, emb.dim, teacher, list(model))


class FrameSource(Protocol):
    def __len__(self) -> int: ...
    def frame(self, idx: int) -> np.ndarray: ...
    def close(self) -> None: ...


HoiFn = Callable[[int, int, np.ndarray], tuple[np.ndarray, np.ndarray]]


def extract_with(
    video_id: str,
    fps: float,
    duration: float,
    out: Path,
    images_fn: Callable[..., np.ndarray],
    dim: int,
    hoi_fn: HoiFn,
    model_meta: object,
    reader: FrameSource | None = None,
) -> Path:
    """Core extraction loop with a pluggable embedder (``images_fn(list[img], batch=) -> [N,D]``) and
    hand–object source (``hoi_fn(step, native_frame_idx, img) -> (hands[2,7], hobj[2,5])``). Used by the
    teacher cache above and by M5 (student HOI + quantised LiteRT embedders)."""
    out_dir = out.parent
    reader = reader if reader is not None else FrameReader(video_id)
    n_native = len(reader)
    ts = np.arange(0.0, duration, 1.0 / RATE)
    T = len(ts)
    thumbs = np.zeros((T, *THUMB_HW, 3), np.uint8)
    hands = np.zeros((T, 2, 7), np.float32)
    hobj = np.zeros((T, 2, 5), np.float32)
    frame_imgs: list[np.ndarray] = []
    reg_t: list[int] = []
    reg_kind: list[int] = []
    reg_side: list[int] = []
    reg_box: list[tuple] = []
    reg_imgs: list[np.ndarray] = []
    frame_emb = []
    prev_obj = np.zeros((2, 5), np.float32)
    BATCH = 64

    def flush() -> None:
        if frame_imgs:
            frame_emb.append(images_fn(frame_imgs, batch=BATCH).astype(np.float16))
            frame_imgs.clear()

    region_emb: list[np.ndarray] = []

    def flush_regions() -> None:
        if reg_imgs:
            region_emb.append(images_fn(reg_imgs, batch=BATCH).astype(np.float16))
            reg_imgs.clear()

    for i, t in enumerate(tqdm(ts, desc=video_id, leave=False, disable=True)):
        fidx = min(n_native, int(round(t * fps)) + 1)
        img = reader.frame(fidx)
        thumbs[i] = np.asarray(Image.fromarray(img).resize(THUMB_HW[::-1], Image.Resampling.BILINEAR))
        H, O = hoi_fn(i, fidx, img)
        hands[i], hobj[i] = H, O
        frame_imgs.append(img)
        for side in (0, 1):
            if O[side, 0] and H[side, 2] in (PORTABLE, STATIONARY):
                for kind, f in (("obj", 1.15),):
                    reg_t.append(i)
                    reg_kind.append(REGION_KINDS.index(kind))
                    reg_side.append(side)
                    b = expand_box(O[side, 1:], f)
                    reg_box.append(b)
                    reg_imgs.append(crop(img, b))
            if prev_obj[side, 0]:
                for kind, f in (("pctx", 3.0),):
                    reg_t.append(i)
                    reg_kind.append(REGION_KINDS.index(kind))
                    reg_side.append(side)
                    b = expand_box(prev_obj[side, 1:], f)
                    reg_box.append(b)
                    reg_imgs.append(crop(img, b))
        prev_obj = O.copy()
        if len(frame_imgs) >= BATCH:
            flush()
        if len(reg_imgs) >= 4 * BATCH:
            flush_regions()
    flush()
    flush_regions()
    reader.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        t=ts.astype(np.float32),
        thumb=thumbs,
        hands=hands,
        hobj=hobj,
        frame_emb=np.concatenate(frame_emb),
        reg_t=np.array(reg_t, np.int32),
        reg_kind=np.array(reg_kind, np.int8),
        reg_side=np.array(reg_side, np.int8),
        reg_box=np.array(reg_box, np.float32).reshape(-1, 4),
        reg_emb=np.concatenate(region_emb) if region_emb else np.zeros((0, dim), np.float16),
        meta=json.dumps({"video_id": video_id, "fps": fps, "rate": RATE, "model": model_meta}),
    )
    return out


def load_feats(video_id: str, model: tuple[str, str] = DEFAULT_MODEL) -> dict[str, np.ndarray]:
    z = np.load(CACHE / "feats" / model[0] / f"{video_id}.npz")
    return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=None)
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--model", default=f"{DEFAULT_MODEL[0]}/{DEFAULT_MODEL[1]}")
    ap.add_argument("--order", default="fwd", choices=["fwd", "rev", "mid"])
    a = ap.parse_args()
    model = tuple(a.model.split("/"))
    subset = json.loads((REPO / "configs" / "epic_subset.json").read_text())
    if a.order == "rev":
        subset = subset[::-1]
    elif a.order == "mid":
        subset = subset[len(subset) // 2 :] + subset[: len(subset) // 2]
    for v in subset:
        if a.split and v["split"] != a.split:
            continue
        if a.videos and v["video_id"] not in a.videos:
            continue
        if not (tar_path(v["video_id"]).with_name(v["video_id"] + ".tar.ok")).exists():
            print("skip (not downloaded)", v["video_id"])
            continue
        p = extract_video(v["video_id"], v["fps"], v["duration"], model)  # type: ignore[arg-type]
        print("ok", p)


if __name__ == "__main__":
    main()
