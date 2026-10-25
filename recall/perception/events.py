"""Hand–object interaction → episodic events, from whichever frames a trigger policy processed.

For each hand (left/right) we track a *held* object across processed frames.

* contact starts                 → ``take`` event
* contact ends / object changes  → ``put`` event for the previously held object
* stationary contact (fridge door, tap, knob …) → ``touch`` event (one per contact bout)

Only frames in ``processed`` contribute (the heavy model was not run on the
others), so sparse triggering genuinely loses information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recall.data.loaders.hoi import PORTABLE, STATIONARY
from recall.memory.schema import Event
from recall.perception.extract import REGION_KINDS


@dataclass
class ExtractorConfig:
    max_gap: int = 6  # processed frames further apart than this (×0.5 s) break a holding bout
    switch_sim: float = 0.40  # cosine sim below which the held object is considered a different object
    min_obs: int = 1
    release_frames: int = 2  # consecutive processed no-contact frames before a release is declared (hysteresis)


class RegionIndex:
    """Fast lookup (frame, kind, side) → region embedding row."""

    def __init__(self, feats: dict[str, np.ndarray]):
        self.emb = feats["reg_emb"]
        self.box = feats["reg_box"]
        self._ix: dict[tuple[int, int, int], int] = {}
        for j, (t, k, s) in enumerate(zip(feats["reg_t"], feats["reg_kind"], feats["reg_side"], strict=True)):
            self._ix[(int(t), int(k), int(s))] = j

    def get(self, i: int, kind: str, side: int) -> int | None:
        return self._ix.get((i, REGION_KINDS.index(kind), side))


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def extract_events(
    feats: dict[str, np.ndarray],
    processed: np.ndarray,
    stream_id: str,
    cfg: ExtractorConfig | None = None,
    regions: RegionIndex | None = None,
) -> list[Event]:
    cfg = cfg or ExtractorConfig()
    reg = regions or RegionIndex(feats)
    t = feats["t"]
    hands, femb = feats["hands"], feats["frame_emb"]
    idx = np.flatnonzero(processed)
    out: list[Event] = []

    for side in (0, 1):
        held: dict | None = None
        touch: dict | None = None
        last_i = -(10**9)

        def close_held(i_rel: int | None, side: int = side) -> None:
            nonlocal held
            assert held is not None
            o = _unit(np.mean(held["embs"], 0))
            # where did it go? prefer the crop around the old box *after* release
            loc, rel_obs = None, False
            if i_rel is not None:
                j = reg.get(i_rel, "pctx", side)
                if j is not None:
                    loc, rel_obs = reg.emb[j].astype(np.float32), True
            if loc is None:
                loc = held["ctx"][-1]
            ti = i_rel if i_rel is not None else held["last"]
            out.append(
                Event(
                    stream_id,
                    float(t[ti]),
                    "put",
                    side,
                    int(ti),
                    o,
                    _unit(loc),
                    _unit(femb[ti].astype(np.float32)),
                    n_obs=len(held["embs"]),
                    hand_score=float(np.mean(held["scores"])),
                    release_observed=rel_obs,
                    box=held["box"],
                    t_start=float(t[held["first"]]),
                )
            )
            held = None

        def close_touch(side: int = side) -> None:
            nonlocal touch
            assert touch is not None
            o = _unit(np.mean(touch["embs"], 0))
            c = _unit(np.mean(touch["ctx"], 0))
            ti = touch["last"]
            out.append(
                Event(
                    stream_id,
                    float(t[ti]),
                    "touch",
                    side,
                    int(ti),
                    o,
                    c,
                    _unit(femb[ti].astype(np.float32)),
                    n_obs=len(touch["embs"]),
                    hand_score=float(np.mean(touch["scores"])),
                    box=touch["box"],
                    t_start=float(t[touch["first"]]),
                )
            )
            touch = None

        for i in (int(x) for x in idx):
            gap = i - last_i
            if held is not None and gap > cfg.max_gap:
                close_held(held["rel"] if held["miss"] else None)  # release not (fully) observed
            if touch is not None and gap > cfg.max_gap:
                close_touch()
            present, score, state = hands[i, side, 0], hands[i, side, 1], int(hands[i, side, 2])
            jo = reg.get(i, "obj", side)
            jc = reg.get(i, "ctx", side)
            portable = bool(present) and state == PORTABLE and jo is not None
            stationary = bool(present) and state == STATIONARY and jo is not None
            if portable:
                e = reg.emb[jo].astype(np.float32)
                b4 = reg.box[jo]
                box = (float(b4[0]), float(b4[1]), float(b4[2]), float(b4[3]))
                if held is not None:
                    sim = float(_unit(np.mean(held["embs"], 0)) @ _unit(e))
                    if sim < cfg.switch_sim:
                        close_held(held["rel"] if held["miss"] else (i if gap == 1 else None))
                    else:
                        held["miss"], held["rel"] = 0, None  # brief flicker: same object, keep holding
                if held is None:
                    held = {"embs": [], "ctx": [], "scores": [], "first": i, "last": i, "box": box, "miss": 0,
                            "rel": None}
                    ctx = reg.emb[jc].astype(np.float32) if jc is not None else femb[i].astype(np.float32)
                    out.append(
                        Event(
                            stream_id,
                            float(t[i]),
                            "take",
                            side,
                            i,
                            _unit(e),
                            _unit(ctx),
                            _unit(femb[i].astype(np.float32)),
                            hand_score=float(score),
                            box=box,
                            t_start=float(t[i]),
                        )
                    )
                held["embs"].append(e)
                held["ctx"].append(reg.emb[jc].astype(np.float32) if jc is not None else femb[i].astype(np.float32))
                held["scores"].append(float(score))
                held["last"], held["box"] = i, box
            elif held is not None:
                held["miss"] += 1
                if held["miss"] == 1:
                    held["rel"] = i if gap == 1 else None
                if held["miss"] >= cfg.release_frames:
                    close_held(held["rel"])
            if stationary:
                e = reg.emb[jo].astype(np.float32)
                if touch is None:
                    touch = {
                        "embs": [],
                        "ctx": [],
                        "scores": [],
                        "first": i,
                        "last": i,
                        "box": tuple(float(x) for x in reg.box[jo]),
                    }
                touch["embs"].append(e)
                touch["ctx"].append(reg.emb[jc].astype(np.float32) if jc is not None else femb[i].astype(np.float32))
                touch["scores"].append(float(score))
                touch["last"] = i
            elif touch is not None:
                close_touch()
            last_i = i
        if held is not None:
            close_held(held["rel"] if held["miss"] else None)
        if touch is not None:
            close_touch()
    out = [ev for ev in out if ev.n_obs >= cfg.min_obs or ev.kind == "take"]
    out.sort(key=lambda ev: (ev.t, ev.side))
    for k, ev in enumerate(out):
        ev.event_id = k
    return out
