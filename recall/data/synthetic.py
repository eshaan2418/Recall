"""Synthetic egocentric "videos" in the feature-cache format, for CI and unit tests.

A synthetic stream contains objects that are taken and put on locations at
known times. Embeddings are noisy copies of random class prototypes, so the
whole pipeline (trigger → events → memory → QA → metrics) can run without any
licensed data. Nothing here is used for reported results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recall.data.loaders.hoi import NO_CONTACT, PORTABLE
from recall.data.queries import Query
from recall.perception.extract import REGION_KINDS, THUMB_HW
from recall.perception.heads import BlendHead, Heads


@dataclass
class SynthWorld:
    n_obj: int = 12
    n_loc: int = 6
    dim: int = 32
    noise: float = 0.35
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.obj_proto = _unit(rng.normal(size=(self.n_obj, self.dim)))
        self.loc_proto = _unit(rng.normal(size=(self.n_loc, self.dim)))

    def heads(self) -> Heads:
        o = BlendHead(self.obj_proto.astype(np.float32), 30.0)
        l = BlendHead(self.loc_proto.astype(np.float32), 30.0)
        return Heads(o, l, o, l, list(range(100, 100 + self.n_loc)), {"synthetic": True})


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_video(
    world: SynthWorld, vid: str, duration_s: float = 600.0, n_interactions: int = 20, seed: int = 0
) -> tuple[dict, pd.DataFrame]:
    """Return (feats dict in the cache format, narrations DataFrame)."""
    rng = np.random.default_rng(seed)
    T = int(duration_s * 2)
    t = np.arange(T) / 2.0
    hands = np.zeros((T, 2, 7), np.float32)
    hobj = np.zeros((T, 2, 5), np.float32)
    thumb = rng.integers(0, 30, size=(T, *THUMB_HW, 3)).astype(np.uint8)
    frame_emb = _unit(rng.normal(size=(T, world.dim)) * 0.2 + world.loc_proto[0])
    hands[:, 1, :3] = [1, 0.95, NO_CONTACT]
    rows = []
    reg = []  # (t, kind, side, box, emb)
    starts = np.sort(rng.choice(np.arange(20, T - 40, 12), size=n_interactions, replace=False))
    for s in starts:
        o = int(rng.integers(world.n_obj - 2))  # last two classes never appear (absent queries)
        loc = int(rng.integers(world.n_loc))
        L = int(rng.integers(3, 8))
        for i in range(s, s + L):
            hands[i, 1] = [1, 0.95, PORTABLE, 0.5, 0.5, 0.7, 0.8]
            hobj[i, 1] = [1, 0.4, 0.4, 0.6, 0.6]
            thumb[i] = np.clip(thumb[i].astype(int) + 120, 0, 255)
            reg.append(
                (
                    i,
                    "obj",
                    1,
                    (0.4, 0.4, 0.6, 0.6),
                    world.obj_proto[o] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
                )
            )
            reg.append(
                (
                    i,
                    "ctx",
                    1,
                    (0.3, 0.3, 0.7, 0.7),
                    world.loc_proto[loc] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
                )
            )
            if i > s:
                reg.append(
                    (
                        i,
                        "pobj",
                        1,
                        (0.4, 0.4, 0.6, 0.6),
                        world.obj_proto[o] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
                    )
                )
                reg.append(
                    (
                        i,
                        "pctx",
                        1,
                        (0.3, 0.3, 0.7, 0.7),
                        world.loc_proto[loc] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
                    )
                )
        e = s + L  # release frame: previous box shows the location
        reg.append(
            (
                e,
                "pobj",
                1,
                (0.4, 0.4, 0.6, 0.6),
                world.obj_proto[o] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
            )
        )
        reg.append(
            (
                e,
                "pctx",
                1,
                (0.3, 0.3, 0.7, 0.7),
                world.loc_proto[loc] + rng.normal(size=world.dim) * world.noise / np.sqrt(world.dim),
            )
        )
        rows.append(
            dict(
                video_id=vid,
                start_s=t[s],
                stop_s=t[s + 1],
                verb_class=0,
                noun_class=o,
                all_noun_classes=[o],
                narration=f"take obj{o}",
            )
        )
        rows.append(
            dict(
                video_id=vid,
                start_s=t[s + L - 2],
                stop_s=t[e],
                verb_class=1,
                noun_class=o,
                all_noun_classes=[o, 100 + loc],
                narration=f"put obj{o} on loc{loc}",
            )
        )
    reg.sort(key=lambda r: (r[0], REGION_KINDS.index(r[1]), r[2]))
    feats = {
        "t": t.astype(np.float32),
        "thumb": thumb,
        "hands": hands,
        "hobj": hobj,
        "frame_emb": frame_emb.astype(np.float16),
        "reg_t": np.array([r[0] for r in reg], np.int32),
        "reg_kind": np.array([REGION_KINDS.index(r[1]) for r in reg], np.int8),
        "reg_side": np.array([r[2] for r in reg], np.int8),
        "reg_box": np.array([r[3] for r in reg], np.float32),
        "reg_emb": _unit(np.array([r[4] for r in reg])).astype(np.float16),
    }
    return feats, pd.DataFrame(rows)


def where_queries(narr: pd.DataFrame, vid: str, duration_s: float) -> list[Query]:
    out = []
    puts = narr[narr.verb_class == 1].sort_values("start_s")
    for o, g in puts.groupby("noun_class"):
        r = g.iloc[-1]
        out.append(
            Query(
                f"{vid}_w{o}",
                vid,
                "where",
                f"Where is obj{o}?",
                duration_s - 1,
                int(o),
                f"obj{o}",
                float(r.start_s),
                float(r.stop_s),
                int(r.all_noun_classes[1]),
                f"loc{r.all_noun_classes[1]}",
                verb_class=1,
            )
        )
    out.append(Query(f"{vid}_a", vid, "absent", "Where is the unicorn?", duration_s - 1, 11, "unicorn"))
    return out
