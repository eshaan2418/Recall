"""Narration-derived ground truth and query sets for EPIC-KITCHENS-100.

Three query families (mirroring the Recall use cases):

* ``where``    — "Where is the X?"  GT = last *put* of X before the query
                  (object class, location class if narrated, time window).
* ``state``    — "Did I turn off the X?"  GT = last turn-on/turn-off of X.
* ``temporal`` — "When did I last <verb> the X?"  GT = time of the last such
                  narration (an NLQ-style query).
* ``absent``   — "Where is the X?" for X never narrated in the video; the
                  correct answer is to abstain ("I don't know").

Queries are issued *within* a video at time ``t_query``; only memory written
before ``t_query`` may be used to answer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

VERB_TAKE = 0
VERB_PUT = 1
VERB_INSERT = 5
VERB_TURN_ON = 6
VERB_TURN_OFF = 8
VERB_THROW = 13
PUT_VERBS = {VERB_PUT, VERB_INSERT, VERB_THROW}

# Location/surface classes (EPIC noun-class keys) that can be the answer to a
# "where" question. Curated from the second-noun distribution of put-narrations;
# excludes ingredients ("put salt on meat") and body parts.
LOCATION_KEYS = [
    "top",
    "rack:drying",
    "plate",
    "hob",
    "pan",
    "board:chopping",
    "oven",
    "bowl",
    "tray",
    "fridge",
    "pot",
    "sink",
    "box",
    "stand",
    "shelf",
    "cupboard",
    "bin",
    "maker:coffee",
    "scale",
    "cooker:slow",
    "container",
    "bag",
    "jar",
    "floor",
    "table",
    "drawer",
    "dishwasher",
    "microwave",
    "freezer",
    "mat",
    "basket",
    "colander",
    "kettle",
    "lid",
    "cup",
    "mug",
    "glass",
    "jug",
    "toaster",
    "machine:washing",
    "holder",
    "rack",
    "tin",
    "processor:food",
    "blender",
    "wok",
    "pitta",
    "tortilla",
]
# Human-readable names for prompts / answers.
LOCATION_NAMES = {
    "top": "kitchen counter",
    "rack:drying": "drying rack",
    "board:chopping": "chopping board",
    "maker:coffee": "coffee maker",
    "cooker:slow": "slow cooker",
    "machine:washing": "washing machine",
    "processor:food": "food processor",
}
# Object categories that make sense as "where did I leave X" targets.
OBJECT_CATEGORIES = {
    "utensils",
    "cutlery",
    "crockery",
    "cookware",
    "containers",
    "cleaning equipment and material",
    "appliances",
    "vegetables",
    "fruits and nuts",
    "dairy and eggs",
    "baked goods and grains",
    "storage",
}
NON_PORTABLE = {
    "tap",
    "cupboard",
    "drawer",
    "fridge",
    "hob",
    "oven",
    "sink",
    "dishwasher",
    "freezer",
    "microwave",
    "light",
    "machine:washing",
    "fan:extractor",
    "heater",
    "door:kitchen",
    "button",
    "knob",
    "plug",
    "power",
    "alarm",
    "heat",
    "top",
    "wall",
    "floor",
    "window",
    "kitchen",
}
# Appliances whose on/off state users ask about.
STATE_KEYS = ["tap", "hob", "oven", "light", "kettle", "microwave", "maker:coffee", "fan:extractor"]

MIN_DELAY_S = 20.0  # a "where" query is only asked if the object was put down ≥20 s earlier


def readable(key: str) -> str:
    if key in LOCATION_NAMES:
        return LOCATION_NAMES[key]
    if ":" in key:
        a, b = key.split(":", 1)
        return f"{b} {a}"
    return key


@dataclass
class Query:
    qid: str
    video_id: str
    kind: str  # where | state | temporal | absent
    text: str
    t_query: float
    obj_class: int
    obj_key: str
    gt_t0: float | None = None
    gt_t1: float | None = None
    gt_loc_class: int | None = None
    gt_loc_key: str | None = None
    gt_state: str | None = None  # "on" | "off"
    verb_class: int | None = None
    extra: dict = field(default_factory=dict)


def _second_location(noun_classes: list[int], loc_ids: set[int]) -> int | None:
    for c in noun_classes[1:]:
        if c in loc_ids:
            return c
    return None


def build_queries(
    narr: pd.DataFrame,
    nouns: pd.DataFrame,
    video_ids: list[str],
    durations: dict[str, float],
    seed: int = 0,
    n_absent_per_video: int = 3,
) -> list[Query]:
    rng = np.random.default_rng(seed)
    key2id = dict(zip(nouns.key, nouns.id, strict=True))
    id2key = dict(zip(nouns.id, nouns.key, strict=True))
    loc_ids = {key2id[k] for k in LOCATION_KEYS if k in key2id}
    obj_ids = {
        int(i)
        for i, k, c in zip(nouns.id, nouns.key, nouns.category, strict=True)
        if c in OBJECT_CATEGORIES and k not in NON_PORTABLE
    }
    out: list[Query] = []
    for vid in video_ids:
        d = narr[narr.video_id == vid].sort_values("start_s")
        dur = durations[vid]
        # ---- where ----
        for oc in sorted(set(d.noun_class) & obj_ids):
            ev = d[(d.noun_class == oc) & (d.verb_class.isin(PUT_VERBS | {VERB_TAKE}))]
            rows = list(ev.itertuples())
            for i, r in enumerate(rows):
                if r.verb_class not in PUT_VERBS:
                    continue
                nxt = next((s for s in rows[i + 1 :] if s.verb_class == VERB_TAKE), None)
                nxt_put = next((s for s in rows[i + 1 :] if s.verb_class in PUT_VERBS), None)
                if nxt_put is not None and (nxt is None or nxt_put.start_s < nxt.start_s):
                    continue  # another put happens before any take → that one is the GT
                t_q = (nxt.start_s - 1.0) if nxt is not None else dur - 0.5
                if t_q - r.stop_s < MIN_DELAY_S:
                    continue
                loc = _second_location(list(r.all_noun_classes), loc_ids)
                k = id2key[oc]
                out.append(
                    Query(
                        qid=f"{vid}_w{len(out)}",
                        video_id=vid,
                        kind="where",
                        text=f"Where is the {readable(k)}?",
                        t_query=float(t_q),
                        obj_class=int(oc),
                        obj_key=k,
                        gt_t0=float(r.start_s),
                        gt_t1=float(r.stop_s),
                        gt_loc_class=None if loc is None else int(loc),
                        gt_loc_key=None if loc is None else id2key[loc],
                        verb_class=int(r.verb_class),
                        extra={"narration": r.narration, "delay_s": float(t_q - r.stop_s)},
                    )
                )
        # ---- state ----
        for sk in STATE_KEYS:
            if sk not in key2id:
                continue
            sc = key2id[sk]
            ev = d[(d.noun_class == sc) & (d.verb_class.isin([VERB_TURN_ON, VERB_TURN_OFF]))]
            rows = list(ev.itertuples())
            for i, r in enumerate(rows):
                nxt_t = rows[i + 1].start_s if i + 1 < len(rows) else dur
                if nxt_t - r.stop_s < MIN_DELAY_S:
                    continue
                t_q = float(rng.uniform(r.stop_s + MIN_DELAY_S / 2, nxt_t - 0.5))
                st = "on" if r.verb_class == VERB_TURN_ON else "off"
                out.append(
                    Query(
                        qid=f"{vid}_s{len(out)}",
                        video_id=vid,
                        kind="state",
                        text=f"Did I turn off the {readable(sk)}?",
                        t_query=t_q,
                        obj_class=int(sc),
                        obj_key=sk,
                        gt_t0=float(r.start_s),
                        gt_t1=float(r.stop_s),
                        gt_state=st,
                        verb_class=int(r.verb_class),
                        extra={"narration": r.narration},
                    )
                )
        # ---- temporal (NLQ-style): last occurrence of (verb, noun) ----
        pairs = d.groupby(["verb_class", "noun_class"]).size()
        pairs = pairs[pairs >= 1].index.tolist()
        rng.shuffle(pairs)
        n_t = 0
        for vc, nc in pairs:
            if n_t >= 12:
                break
            if vc not in (VERB_TAKE, VERB_PUT, 3, 4, VERB_TURN_ON, VERB_TURN_OFF, 2):
                continue  # take, put, open, close, turn-on, turn-off, wash
            ev = d[(d.verb_class == vc) & (d.noun_class == nc)]
            r = ev.iloc[-1]
            t_q = float(min(dur - 0.5, r.stop_s + rng.uniform(MIN_DELAY_S, 180.0)))
            if t_q - r.stop_s < MIN_DELAY_S:
                continue
            verb = r.verb
            out.append(
                Query(
                    qid=f"{vid}_t{len(out)}",
                    video_id=vid,
                    kind="temporal",
                    text=f"When did I last {verb.replace('-', ' ')} the {readable(id2key[nc])}?",
                    t_query=t_q,
                    obj_class=int(nc),
                    obj_key=id2key[nc],
                    gt_t0=float(r.start_s),
                    gt_t1=float(r.stop_s),
                    verb_class=int(vc),
                    extra={"narration": r.narration},
                )
            )
            n_t += 1
        # ---- absent: objects this *participant* never narrates ----
        part = vid.split("_")[0]
        seen = set(narr[narr.video_id.str.startswith(part + "_")].noun_class)
        cands = sorted(obj_ids - seen)
        for oc in rng.choice(cands, size=min(n_absent_per_video, len(cands)), replace=False):
            k = id2key[int(oc)]
            out.append(
                Query(
                    qid=f"{vid}_a{len(out)}",
                    video_id=vid,
                    kind="absent",
                    text=f"Where is the {readable(k)}?",
                    t_query=float(dur - 0.5),
                    obj_class=int(oc),
                    obj_key=k,
                )
            )
    return out


def save_queries(qs: list[Query], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(asdict(q)) for q in qs) + "\n")


def load_queries(path: Path) -> list[Query]:
    return [Query(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def interaction_windows(narr: pd.DataFrame, video_id: str, verbs: set[int] | None = None) -> np.ndarray:
    """[N,2] array of (start_s, stop_s) for hand–object interactions used as trigger labels."""
    verbs = verbs or (PUT_VERBS | {VERB_TAKE, 3, 4, VERB_TURN_ON, VERB_TURN_OFF})
    d = narr[(narr.video_id == video_id) & (narr.verb_class.isin(verbs))]
    return d[["start_s", "stop_s"]].to_numpy(dtype=float)
