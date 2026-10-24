"""Build the narration-derived query sets → configs/queries/{train,val,test}.jsonl (deterministic, seed 0)."""

from __future__ import annotations

import collections
import json

import pandas as pd

from recall.data.loaders.epic import fetch_annotations, load_narrations
from recall.data.queries import build_queries, save_queries
from recall.paths import EPIC_ANN, REPO


def main() -> None:
    fetch_annotations()
    narr = load_narrations()
    nouns = pd.read_csv(EPIC_ANN / "EPIC_100_noun_classes.csv")
    subset = json.loads((REPO / "configs" / "epic_subset.json").read_text())
    for split in ("train", "val", "test"):
        vs = [v for v in subset if v["split"] == split]
        qs = build_queries(narr, nouns, [v["video_id"] for v in vs], {v["video_id"]: v["duration"] for v in vs})
        save_queries(qs, REPO / "configs" / "queries" / f"{split}.jsonl")
        c = collections.Counter(q.kind + ("+loc" if q.gt_loc_class is not None else "") for q in qs)
        print(split, dict(sorted(c.items())))


if __name__ == "__main__":
    main()
