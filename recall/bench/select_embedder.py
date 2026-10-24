"""Model selection for the image/text embedder (train split only — never test).

Zero-shot object recognition: at the midpoint of every "take X" narration, crop
the object the hand detector says is in contact and classify it among all 300
EPIC noun classes. Location recognition: at the end of every "put X in/on Y"
narration, classify the context crop around the released object among the
location vocabulary.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from recall.bench.common import result_header, write_result
from recall.data.loaders.epic import FrameReader, load_narrations, tar_path
from recall.data.loaders.hoi import PORTABLE, load_hoi
from recall.data.queries import LOCATION_KEYS, PUT_VERBS, VERB_TAKE
from recall.paths import EPIC_RAW, REPO
from recall.perception.embed import load_embedder, zero_shot_probs
from recall.perception.extract import crop, expand_box, hands_array
from recall.perception.vocab import location_bank, location_ids, noun_table, object_bank

CANDIDATES = [
    ("MobileCLIP2-S0", "dfndr2b"),
    ("MobileCLIP2-S2", "dfndr2b"),
    ("ViT-B-16-SigLIP2-256", "webli"),
    ("ViT-B-16", "dfn2b"),
    ("ViT-L-14", "dfn2b"),
]


def collect(videos: list[str], max_per_video: int = 120) -> tuple[list, list, list, list]:
    narr = load_narrations()
    subset = {v["video_id"]: v for v in json.loads((REPO / "configs/epic_subset.json").read_text())}
    k2i = dict(zip(noun_table().key, noun_table().id, strict=True))
    loc_ids = {k2i[k] for k in LOCATION_KEYS if k in k2i}
    obj_imgs, obj_lab, loc_imgs, loc_lab = [], [], [], []
    for vid in videos:
        fps = subset[vid]["fps"]
        hoi = load_hoi(EPIC_RAW / "hoi" / f"{vid}.pkl")
        rd = FrameReader(vid)
        d = narr[narr.video_id == vid]
        n = 0
        for r in d[d.verb_class == VERB_TAKE].itertuples():
            f = int(round((r.start_s + r.stop_s) / 2 * fps)) + 1
            H, O = hands_array(hoi[min(f - 1, len(hoi) - 1)])
            sides = [s for s in (1, 0) if O[s, 0] and H[s, 2] == PORTABLE]
            if not sides or n >= max_per_video:
                continue
            obj_imgs.append(crop(rd.frame(f), expand_box(O[sides[0], 1:], 1.15)))
            obj_lab.append(int(r.noun_class))
            n += 1
        for r in d[d.verb_class.isin(PUT_VERBS)].itertuples():
            locs = [c for c in r.all_noun_classes[1:] if c in loc_ids]
            if not locs:
                continue
            # object box while still held, near the end of the action
            f_hold = int(round((r.stop_s - 0.3) * fps)) + 1
            H, O = hands_array(hoi[min(f_hold - 1, len(hoi) - 1)])
            sides = [s for s in (1, 0) if O[s, 0]]
            box = expand_box(O[sides[0], 1:], 3.0) if sides else (0.0, 0.0, 1.0, 1.0)
            f_rel = int(round((r.stop_s + 0.5) * fps)) + 1
            loc_imgs.append(crop(rd.frame(f_rel), box))
            loc_lab.append(int(locs[0]))
        rd.close()
    return obj_imgs, obj_lab, loc_imgs, loc_lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    a = ap.parse_args()
    subset = json.loads((REPO / "configs/epic_subset.json").read_text())
    vids = a.videos or [
        v["video_id"]
        for v in subset
        if v["split"] == "train" and tar_path(v["video_id"]).with_name(v["video_id"] + ".tar.ok").exists()
    ]
    obj_imgs, obj_lab, loc_imgs, loc_lab = collect(vids)
    lid = location_ids()
    loc_y = np.array([lid.index(c) for c in loc_lab])
    rows = []
    for name, pre in CANDIDATES:
        emb = load_embedder(name, pre)
        t0 = time.time()
        oe = emb.images(obj_imgs)
        dt = (time.time() - t0) / max(1, len(obj_imgs))
        le = emb.images(loc_imgs)
        po = zero_shot_probs(oe, object_bank(emb))
        pl = zero_shot_probs(le, location_bank(emb))
        y = np.array(obj_lab)
        top1 = float((po.argmax(1) == y).mean())
        top5 = float((np.argsort(-po, 1)[:, :5] == y[:, None]).any(1).mean())
        ltop1 = float((pl.argmax(1) == loc_y).mean())
        ltop3 = float((np.argsort(-pl, 1)[:, :3] == loc_y[:, None]).any(1).mean())
        params = sum(int(p.numel()) for p in emb.model.visual.parameters()) / 1e6  # type: ignore[union-attr,misc]
        rows.append(
            dict(
                model=f"{name}/{pre}",
                obj_top1=top1,
                obj_top5=top5,
                loc_top1=ltop1,
                loc_top3=ltop3,
                img_encoder_params_M=params,
                mps_ms_per_img=dt * 1e3,
            )
        )
        print(rows[-1])
        del emb
        load_embedder.cache_clear()
    write_result(
        "m1/select_embedder.json",
        {
            **result_header(stage="desktop-proxy", seed=0, config={"videos": vids}),
            "n_obj": len(obj_lab),
            "n_loc": len(loc_lab),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
