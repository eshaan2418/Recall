"""EPIC-KITCHENS-100 loader: annotation parsing, subset selection, downloads, frames.

License: EPIC-KITCHENS-100 is CC BY-NC 4.0. We download it locally for
non-commercial research and never redistribute frames; only code, configs and
derived metrics are committed.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from recall.data.download import download
from recall.paths import EPIC_ANN, EPIC_RAW, REPO

BASE55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
BASE100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
BASE_HOI = "https://data.bris.ac.uk/datasets/3l8eci2oqgst92n14w2yqi5ytu"
ANN_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master"
MD5_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-download-scripts/master/data/md5.csv"
ANN_FILES = [
    "EPIC_100_train.csv",
    "EPIC_100_validation.csv",
    "EPIC_100_video_info.csv",
    "EPIC_100_noun_classes.csv",
    "EPIC_100_verb_classes.csv",
]

# Participant-disjoint split (see DECISIONS.md 2026-10-24). Participants were chosen
# among those with many located put-events; test participants never appear in
# trigger or calibration training.
SPLIT_PARTICIPANTS = {
    "train": ["P02", "P04", "P06", "P24", "P28"],
    "val": ["P18", "P30"],
    "test": ["P01", "P03", "P10", "P11", "P23", "P37"],
}
HOURS_PER_PARTICIPANT = {"train": 1.3, "val": 1.0, "test": 1.0}
MIN_DUR_S, MAX_DUR_S = 360.0, 1800.0


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    duration: float
    fps: float
    split: str

    @property
    def participant(self) -> str:
        return self.video_id.split("_")[0]

    @property
    def is_ext(self) -> bool:
        """EPIC-100 extension videos have a 3-digit suffix (P01_101)."""
        return len(self.video_id.split("_")[1]) == 3


def fetch_annotations(dest: Path = EPIC_ANN) -> None:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    for f in [*ANN_FILES]:
        p = dest / f
        if not p.exists():
            r = requests.get(f"{ANN_URL}/{f}", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
    p = dest / "md5.csv"
    if not p.exists():
        r = requests.get(MD5_URL, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)


def load_narrations(ann: Path = EPIC_ANN) -> pd.DataFrame:
    d = pd.concat(
        [pd.read_csv(ann / "EPIC_100_train.csv"), pd.read_csv(ann / "EPIC_100_validation.csv")],
        ignore_index=True,
    )
    for c in ["start_timestamp", "stop_timestamp", "narration_timestamp"]:
        d[c.replace("timestamp", "s")] = pd.to_timedelta(d[c]).dt.total_seconds()
    d["all_nouns"] = d["all_nouns"].apply(ast.literal_eval)
    d["all_noun_classes"] = d["all_noun_classes"].apply(ast.literal_eval)
    return d


def select_subset(ann: Path = EPIC_ANN) -> list[VideoInfo]:
    """Deterministic video subset: per participant, prefer videos rich in located puts."""
    d = load_narrations(ann)
    info = pd.read_csv(ann / "EPIC_100_video_info.csv").set_index("video_id")
    puts = d[d.verb.str.startswith("put")]
    located = puts[puts.all_noun_classes.apply(len) >= 2].groupby("video_id").size()
    narrated = set(d.video_id)
    out: list[VideoInfo] = []
    for split, parts in SPLIT_PARTICIPANTS.items():
        for p in parts:
            vids = info[info.index.str.startswith(p + "_")]
            vids = vids[(vids.duration >= MIN_DUR_S) & (vids.duration <= MAX_DUR_S)]
            vids = vids[vids.index.isin(narrated)]  # EPIC-100 test-split videos have no public narrations
            vids = vids.assign(lp=located.reindex(vids.index).fillna(0))
            vids = vids.sort_values(["lp", "duration"], ascending=False)
            budget = HOURS_PER_PARTICIPANT[split] * 3600
            used = 0.0
            for vid, row in vids.iterrows():
                if used >= budget:
                    break
                if used + row.duration > budget * 1.3 and used > 0:
                    continue
                out.append(VideoInfo(str(vid), float(row.duration), float(row.fps), split))
                used += row.duration
    return out


def rgb_tar_remote(v: VideoInfo, md5: pd.DataFrame) -> tuple[str, str | None]:
    if v.is_ext:
        rel = f"{v.participant}/rgb_frames/{v.video_id}.tar"
        url = f"{BASE100}/{rel}"
    else:
        hit = md5[
            md5.file_remote_path.str.match(rf"frames_rgb_flow/rgb/(train|test)/{v.participant}/{v.video_id}\.tar")
        ]
        rel = hit.file_remote_path.iloc[0]
        url = f"{BASE55}/{rel}"
    m = md5[md5.file_remote_path == rel]
    return url, (m.md5.iloc[0] if len(m) else None)


def imu_remote(v: VideoInfo) -> list[str]:
    if not v.is_ext:
        return []
    return [f"{BASE100}/{v.participant}/meta_data/{v.video_id}-{k}.csv" for k in ("accl", "gyro")]


def hoi_remote(v: VideoInfo) -> str:
    return f"{BASE_HOI}/hand-objects/{v.participant}/{v.video_id}.pkl"


def tar_path(video_id: str) -> Path:
    return EPIC_RAW / "rgb" / f"{video_id}.tar"


def frames_dir(video_id: str) -> Path:
    return EPIC_RAW / "frames" / video_id


class FrameReader:
    """Reads EPIC rgb frames (456x256 JPEG, 1-indexed, at native fps) from the tar."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self._tar = tarfile.open(tar_path(video_id))  # noqa: SIM115 - closed in close()
        self._index: dict[int, tarfile.TarInfo] = {}
        for m in self._tar.getmembers():
            name = Path(m.name).name
            if name.startswith("frame_") and name.endswith(".jpg"):
                self._index[int(name[6:-4])] = m

    def __len__(self) -> int:
        return len(self._index)

    def frame(self, idx: int) -> np.ndarray:
        idx = min(max(idx, 1), max(self._index))
        f = self._tar.extractfile(self._index[idx])
        assert f is not None
        return np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"))

    def close(self) -> None:
        self._tar.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "download"])
    ap.add_argument("--split", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    fetch_annotations()
    subset = select_subset()
    manifest = REPO / "configs" / "epic_subset.json"
    manifest.write_text(json.dumps([v.__dict__ for v in subset], indent=1))
    if a.cmd == "select":
        for s in SPLIT_PARTICIPANTS:
            vs = [v for v in subset if v.split == s]
            print(s, len(vs), f"{sum(v.duration for v in vs) / 3600:.2f} h")
        return
    md5 = pd.read_csv(EPIC_ANN / "md5.csv")
    order = sorted(subset, key=lambda v: {"test": 0, "val": 1, "train": 2}[v.split])
    for v in order:
        if a.split and v.split != a.split:
            continue
        url, m = rgb_tar_remote(v, md5)
        print(f"[{v.split}] {v.video_id} {v.duration / 60:.1f} min <- {url}")
        download(url, tar_path(v.video_id), m, a.workers)
        for u in imu_remote(v):
            download(u, EPIC_RAW / "imu" / Path(u).name, None, 1)
        try:
            download(hoi_remote(v), EPIC_RAW / "hoi" / f"{v.video_id}.pkl", None, 2)
        except Exception as e:  # pragma: no cover - optional
            print("  hand-object bboxes unavailable:", e)


if __name__ == "__main__":
    main()
