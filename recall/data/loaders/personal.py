"""Personal first-person recordings (M9). Everything under ``data/personal/`` is git-ignored.

Layout::

    data/personal/raw/<name>.mp4                 your recordings (never committed)
    data/personal/annotations/<name>.jsonl       written by the annotation tool (recall/data/annotate)
    data/personal/cache/                         features / events (derived, still private)

Annotation line (one per interaction)::

    {"video": "<name>", "t": 123.4, "kind": "put" | "take", "object": "keys", "place": "hallway shelf"}

"where is X" ground truth = the last ``put`` of X before the query time; the query is asked at the next
``take`` of X (just before it — the moment the user would be looking for it) or at the end of the recording,
and only if the put is ≥20 s earlier (same rule as the EPIC query set).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from recall.data.queries import MIN_DELAY_S, Query
from recall.paths import REPO

PERSONAL = REPO / "data" / "personal"


class VideoFileReader:
    """FrameSource over an MP4/MOV via OpenCV (1-indexed native frames, like the EPIC reader)."""

    def __init__(self, path: Path, max_side: int = 456):
        import cv2

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_side = max_side
        self._pos = 0

    def __len__(self) -> int:
        return self.n

    @property
    def duration(self) -> float:
        return self.n / self.fps

    def frame(self, idx: int) -> np.ndarray:
        import cv2

        idx = min(max(idx, 1), self.n)
        if idx - 1 != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, bgr = self.cap.read()
        self._pos = idx
        if not ok:
            raise IndexError(idx)
        h, w = bgr.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


@dataclass
class Annotation:
    video: str
    t: float
    kind: str
    object: str
    place: str | None = None


def load_annotations(path: Path) -> list[Annotation]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(
                Annotation(d["video"], float(d["t"]), d["kind"], d["object"].strip().lower(), (d.get("place") or None))
            )
    return sorted(out, key=lambda a: a.t)


def build_queries(
    anns: list[Annotation],
    duration: float,
    obj_class: dict[str, int],
    loc_class: dict[str, int],
    absent_objects: list[str],
    window_s: float = 3.0,
) -> list[Query]:
    """Where-is-X queries from put/take annotations + "absent" queries for never-annotated objects."""
    qs: list[Query] = []
    if not anns:
        return qs
    vid = anns[0].video
    for i, a in enumerate(anns):
        if a.kind != "put":
            continue
        nxt = next((b for b in anns[i + 1 :] if b.object == a.object), None)
        if nxt is not None and nxt.kind == "put":
            continue  # re-placed without an annotated take; ambiguous → skip
        t_q = (nxt.t - 1.0) if nxt is not None else duration
        if t_q - a.t < MIN_DELAY_S:
            continue
        qs.append(
            Query(
                f"{vid}_where_{i}",
                vid,
                "where",
                f"where is the {a.object}?",
                t_q,
                obj_class.get(a.object, -1),
                a.object,
                a.t - window_s,
                a.t + window_s,
                loc_class.get(a.place or ""),
                a.place,
            )
        )
    for j, o in enumerate(absent_objects):
        qs.append(Query(f"{vid}_absent_{j}", vid, "absent", f"where is the {o}?", duration, obj_class.get(o, -1), o))
    return qs
