"""Decoder for EPIC-100 hand–object detections (Shan et al., CVPR 2020 detector outputs).

The release is a pickled ``list[bytes]`` of protobuf ``Detections`` messages
(schema: epic-kitchens-100-hand-object-bboxes/types.proto). We decode the
protobuf wire format directly (no generated code) and unpickle with a
restricted unpickler that refuses to import any global, so a tampered file
cannot execute code.

Coordinates are normalised to [0, 1] (left, top, right, bottom).
"""

from __future__ import annotations

import pickle
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NO_CONTACT, SELF_CONTACT, OTHER_PERSON, PORTABLE, STATIONARY = range(5)


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def _varint(b: bytes, i: int) -> tuple[int, int]:
    shift = res = 0
    while True:
        c = b[i]
        i += 1
        res |= (c & 0x7F) << shift
        if not c & 0x80:
            return res, i
        shift += 7


def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield fn, v
        elif wt == 5:
            yield fn, struct.unpack_from("<f", b, i)[0]
            i += 4
        elif wt == 1:
            yield fn, struct.unpack_from("<d", b, i)[0]
            i += 8
        elif wt == 2:
            ln, i = _varint(b, i)
            yield fn, b[i : i + ln]
            i += ln
        else:  # pragma: no cover
            raise ValueError(f"wire type {wt}")


def _bbox(b: bytes) -> tuple[float, float, float, float]:
    v = [0.0, 0.0, 0.0, 0.0]
    for fn, x in _fields(b):
        v[fn - 1] = float(x)
    return v[0], v[1], v[2], v[3]


def _vec(b: bytes) -> tuple[float, float]:
    v = [0.0, 0.0]
    for fn, x in _fields(b):
        v[fn - 1] = float(x)
    return v[0], v[1]


@dataclass
class Hand:
    box: tuple[float, float, float, float]
    score: float
    state: int
    offset: tuple[float, float]
    side: int  # 0 left, 1 right


@dataclass
class Obj:
    box: tuple[float, float, float, float]
    score: float


@dataclass
class FrameHOI:
    frame: int
    hands: list[Hand]
    objects: list[Obj]

    def contacted_object(self, hand: Hand, obj_thresh: float = 0.01) -> Obj | None:
        """Object a hand is in contact with (nearest object to hand centre + offset)."""
        if hand.state not in (PORTABLE, STATIONARY) or not self.objects:
            return None
        l, t, r, b = hand.box
        cx, cy = (l + r) / 2 + hand.offset[0], (t + b) / 2 + hand.offset[1]
        best, bd = None, 1e9
        for o in self.objects:
            if o.score < obj_thresh:
                continue
            ox, oy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
            d = (ox - cx) ** 2 + (oy - cy) ** 2
            if d < bd:
                best, bd = o, d
        return best


def decode_frame(b: bytes) -> FrameHOI:
    frame = 0
    hands: list[Hand] = []
    objs: list[Obj] = []
    for fn, v in _fields(b):
        if fn == 2:
            frame = int(v)
        elif fn == 3:
            box, score, state, off, side = (0.0, 0.0, 0.0, 0.0), 0.0, 0, (0.0, 0.0), 0
            for f2, x in _fields(v):
                if f2 == 1:
                    box = _bbox(x)
                elif f2 == 2:
                    score = float(x)
                elif f2 == 3:
                    state = int(x)
                elif f2 == 4:
                    off = _vec(x)
                elif f2 == 5:
                    side = int(x)
            hands.append(Hand(box, score, state, off, side))
        elif fn == 4:
            box, score = (0.0, 0.0, 0.0, 0.0), 0.0
            for f2, x in _fields(v):
                if f2 == 1:
                    box = _bbox(x)
                elif f2 == 2:
                    score = float(x)
            objs.append(Obj(box, score))
    return FrameHOI(frame, hands, objs)


def load_hoi(path: Path) -> list[FrameHOI]:
    with open(path, "rb") as f:
        raw = _SafeUnpickler(f).load()
    return [decode_frame(s) for s in raw]


def hoi_features(fr: FrameHOI, hand_thresh: float = 0.5) -> np.ndarray:
    """Compact per-frame summary: [n_hands, max_score, any_portable_contact, any_stationary, n_objects]."""
    hs = [h for h in fr.hands if h.score >= hand_thresh]
    return np.array(
        [
            len(hs),
            max((h.score for h in fr.hands), default=0.0),
            float(any(h.state == PORTABLE for h in hs)),
            float(any(h.state == STATIONARY for h in hs)),
            len([o for o in fr.objects if o.score >= 0.5]),
        ],
        dtype=np.float32,
    )
