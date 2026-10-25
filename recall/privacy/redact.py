"""Python port of the on-device redactor (android/.../privacy/Detectors.kt + Redactor.kt).

Mirrors the Kotlin code step by step so desktop numbers describe what the phone does:

* :class:`BlazeFaceDetector` — MediaPipe BlazeFace short-range (``face_detector.tflite``, Apache-2.0).
  The whole image is squashed (no letterbox, aspect not preserved) to 128x128 RGB in [-1, 1]; outputs are
  regressors [1,896,16] and logits [1,896,1] (identified by shape); anchors per the MediaPipe SSD config (strides
  8,16,16,16, unit anchor size, 0.5 offsets). Every anchor with sigmoid(score) >= 0.4 becomes a box padded by 25% in
  width and height; right/bottom are ``toInt() + 1``. **No NMS** (overlapping boxes are harmless for redaction),
  keypoints ignored.
* :class:`ScreenDetector` — 16x9 grid; a cell is flagged when mean luma >= 150 and >= 10% of its pixels have a
  luma step > 40 to the right or below neighbour (last row/col of each cell excluded, as in Kotlin).
* :class:`Redactor` — fail closed: needs a detector named ``"face"`` (unless ``require_face_detector=False``) and
  returns ``None`` if any detector raises. Detectors run in order on the *progressively redacted* copy; each box is
  pixelated with block ``max(6, min(box_w, box_h) // 6)`` px (block colour = pixel at block centre, clamped to the
  clipped box), then the result is downscaled to 160x90 and JPEG-encoded at quality 70.

Known deviations from the Kotlin (documented in privacy/README.md):

* Resizing uses a numpy half-pixel-centre bilinear (no antialiasing), the closest match to
  ``Bitmap.createScaledBitmap(..., filter=true)``; Skia's fixed-point rounding can differ by about 1 LSB, which can
  flip borderline detections.
* Luma / sigmoid are computed in float64 rather than float32.
* JPEG encoding uses libjpeg via Pillow rather than Android's encoder (thumbnail bytes differ; redaction does not).

Only numpy + Pillow are needed, except :class:`BlazeFaceDetector`, which imports ``ai_edge_litert`` lazily.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from recall.paths import MODELS

Box = tuple[
    int, int, int, int
]  # (left, top, right, bottom), pixel coords, right/bottom exclusive (android.graphics.Rect)

FACE_MODEL = MODELS / "litert" / "face_detector.tflite"
BF_SIZE = 128


class SensitiveRegionDetector(Protocol):
    name: str

    def detect(self, img: np.ndarray) -> list[Box]: ...


# ----------------------------------------------------------------------------------------------- resizing


def resize_bilinear(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """HxWxC uint8 -> hxwxC uint8, half-pixel-centre bilinear without antialiasing (Skia ``filter=true``)."""
    H, W = img.shape[:2]
    if (h, w) == (H, W):
        return img.copy()

    def coords(n_out: int, n_in: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        s = np.clip(s, 0, n_in - 1)
        i0 = np.floor(s).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (s - i0)

    y0, y1, fy = coords(h, H)
    x0, x1, fx = coords(w, W)
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - fx)[None, :, None] + f[y0][:, x1] * fx[None, :, None]
    bot = f[y1][:, x0] * (1 - fx)[None, :, None] + f[y1][:, x1] * fx[None, :, None]
    out = top * (1 - fy)[:, None, None] + bot * fy[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- BlazeFace


def blazeface_anchors() -> np.ndarray:
    """(896, 2) anchor centres in [0,1], same order as the Kotlin ``buildList``: 16x16x2 (stride 8), then 8x8x6."""
    strides = [8, 16, 16, 16]
    out: list[tuple[float, float]] = []
    i = 0
    while i < len(strides):
        j, n = i, 0
        while j < len(strides) and strides[j] == strides[i]:
            n += 2
            j += 1
        fm = BF_SIZE // strides[i]
        for y in range(fm):
            for x in range(fm):
                out.extend([((x + 0.5) / fm, (y + 0.5) / fm)] * n)
        i = j
    return np.asarray(out, dtype=np.float32)


ANCHORS = blazeface_anchors()


def blazeface_input(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> [1,128,128,3] float32 in [-1, 1] (``p / 127.5 - 1``)."""
    s = resize_bilinear(img, BF_SIZE, BF_SIZE).astype(np.float32)
    return (s / 127.5 - 1.0)[None]


def sigmoid_scores(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, np.float64).reshape(-1), -100, 100)))


def decode_blazeface(
    reg: np.ndarray, logits: np.ndarray, img_w: int, img_h: int, thresh: float = 0.4, pad: float = 0.25
) -> list[Box]:
    """Kotlin ``BlazeFaceDetector.detect`` post-processing. ``reg``: [896,16] (or [1,896,16]); ``logits``: [896(,1)]."""
    reg = np.asarray(reg, np.float32).reshape(-1, 16)
    p = sigmoid_scores(logits)
    keep = np.flatnonzero(p >= thresh)
    a = ANCHORS[keep]
    r = reg[keep]
    cx = a[:, 0] + r[:, 0] / BF_SIZE
    cy = a[:, 1] + r[:, 1] / BF_SIZE
    w = r[:, 2] / BF_SIZE * (1 + pad)
    h = r[:, 3] / BF_SIZE * (1 + pad)
    # Kotlin Float.toInt() truncates toward zero.
    ltrb = np.trunc(
        np.stack([(cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h], 1)
    )
    return [(int(b[0]), int(b[1]), int(b[2]) + 1, int(b[3]) + 1) for b in ltrb]


class BlazeFaceDetector:
    name = "face"

    def __init__(self, model_path: Path | str = FACE_MODEL, thresh: float = 0.4, pad: float = 0.25, threads: int = 2):
        from ai_edge_litert.interpreter import Interpreter

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing: run `python -m recall.privacy.assets face_detector`")
        self.thresh, self.pad = thresh, pad
        self.interp: Any = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interp.allocate_tensors()
        self._in = self.interp.get_input_details()[0]["index"]
        outs = {tuple(d["shape"]): d["index"] for d in self.interp.get_output_details()}
        if (1, 896, 16) not in outs or (1, 896, 1) not in outs:
            raise ValueError(f"{model_path}: expected outputs [1,896,16] and [1,896,1], got {list(outs)}")
        self._reg, self._cls = outs[(1, 896, 16)], outs[(1, 896, 1)]

    def raw(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Regressors [896,16] and logits [896] for one RGB image (lets callers sweep thresholds cheaply)."""
        self.interp.set_tensor(self._in, blazeface_input(img))
        self.interp.invoke()
        return self.interp.get_tensor(self._reg)[0].copy(), self.interp.get_tensor(self._cls)[0, :, 0].copy()

    def detect(self, img: np.ndarray) -> list[Box]:
        reg, logits = self.raw(img)
        return decode_blazeface(reg, logits, img.shape[1], img.shape[0], self.thresh, self.pad)


# ----------------------------------------------------------------------------------------------- screens / documents


def luma(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


class ScreenDetector:
    name = "screen"

    def __init__(self, grid_w: int = 16, grid_h: int = 9, min_mean: float = 150.0, min_edge: float = 0.10):
        self.grid_w, self.grid_h, self.min_mean, self.min_edge = grid_w, grid_h, min_mean, min_edge

    def detect(self, img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        cw, ch = w // self.grid_w, h // self.grid_h
        if cw < 3 or ch < 3:
            return []
        L = luma(img)
        edge = np.zeros((h, w), bool)
        edge[:, :-1] |= np.abs(L[:, :-1] - L[:, 1:]) > 40.0
        edge[:-1, :] |= np.abs(L[:-1, :] - L[1:, :]) > 40.0
        out: list[Box] = []
        # Kotlin iterates y in [gy*ch, (gy+1)*ch - 1) and x likewise: the last row/col of each cell is excluded.
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                ys, xs = slice(gy * ch, (gy + 1) * ch - 1), slice(gx * cw, (gx + 1) * cw - 1)
                if L[ys, xs].mean() >= self.min_mean and edge[ys, xs].mean() >= self.min_edge:
                    out.append((gx * cw, gy * ch, (gx + 1) * cw, (gy + 1) * ch))
        return out


# ----------------------------------------------------------------------------------------------- redaction


def clip_box(b: Box, w: int, h: int) -> Box | None:
    l, t, r, bt = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
    return (l, t, r, bt) if l < r and t < bt else None


def pixelate(img: np.ndarray, box: Box, block: int = 12) -> None:
    """In place. Each ``block``x``block`` tile takes the colour of the pixel at its centre (clamped to the box)."""
    rr = clip_box(box, img.shape[1], img.shape[0])
    if rr is None:
        return
    l, t, r, b = rr
    for y in range(t, b, block):
        for x in range(l, r, block):
            c = img[min(y + block // 2, b - 1), min(x + block // 2, r - 1)].copy()
            img[y : min(y + block, b), x : min(x + block, r)] = c


def block_for(box: Box) -> int:
    """Kotlin ``Redactor.blockFor``: mosaic cell scales with the (unclipped) box so large faces get coarse cells."""
    return max(6, min(box[2] - box[0], box[3] - box[1]) // 6)


def coverage_mask(boxes: Sequence[Box], w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for bx in boxes:
        c = clip_box(bx, w, h)
        if c is not None:
            m[c[1] : c[3], c[0] : c[2]] = True
    return m


class Redactor:
    """Fail closed: :meth:`redact` / :meth:`thumbnail` return ``None`` (nothing stored) when no face detector is
    configured or any detector raises."""

    def __init__(
        self,
        detectors: Sequence[SensitiveRegionDetector],
        thumb_w: int = 160,
        thumb_h: int = 90,
        require_face_detector: bool = True,
    ):
        self.detectors = list(detectors)
        self.thumb_w, self.thumb_h = thumb_w, thumb_h
        self.require_face_detector = require_face_detector

    @property
    def available(self) -> bool:
        return bool(self.detectors) and (
            not self.require_face_detector or any(d.name == "face" for d in self.detectors)
        )

    def redact_with_boxes(self, img: np.ndarray) -> tuple[np.ndarray, list[list[Box]]] | None:
        """Pixelated full-resolution copy plus the boxes from each detector (in order), or ``None``."""
        if not self.available:
            return None
        bm = np.ascontiguousarray(img[..., :3]).copy()
        per: list[list[Box]] = []
        try:
            for d in self.detectors:
                boxes = d.detect(bm)
                for r in boxes:
                    pixelate(bm, r, block_for(r))
                per.append(boxes)
        except Exception:
            return None
        return bm, per

    def redact(self, img: np.ndarray) -> np.ndarray | None:
        out = self.redact_with_boxes(img)
        return None if out is None else out[0]

    def thumbnail(self, crop: np.ndarray) -> bytes | None:
        from PIL import Image

        red = self.redact(crop)
        if red is None:
            return None
        t = resize_bilinear(red, self.thumb_w, self.thumb_h)
        buf = io.BytesIO()
        Image.fromarray(t).save(buf, format="JPEG", quality=70)
        return buf.getvalue()


def default_redactor(model_path: Path | str = FACE_MODEL) -> Redactor:
    """Same composition as ``RecallRuntime``: faces then screens; without the model the redactor is unavailable."""
    if not Path(model_path).exists():
        return Redactor([])
    return Redactor([BlazeFaceDetector(model_path), ScreenDetector()])
