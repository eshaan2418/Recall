"""Reproducible SYNTHETIC composite set for screen / document redaction recall.

No public labelled set of egocentric screens/documents with region masks was available under a licence we could
use, so every image here is generated from a seed: procedural backgrounds (low-frequency colour fields, muted
clutter shapes, wood-like stripes) plus text-bearing patches warped with a random perspective. Nothing is read
from EPIC-KITCHENS or any photo set; nothing is written to disk by default.

Positive patch kinds (ground truth = warped patch mask):
  document      cream/white paper, dark text lines
  screen_light  light UI (header bar, text, buttons) inside a dark bezel; GT = display area
  screen_dark   dark-mode UI / code (light text on near-black) inside a bezel; GT = display area
  label         small product label: mid-saturation colour, dark text (text on objects)

Distractors (no text, should NOT be redacted; used for over-redaction):
  tiles         bright wall tiles with dark grout lines
  plate         bright plain ellipse
  pattern       bright packaging-like graphic (stripes / dots), no glyphs
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

POSITIVE_KINDS = ("document", "screen_light", "screen_dark", "label")
DISTRACTOR_KINDS = ("tiles", "plate", "pattern")
SIZES = ((640, 360), (512, 512), (480, 640), (800, 450))


@dataclass
class Region:
    kind: str
    mask: np.ndarray  # HxW bool
    text_px: float = 0.0  # nominal glyph height in output pixels (0 for distractors)
    positive: bool = True


@dataclass
class SynthImage:
    image: np.ndarray  # HxWx3 uint8
    regions: list[Region] = field(default_factory=list)


_FONTS: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if px not in _FONTS:
        _FONTS[px] = ImageFont.load_default(size=px)  # Pillow's bundled font: identical on every machine
    return _FONTS[px]


def _words(rng: np.random.Generator, n: int) -> str:
    letters = string.ascii_letters + string.digits
    return " ".join("".join(rng.choice(list(letters), size=int(rng.integers(2, 10)))) for _ in range(n))


def _background(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    # low-frequency colour field
    g = rng.uniform(40, 200, size=(4, 6, 3)).astype(np.uint8)
    bg = Image.fromarray(g).resize((w, h), Image.Resampling.BICUBIC)
    d = ImageDraw.Draw(bg)
    if rng.random() < 0.5:  # wood-like stripes
        base = rng.uniform(80, 160, 3)
        arr = np.asarray(bg).astype(np.float32)
        yy = np.arange(h)[:, None] + 8 * np.sin(np.arange(w)[None, :] / rng.uniform(20, 60))
        stripes = 0.5 + 0.5 * np.sin(yy / rng.uniform(3, 9))
        arr = 0.5 * arr + 0.5 * (base[None, None, :] * (0.8 + 0.3 * stripes[..., None]))
        bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(bg)
    for _ in range(int(rng.integers(3, 12))):  # muted clutter
        c = tuple(int(v) for v in rng.uniform(30, 170, 3))
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(10, w // 3)), y0 + int(rng.integers(10, h // 3))
        (d.rectangle if rng.random() < 0.5 else d.ellipse)([x0, y0, x1, y1], fill=c)
    return bg.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0))))


def _text_block(
    d: ImageDraw.ImageDraw, rng: np.random.Generator, box: tuple[int, int, int, int], px: int, color
) -> None:
    """Lines of random words inside ``box``; glyphs are clipped to the box (drawn on a layer, then pasted)."""
    x0, y0, x1, y1 = box
    if x1 - x0 < 4 or y1 - y0 < px:
        return
    layer = Image.new("L", (x1 - x0, y1 - y0), 0)
    ld = ImageDraw.Draw(layer)
    f = _font(px)
    y = 0
    while y + px < y1 - y0:
        ld.text((0, y), _words(rng, int(rng.integers(2, 9))), font=f, fill=255)
        y += int(px * rng.uniform(1.3, 1.8))
    d.bitmap((x0, y0), layer, fill=color)


def _patch(kind: str, rng: np.random.Generator, pw: int, ph: int, px: int) -> tuple[Image.Image, Image.Image]:
    """Returns (RGBA patch, L mask of the ground-truth sensitive area within the patch)."""
    p = Image.new("RGBA", (pw, ph), (0, 0, 0, 255))
    gt = Image.new("L", (pw, ph), 255)
    d = ImageDraw.Draw(p)
    if kind == "document":
        paper = tuple(int(v) for v in rng.uniform(225, 255, 3))
        d.rectangle([0, 0, pw, ph], fill=(*paper, 255))
        m = max(2, pw // 12)
        _text_block(d, rng, (m, m, pw - m, ph - m), px, (int(rng.uniform(0, 60)),) * 3)
    elif kind in ("screen_light", "screen_dark"):
        bez = max(2, min(pw, ph) // 14)
        d.rectangle([0, 0, pw, ph], fill=(int(rng.uniform(10, 40)),) * 3 + (255,))
        light = kind == "screen_light"
        bgc = tuple(int(v) for v in (rng.uniform(235, 255, 3) if light else rng.uniform(15, 45, 3)))
        fgc = (int(rng.uniform(0, 70)),) * 3 if light else tuple(int(v) for v in rng.uniform(170, 240, 3))
        d.rectangle([bez, bez, pw - bez, ph - bez], fill=(*bgc, 255))
        hb = max(3, (ph - 2 * bez) // 8)
        d.rectangle([bez, bez, pw - bez, bez + hb], fill=(*tuple(int(v) for v in rng.uniform(30, 220, 3)), 255))
        _text_block(d, rng, (bez + 4, bez + hb + 4, pw - bez - 4, ph - bez - 4), px, fgc)
        for _ in range(int(rng.integers(0, 3))):
            bx = int(rng.integers(bez, max(bez + 1, pw - bez - 30)))
            by = int(rng.integers(bez + hb, max(bez + hb + 1, ph - bez - 12)))
            d.rectangle([bx, by, bx + 30, by + 12], fill=(*tuple(int(v) for v in rng.uniform(40, 220, 3)), 255))
        gt = Image.new("L", (pw, ph), 0)
        ImageDraw.Draw(gt).rectangle([bez, bez, pw - bez - 1, ph - bez - 1], fill=255)
    elif kind == "label":
        c = tuple(int(v) for v in rng.uniform(90, 230, 3))
        d.rectangle([0, 0, pw, ph], fill=(*c, 255))
        _text_block(d, rng, (2, 2, pw - 2, ph - 2), px, (int(rng.uniform(0, 50)),) * 3)
    elif kind == "tiles":
        c = tuple(int(v) for v in rng.uniform(200, 250, 3))
        d.rectangle([0, 0, pw, ph], fill=(*c, 255))
        step = int(rng.integers(14, 40))
        grout = (int(rng.uniform(60, 140)),) * 3 + (255,)
        for x in range(0, pw, step):
            d.line([x, 0, x, ph], fill=grout, width=2)
        for y in range(0, ph, step):
            d.line([0, y, pw, y], fill=grout, width=2)
    elif kind == "plate":
        p = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        ImageDraw.Draw(p).ellipse([0, 0, pw - 1, ph - 1], fill=(int(rng.uniform(215, 250)),) * 3 + (255,))
    elif kind == "pattern":
        c = tuple(int(v) for v in rng.uniform(190, 245, 3))
        d.rectangle([0, 0, pw, ph], fill=(*c, 255))
        c2 = (*tuple(int(v) for v in rng.uniform(40, 200, 3)), 255)
        if rng.random() < 0.5:
            s = int(rng.integers(6, 20))
            for x in range(-ph, pw, s * 2):
                d.polygon([(x, 0), (x + s, 0), (x + s + ph, ph), (x + ph, ph)], fill=c2)
        else:
            s = int(rng.integers(8, 24))
            for y in range(s // 2, ph, s):
                for x in range(s // 2, pw, s):
                    d.ellipse([x - s // 4, y - s // 4, x + s // 4, y + s // 4], fill=c2)
    else:
        raise ValueError(kind)
    return p, gt


def _perspective_coeffs(dst: np.ndarray, src: np.ndarray) -> tuple[float, ...]:
    """PIL PERSPECTIVE data mapping output (dst) coords -> input (src) coords."""
    A, b = [], []
    for (x, y), (u, v) in zip(dst, src):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b += [u, v]
    return tuple(float(c) for c in np.linalg.solve(np.asarray(A, float), np.asarray(b, float)))


def _place(rng: np.random.Generator, w: int, h: int, area_frac: float, aspect: float) -> tuple[np.ndarray, int, int]:
    pw = int(np.sqrt(area_frac * w * h * aspect))
    ph = int(np.sqrt(area_frac * w * h / aspect))
    pw, ph = max(pw, 12), max(ph, 8)
    cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    ang = np.deg2rad(rng.uniform(-30, 30))
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    base = base + rng.uniform(-0.12, 0.12, size=(4, 2)) * [pw, ph]  # perspective jitter
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return base @ rot.T + [cx, cy], pw, ph


def _composite(canvas: Image.Image, patch: Image.Image, gt: Image.Image, dst: np.ndarray) -> np.ndarray:
    pw, ph = patch.size
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], float)
    coeffs = _perspective_coeffs(dst, src)
    W, H = canvas.size
    warped = patch.transform((W, H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
    gt_full = Image.new("L", (pw, ph), 0)
    gt_full.paste(gt, (0, 0), patch.getchannel("A"))
    wgt = gt_full.transform((W, H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
    canvas.paste(warped, (0, 0), warped)
    return np.asarray(wgt) > 127


def make_image(seed: int) -> SynthImage:
    rng = np.random.default_rng(seed)
    w, h = SIZES[int(rng.integers(len(SIZES)))]
    canvas = _background(rng, w, h).convert("RGB")
    placed: list[tuple[str, np.ndarray, float, bool]] = []
    # 15% negatives-only images; otherwise 1-2 positives and 0-2 distractors
    n_pos = 0 if rng.random() < 0.15 else int(rng.integers(1, 3))
    n_neg = int(rng.integers(0, 3)) if n_pos else int(rng.integers(1, 3))
    jobs = [(str(rng.choice(POSITIVE_KINDS)), True) for _ in range(n_pos)]
    jobs += [(str(rng.choice(DISTRACTOR_KINDS)), False) for _ in range(n_neg)]
    rng.shuffle(jobs)  # type: ignore[arg-type]
    for kind, pos in jobs:
        frac = float(np.exp(rng.uniform(np.log(0.004), np.log(0.03 if kind == "label" else 0.35))))
        aspect = float(rng.uniform(0.6, 1.8)) if kind != "screen_light" or rng.random() < 0.5 else 0.5
        dst, pw, ph = _place(rng, w, h, frac, aspect)
        px = int(np.clip(rng.uniform(5, 22), 5, max(5, ph // 3)))
        patch, gt = _patch(kind, rng, pw, ph, px)
        m = _composite(canvas, patch, gt, dst)
        # later patches occlude earlier ones
        placed = [(k, pm & ~m, t, p) for k, pm, t, p in placed]
        placed.append((kind, m, float(px) if pos else 0.0, pos))
    # camera-ish degradation: slight blur, noise, exposure
    canvas = canvas.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.0, 1.0))))
    arr = np.asarray(canvas).astype(np.float32) * rng.uniform(0.75, 1.15) + rng.normal(0, rng.uniform(1, 6), (h, w, 3))
    img = np.clip(arr, 0, 255).astype(np.uint8)
    regions = [Region(k, pm, t, p) for k, pm, t, p in placed if pm.any()]
    return SynthImage(img, regions)


def digest_update(hsh: Any, s: SynthImage) -> None:
    """Feed pixels + masks into a sha256 so a reader can confirm they regenerated the identical set."""
    hsh.update(s.image.tobytes())
    for r in s.regions:
        hsh.update(r.kind.encode())
        hsh.update(np.packbits(r.mask).tobytes())


def dataset_digest(images: list[SynthImage]) -> str:
    hsh = hashlib.sha256()
    for s in images:
        digest_update(hsh, s)
    return hsh.hexdigest()
