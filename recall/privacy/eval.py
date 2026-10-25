"""M7 redaction-recall evaluation -> results/m7/redaction.json (evidence stage: desktop-proxy).

  PYTHONPATH=. .venv/bin/python -m recall.privacy.assets face_detector wider_val
  PYTHONPATH=. .venv/bin/python -m recall.privacy.eval            # full run
  PYTHONPATH=. .venv/bin/python -m recall.privacy.eval --limit 200 --n-synth 100 --out m7/redaction_smoke.json

Faces: WIDER FACE validation (third-person web photos, NOT egocentric). Each image goes through the exact port of the
on-device redactor (recall.privacy.redact). A ground-truth face counts as redacted if >= ``coverage`` of its box
area lies inside the union of redaction boxes. Invalid-flagged WIDER boxes and zero-size boxes are excluded from
recall but still count as "face area" when measuring over-redaction.

Screens / documents: the SYNTHETIC composite set from recall.privacy.synth (no suitable licensed labelled set).

CIs: 95% percentile bootstrap, 2000 resamples. Per-region recall uses a cluster bootstrap over images (all regions
of a resampled image move together), computed as a ratio of resampled sums - identical to
recall.bench.common.bootstrap_ci(groups=...) with ``np.mean`` but vectorised. Per-image area metrics use
recall.bench.common.bootstrap_ci directly.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from recall.bench.common import bootstrap_ci, result_header, write_result
from recall.paths import REPO
from recall.privacy import redact as rd
from recall.privacy.assets import ASSETS, FACE_MODEL, WIDER_DIR, sha256sum

KOTLIN_SOURCES = (
    "android/app/src/main/java/ai/recall/privacy/Detectors.kt",
    "android/app/src/main/java/ai/recall/privacy/Redactor.kt",
)
N_BOOT = 2000
SEED = 0
COVERAGE = 0.7
COVERAGE_SWEEP = (0.5, 0.7, 0.9)
THRESH_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.75)
HEIGHT_PX_BUCKETS = ((0, 10), (10, 50), (50, 300), (300, 100_000))  # WIDER-style absolute face height
HEIGHT_FRAC_BUCKETS = ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 10.0))  # face h / image h
AREA_FRAC_BUCKETS = ((0.0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 1.01))  # region area / image area
TEXT_PX_BUCKETS = ((0, 8), (8, 14), (14, 100))

NOT_EGOCENTRIC = (
    "WIDER FACE is third-person web photography (events, crowds, portraits), NOT egocentric video. Faces in "
    "head-worn footage are typically bystanders at conversational distance, often partially out of frame or "
    "motion-blurred; these numbers are a desktop proxy for the on-device redactor, not a measurement on Recall's "
    "target distribution."
)


# ----------------------------------------------------------------------------------------------- stats helpers


def cluster_ratio_ci(hits: np.ndarray, groups: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Mean of 0/1 ``hits`` with a 95% cluster-bootstrap CI over ``groups``."""
    hits = np.asarray(hits, float)
    if len(hits) == 0:
        return {"point": None, "ci95": [None, None], "n": 0, "n_images": 0}
    ug, inv = np.unique(groups, return_inverse=True)
    h = np.bincount(inv, weights=hits, minlength=len(ug))
    c = np.bincount(inv, minlength=len(ug)).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n)
    for s in range(0, n, 250):  # chunk to bound memory
        k = min(250, n - s)
        w = np.stack([np.bincount(rng.integers(0, len(ug), len(ug)), minlength=len(ug)) for _ in range(k)])
        boots[s : s + k] = (w @ h) / np.maximum(w @ c, 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(hits.mean()), "ci95": [float(lo), float(hi)], "n": int(len(hits)), "n_images": int(len(ug))}


def mean_ci(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, float)
    p, lo, hi = bootstrap_ci(v, np.mean, N_BOOT, SEED)
    return {"point": p, "ci95": [lo, hi], "n": int(len(v))}


def bucketed(
    hits: np.ndarray, groups: np.ndarray, key: np.ndarray, buckets: Sequence[tuple[float, float]]
) -> dict[str, Any]:
    out = {}
    for lo, hi in buckets:
        sel = (key >= lo) & (key < hi)
        out[f"[{lo},{hi})"] = cluster_ratio_ci(hits[sel], groups[sel])
    return out


def box_coverage(mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Fraction of each (l,t,r,b) box covered by ``mask``, via an integral image."""
    if len(boxes) == 0:
        return np.zeros(0)
    H, W = mask.shape
    ii = np.zeros((H + 1, W + 1), np.int64)
    ii[1:, 1:] = mask.cumsum(0).cumsum(1)
    b = np.asarray(boxes, np.int64)
    l, t = np.clip(b[:, 0], 0, W), np.clip(b[:, 1], 0, H)
    r, bt = np.clip(b[:, 2], 0, W), np.clip(b[:, 3], 0, H)
    inside = ii[bt, r] - ii[t, r] - ii[bt, l] + ii[t, l]
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1)  # off-image parts count as uncovered
    return inside / area


# ----------------------------------------------------------------------------------------------- WIDER FACE


def parse_wider(gt_file: Path) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """[(relpath, boxes (n,4) ltrb, invalid (n,) bool)] from wider_face_val_bbx_gt.txt."""
    lines = gt_file.read_text().splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        name, n = lines[i].strip(), int(lines[i + 1])
        rows = [list(map(int, ln.split()[:10])) for ln in lines[i + 2 : i + 2 + max(n, 1)]][:n]
        a = np.asarray(rows, np.int64).reshape(-1, 10)
        boxes = (
            np.stack([a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], 1) if n else np.zeros((0, 4), np.int64)
        )
        out.append((name, boxes, a[:, 7].astype(bool) if n else np.zeros(0, bool)))
        i += 2 + max(n, 1)
    return out


def eval_faces(limit: int | None, log_every: int = 250) -> dict[str, Any]:
    from PIL import Image

    gt_file = WIDER_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    img_dir = WIDER_DIR / "WIDER_val" / "images"
    if not gt_file.exists() or not img_dir.exists():
        raise FileNotFoundError(f"{gt_file} / {img_dir} missing: run `python -m recall.privacy.assets wider_val`")
    items = parse_wider(gt_file)
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]  # deterministic, spread over all 61 event classes
    det = rd.BlazeFaceDetector(FACE_MODEL)
    scr = rd.ScreenDetector()

    cov_face: list[np.ndarray] = []
    cov_full: list[np.ndarray] = []
    sweep_cov: dict[float, list[np.ndarray]] = {t: [] for t in THRESH_SWEEP}
    sweep_area: dict[float, list[float]] = {t: [] for t in THRESH_SWEEP}
    h_px, h_frac, gid = [], [], []
    area_face, area_screen, area_full, area_nonface, n_img_with_det = [], [], [], [], 0
    n_dets: list[int] = []
    t0 = time.time()
    for k, (name, boxes, invalid) in enumerate(items):
        img = np.asarray(Image.open(img_dir / name).convert("RGB"))
        H, W = img.shape[:2]
        reg, logits = det.raw(img)
        fboxes = rd.decode_blazeface(reg, logits, W, H, det.thresh, det.pad)
        # faithful pipeline: faces pixelated first, screen detector then runs on the redacted bitmap
        bm = img.copy()
        for b in fboxes:
            rd.pixelate(bm, b, rd.block_for(b))
        sboxes = scr.detect(bm)
        m_face = rd.coverage_mask(fboxes, W, H)
        m_scr = rd.coverage_mask(sboxes, W, H)
        m_full = m_face | m_scr
        gt_mask = rd.coverage_mask([tuple(b) for b in boxes], W, H)
        valid = ~invalid & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        vb = boxes[valid]
        cov_face.append(box_coverage(m_face, vb))
        cov_full.append(box_coverage(m_full, vb))
        h_px.append(vb[:, 3] - vb[:, 1])
        h_frac.append((vb[:, 3] - vb[:, 1]) / H)
        gid.append(np.full(len(vb), k))
        area_face.append(m_face.mean())
        area_screen.append(m_scr.mean())
        area_full.append(m_full.mean())
        area_nonface.append((m_full & ~gt_mask).sum() / max((~gt_mask).sum(), 1))
        n_dets.append(len(fboxes))
        n_img_with_det += bool(fboxes)
        for t in THRESH_SWEEP:
            mb = rd.coverage_mask(rd.decode_blazeface(reg, logits, W, H, t, det.pad), W, H)
            sweep_cov[t].append(box_coverage(mb, vb))
            sweep_area[t].append(mb.mean())
        if log_every and (k + 1) % log_every == 0:
            print(f"  faces {k + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    cf, cu = np.concatenate(cov_face), np.concatenate(cov_full)
    hp, hf, g = np.concatenate(h_px), np.concatenate(h_frac), np.concatenate(gid)

    def recall_block(cov: np.ndarray) -> dict[str, Any]:
        hits = cov >= COVERAGE
        return {
            "overall": cluster_ratio_ci(hits, g),
            "by_face_height_px": bucketed(hits, g, hp, HEIGHT_PX_BUCKETS),
            "by_face_height_frac_of_image": bucketed(hits, g, hf, HEIGHT_FRAC_BUCKETS),
            "mean_covered_fraction": float(cov.mean()),
        }

    return {
        "dataset": {
            "name": "WIDER FACE",
            "split": "val",
            "source": ASSETS["wider_val"][1].url,
            "sha256": {a.dest.name: a.sha256 for a in ASSETS["wider_val"]},
            "licence": "CC BY-NC-ND 4.0 (WIDER FACE; non-commercial research use). Not redistributed or committed.",
            "egocentric": False,
            "note": NOT_EGOCENTRIC,
            "n_images": len(items),
            "n_faces_valid": int(len(cf)),
            "subsampled": bool(limit),
        },
        "protocol": {
            "input": "whole WIDER image (the device redacts an event crop, see Pipeline.crop)",
            "coverage_threshold": COVERAGE,
            "invalid_boxes": "excluded from recall, included in GT face area for over-redaction",
        },
        "recall": {"face_detector": recall_block(cf), "full_redactor": recall_block(cu)},
        "recall_vs_coverage_threshold": {
            str(x): {
                "face_detector": cluster_ratio_ci(cf >= x, g),
                "full_redactor": cluster_ratio_ci(cu >= x, g),
            }
            for x in COVERAGE_SWEEP
        },
        "over_redaction": {
            "image_fraction_redacted_face_detector": mean_ci(area_face),
            "image_fraction_redacted_screen_detector": mean_ci(area_screen),
            "image_fraction_redacted_full": mean_ci(area_full),
            "non_face_area_fraction_redacted_full": mean_ci(area_nonface),
            "fraction_images_with_any_face_box": n_img_with_det / max(len(items), 1),
            "mean_face_boxes_per_image_no_nms": float(np.mean(n_dets)) if n_dets else None,
        },
        "face_threshold_sweep": [
            {
                "score_threshold": t,
                "recall": cluster_ratio_ci(np.concatenate(sweep_cov[t]) >= COVERAGE, g),
                "image_fraction_redacted": mean_ci(sweep_area[t]),
            }
            for t in THRESH_SWEEP
        ],
        "seconds": round(time.time() - t0, 1),
    }


# ----------------------------------------------------------------------------------------------- synthetic screens


def eval_screens(n_images: int, face_model: bool = True) -> dict[str, Any]:
    import hashlib

    from recall.privacy.synth import DISTRACTOR_KINDS, POSITIVE_KINDS, digest_update, make_image

    scr = rd.ScreenDetector()
    det = rd.BlazeFaceDetector(FACE_MODEL) if face_model and FACE_MODEL.exists() else None
    t0 = time.time()
    digest = hashlib.sha256()
    cov_s, cov_f, kinds, afrac, tpx, gid = [], [], [], [], [], []
    dcov, dkind = [], []
    area_full, area_nonpos, area_neg_only = [], [], []
    for k in range(n_images):
        s = make_image(SEED * 1_000_003 + k)
        digest_update(digest, s)
        H, W = s.image.shape[:2]
        bm = s.image.copy()
        fboxes = det.detect(bm) if det else []
        for b in fboxes:
            rd.pixelate(bm, b, rd.block_for(b))
        m_s = rd.coverage_mask(scr.detect(bm), W, H)
        m_f = m_s | rd.coverage_mask(fboxes, W, H)
        pos = np.zeros((H, W), bool)
        for r in s.regions:
            if r.positive:
                pos |= r.mask
                cov_s.append((m_s & r.mask).sum() / r.mask.sum())
                cov_f.append((m_f & r.mask).sum() / r.mask.sum())
                kinds.append(r.kind)
                afrac.append(r.mask.mean())
                tpx.append(r.text_px)
                gid.append(k)
            else:
                dcov.append((m_f & r.mask).sum() / r.mask.sum())
                dkind.append(r.kind)
        area_full.append(m_f.mean())
        area_nonpos.append((m_f & ~pos).sum() / max((~pos).sum(), 1))
        if not pos.any():
            area_neg_only.append(m_f.mean())

    cs, cfu = np.asarray(cov_s), np.asarray(cov_f)
    kinds_a, af, tp, g = np.asarray(kinds), np.asarray(afrac), np.asarray(tpx), np.asarray(gid)
    dc, dk = np.asarray(dcov), np.asarray(dkind)

    def block(cov: np.ndarray) -> dict[str, Any]:
        hits = cov >= COVERAGE
        return {
            "overall": cluster_ratio_ci(hits, g),
            "by_kind": {kd: cluster_ratio_ci(hits[kinds_a == kd], g[kinds_a == kd]) for kd in POSITIVE_KINDS},
            "by_area_frac_of_image": bucketed(hits, g, af, AREA_FRAC_BUCKETS),
            "by_text_height_px": bucketed(hits, g, tp, TEXT_PX_BUCKETS),
            "mean_covered_fraction": float(cov.mean()) if len(cov) else None,
        }

    return {
        "dataset": {
            "name": "Recall synthetic screen/document composites (recall/privacy/synth.py)",
            "synthetic": True,
            "egocentric": False,
            "note": "SYNTHETIC: procedural backgrounds + rendered text/UI patches with random perspective. No EPIC "
            "frames, no photographs. Measures the heuristic's behaviour on clean rendered text; real screens "
            "(glare, moire, reflections, motion blur) are harder.",
            "n_images": n_images,
            "n_positive_regions": int(len(cs)),
            "n_distractor_regions": int(len(dc)),
            "n_negative_only_images": len(area_neg_only),
            "seed": SEED,
            "digest_sha256": digest.hexdigest(),
            "face_detector_in_pipeline": det is not None,
        },
        "protocol": {"coverage_threshold": COVERAGE, "gt": "pixel mask of the warped display / paper / label area"},
        "recall": {"screen_detector": block(cs), "full_redactor": block(cfu)},
        "recall_vs_coverage_threshold": {str(x): cluster_ratio_ci(cfu >= x, g) for x in COVERAGE_SWEEP},
        "over_redaction": {
            "image_fraction_redacted_full": mean_ci(area_full),
            "non_sensitive_area_fraction_redacted_full": mean_ci(area_nonpos),
            "image_fraction_redacted_negative_only_images": mean_ci(area_neg_only),
            "distractor_redacted_at_coverage": {
                kd: {
                    **mean_ci((dc[dk == kd] >= COVERAGE).astype(float)),
                    "mean_covered_fraction": float(dc[dk == kd].mean()) if (dk == kd).any() else None,
                }
                for kd in DISTRACTOR_KINDS
            },
        },
        "seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate a deterministic subset of WIDER val")
    ap.add_argument("--n-synth", type=int, default=1000)
    ap.add_argument("--out", default="m7/redaction.json")
    args = ap.parse_args(argv)
    import ai_edge_litert
    import PIL

    config = {
        "face_detector": {
            "model": "MediaPipe BlazeFace short-range (float16, v1)",
            "url": ASSETS["face_detector"][0].url,
            "sha256": sha256sum(FACE_MODEL),
            "licence": ASSETS["face_detector"][0].licence,
            "score_threshold": 0.4,
            "box_padding": 0.25,
            "nms": False,
        },
        "screen_detector": {"grid": [16, 9], "min_mean_luma": 150.0, "min_edge_density": 0.10, "edge_step": 40.0},
        "pixelate_block_px": "max(6, min(box_w, box_h) // 6)",
        "thumbnail": [160, 90],
        "coverage_threshold": COVERAGE,
        "n_bootstrap": N_BOOT,
        # sha256 of the Kotlin sources this port mirrors, so a later Kotlin change is detectable
        "port_of": {k: sha256sum(REPO / k) if (REPO / k).exists() else None for k in KOTLIN_SOURCES},
        "lib_versions": {"ai_edge_litert": getattr(ai_edge_litert, "__version__", "2.2.0"), "pillow": PIL.__version__},
        "limit": args.limit,
        "n_synth": args.n_synth,
    }
    payload = result_header("desktop-proxy", SEED, config)
    print("faces (WIDER FACE val) ...", flush=True)
    payload["faces"] = eval_faces(args.limit)
    print("screens/documents (synthetic) ...", flush=True)
    payload["screens_documents"] = eval_screens(args.n_synth)
    p = write_result(args.out, payload)
    print(f"wrote {p}")
    return p


if __name__ == "__main__":
    main()
