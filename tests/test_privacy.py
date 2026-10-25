"""M7 privacy layer: Python port of the on-device redactor + evaluation helpers. Tiny synthetic inputs only."""

from __future__ import annotations

import numpy as np
import pytest

from recall.privacy import redact as rd
from recall.privacy.assets import ASSETS
from recall.privacy.eval import box_coverage, cluster_ratio_ci, parse_wider
from recall.privacy.synth import POSITIVE_KINDS, dataset_digest, make_image


def noise(w: int, h: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


class Fixed:
    def __init__(self, name: str, boxes: list[rd.Box]):
        self.name, self.boxes = name, boxes

    def detect(self, img: np.ndarray) -> list[rd.Box]:
        return self.boxes


class Throwing:
    name = "face"

    def detect(self, img: np.ndarray) -> list[rd.Box]:
        raise RuntimeError("boom")


# ---------------------------------------------------------------- mirrors android/.../PrivacyTest.kt


def test_pixelate_destroys_detail_inside_box_only():
    p = noise(64, 48)
    orig = p.copy()
    rd.pixelate(p, (10, 10, 40, 34), 8)
    for y in range(10, 34):
        for x in range(10, 40):
            cx, cy = 10 + (x - 10) // 8 * 8, 10 + (y - 10) // 8 * 8
            assert (p[y, x] == p[cy, cx]).all()
    outside = np.ones((48, 64), bool)
    outside[10:34, 10:40] = False
    assert (p[outside] == orig[outside]).all()
    assert len({tuple(c) for c in p[10:34, 10:40].reshape(-1, 3)}) <= 16


def test_pixelate_clips_out_of_bounds():
    p = noise(20, 20)
    before = p.copy()
    rd.pixelate(p, (-5, -5, 100, 100), 4)  # must not raise
    q = p.copy()
    rd.pixelate(p, (30, 30, 40, 40), 4)  # fully outside: no-op
    assert (p == q).all() and not (p == before).all()


def test_block_size_scales_with_box():
    assert rd.block_for((0, 0, 20, 20)) == 6
    assert rd.block_for((0, 0, 300, 120)) == 20


def test_redactor_fails_closed():
    img = noise(32, 32)
    assert not rd.Redactor([]).available
    assert rd.Redactor([]).thumbnail(img) is None
    assert rd.Redactor([rd.ScreenDetector()]).redact(img) is None, "screen heuristic alone is not enough"
    assert rd.Redactor([Throwing()]).redact(img) is None, "detector failure -> no thumbnail"
    out = rd.Redactor([Fixed("face", [(0, 0, 16, 16)])]).redact(img)
    assert out is not None
    assert (img == noise(32, 32)).all(), "input not modified"
    assert (out[0, :16] != img[0, :16]).any(), "face region changed"
    assert (out[20:, 20:] == img[20:, 20:]).all()


def test_default_redactor_without_model_is_unavailable(tmp_path):
    assert not rd.default_redactor(tmp_path / "missing.tflite").available


def test_thumbnail_is_small_jpeg():
    img = np.tile(np.linspace(0, 255, 320, dtype=np.uint8)[None, :, None], (180, 1, 3))
    jpg = rd.Redactor([Fixed("face", [(40, 40, 120, 140)])]).thumbnail(img)
    assert jpg is not None and jpg[:2] == b"\xff\xd8" and len(jpg) <= 4096


def test_screen_heuristic_flags_bright_text_and_not_flat_or_dark():
    w, h = 160, 90
    y, x = np.mgrid[0:h, 0:w]
    stroke = (x // 2 + y // 3) % 3 == 0
    img = np.where(x < 80, np.where(stroke, 0, 255), 128).astype(np.uint8)[..., None].repeat(3, 2)
    boxes = rd.ScreenDetector().detect(img)
    assert boxes and all(b[2] <= 80 for b in boxes)
    assert rd.ScreenDetector().detect(np.full((h, w, 3), 16, np.uint8)) == []
    assert rd.ScreenDetector().detect(np.zeros((20, 20, 3), np.uint8)) == []  # cells < 3 px


def test_blazeface_anchors_and_decode():
    a = rd.blazeface_anchors()
    assert a.shape == (896, 2)
    assert a[0, 0] == pytest.approx(0.5 / 16) and a[512, 0] == pytest.approx(0.5 / 8)
    reg = np.zeros((896, 16), np.float32)
    sc = np.full(896, -10.0, np.float32)
    k = 512 + 6 * (4 * 8 + 4)  # anchor at cell (4,4) of the 8x8 map: centre (0.5625, 0.5625)
    sc[k] = 5.0
    reg[k, 2] = reg[k, 3] = 32.0  # 0.25 of the image
    boxes = rd.decode_blazeface(reg, sc, 400, 200, 0.4, 0.0)
    assert len(boxes) == 1
    l, t, r, b = boxes[0]
    assert l == 175 and abs(t - 87) <= 1 and r in (275, 276) and b in (137, 138)
    # padding grows the box symmetrically; threshold gates on sigmoid(score)
    lp, _, rp, _ = rd.decode_blazeface(reg, sc, 400, 200, 0.4, 0.25)[0]
    assert lp < l and rp > r
    assert rd.decode_blazeface(reg, sc, 400, 200, 0.999, 0.0) == []


def test_resize_bilinear():
    img = noise(40, 30)
    assert (rd.resize_bilinear(img, 40, 30) == img).all()
    flat = np.full((50, 70, 3), 77, np.uint8)
    out = rd.resize_bilinear(flat, 128, 128)
    assert out.shape == (128, 128, 3) and (out == 77).all()
    x = rd.blazeface_input(img)
    assert x.shape == (1, 128, 128, 3) and x.dtype == np.float32 and x.min() >= -1 and x.max() <= 1


# ---------------------------------------------------------------- evaluation helpers


def test_box_coverage_matches_brute_force():
    rng = np.random.default_rng(1)
    m = rng.random((37, 53)) < 0.4
    boxes = np.array([[0, 0, 53, 37], [5, 6, 20, 30], [40, 30, 60, 45], [10, 10, 11, 11]])
    got = box_coverage(m, boxes)
    for b, g in zip(boxes, got):
        area = (b[2] - b[0]) * (b[3] - b[1])
        assert g == pytest.approx(m[b[1] : b[3], b[0] : b[2]].sum() / area)


def test_cluster_ratio_ci():
    hits = np.array([1, 1, 0, 0, 1, 1, 1, 0], float)
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    r = cluster_ratio_ci(hits, g, n=500)
    assert r["point"] == pytest.approx(5 / 8) and r["n_images"] == 4
    assert r["ci95"][0] <= r["point"] <= r["ci95"][1]
    assert cluster_ratio_ci(np.ones(5), np.arange(5), n=100)["ci95"] == [1.0, 1.0]
    assert cluster_ratio_ci(np.zeros(0), np.zeros(0))["n"] == 0


def test_parse_wider(tmp_path):
    f = tmp_path / "gt.txt"
    f.write_text(
        "a/1.jpg\n2\n10 20 30 40 0 0 0 0 0 0 \n1 1 5 5 2 0 0 1 0 0 \nb/2.jpg\n0\n0 0 0 0 0 0 0 0 0 0 \n"
        "c/3.jpg\n1\n3 4 5 6 0 0 0 0 0 0 \n"
    )
    items = parse_wider(f)
    assert [n for n, _, _ in items] == ["a/1.jpg", "b/2.jpg", "c/3.jpg"]
    assert items[0][1].tolist() == [[10, 20, 40, 60], [1, 1, 6, 6]] and items[0][2].tolist() == [False, True]
    assert items[1][1].shape == (0, 4)
    assert items[2][1].tolist() == [[3, 4, 8, 10]]


def test_synthetic_set_is_reproducible_and_labelled():
    a = [make_image(i) for i in range(4)]
    b = [make_image(i) for i in range(4)]
    assert dataset_digest(a) == dataset_digest(b)
    assert dataset_digest(a) != dataset_digest([make_image(i) for i in range(1, 5)])
    kinds = set()
    for s in a:
        assert s.image.dtype == np.uint8 and s.image.ndim == 3
        for r in s.regions:
            assert r.mask.shape == s.image.shape[:2] and r.mask.any()
            assert r.positive == (r.kind in POSITIVE_KINDS)
            kinds.add(r.kind)
    assert kinds


def test_assets_are_pinned():
    for assets in ASSETS.values():
        for a in assets:
            assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0
            assert "latest" not in a.url and a.url.startswith("https://")


# ---------------------------------------------------------------- real model (skipped without it)


@pytest.mark.slow
def test_blazeface_model_runs():
    pytest.importorskip("ai_edge_litert")
    if not rd.FACE_MODEL.exists():
        pytest.skip("face_detector.tflite not downloaded")
    det = rd.BlazeFaceDetector()
    reg, logits = det.raw(np.full((90, 160, 3), 128, np.uint8))
    assert reg.shape == (896, 16) and logits.shape == (896,)
    assert det.detect(np.full((90, 160, 3), 128, np.uint8)) == []


def test_privacy_readme_matches_results():
    """privacy/README.md is generated from results/m7/redaction.json; fail if someone edits one without the other."""
    from recall.privacy import report

    if not report.RESULT.exists():
        pytest.skip("results/m7/redaction.json not generated")
    assert report.main(["--check"]) == 0
